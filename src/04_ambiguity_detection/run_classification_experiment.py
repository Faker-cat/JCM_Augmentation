# 情報欠損文章の検出実験スクリプト (最小限のハイブリッド版)
import argparse  # ★ [追加]: 引数解析のため
import os

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 設定 (旧コードのグローバル定数を維持) ---
# 使用するモデルID (Hugging Face)
MODEL_ID = "llm-jp/llm-jp-3.1-13b-instruct4"

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "04_ambiguity_detection")
GT_FILE_PATH = os.path.join(DATA_DIR, "JCM_random_1000_sample_evaluated.csv")
MODERATION_RESULTS_PATH = os.path.join(
    BASE_DIR, "data", "02_method_moderation", "moderation_full_results.csv"
)
# 旧コードのOUTPUT_DIRは削除。代わりにargs.output_dirを使用

# 生成パラメータ
MAX_NEW_TOKENS = 16
TEMPERATURE = 0.1  # 決定論的にするため低めに設定


def load_prompt_template(prompt_path):
    """プロンプトファイルを読み込む（新コードから流用）"""
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"エラー: プロンプトファイルが見つかりません: {prompt_path}")
        exit(1)


def load_model():
    """
    モデルとトークナイザーを読み込む（旧コードと同じ: グローバルなMODEL_IDを使用）
    """
    print(f"Loading model: {MODEL_ID} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        exit(1)


def load_and_merge_data():
    """GTデータとModeration結果をIDベースで統合する（旧コードと同じ）"""
    # ... (変更なし) ...
    print(f"Loading GT data from: {GT_FILE_PATH}")
    if not os.path.exists(GT_FILE_PATH):
        raise FileNotFoundError(
            f"GTファイルが見つかりません: {GT_FILE_PATH}\nパスやファイル名を確認してください。"
        )
    df_gt = pd.read_csv(GT_FILE_PATH)
    if "Aのフラグ" in df_gt.columns:
        df_gt["gt_is_ambiguous"] = df_gt["Aのフラグ"].map(
            lambda x: True if str(x).strip().upper() == "TRUE" else False
        )
    else:
        raise ValueError("GTファイルに 'Aのフラグ' カラムが見つかりません。")

    print(f"Loading Moderation results from: {MODERATION_RESULTS_PATH}")
    if not os.path.exists(MODERATION_RESULTS_PATH):
        raise FileNotFoundError(
            f"Moderation結果ファイルが見つかりません: {MODERATION_RESULTS_PATH}"
        )
    df_mod = pd.read_csv(MODERATION_RESULTS_PATH)
    print("Merging data...")
    df_gt["Original_ID"] = pd.to_numeric(df_gt["Original_ID"], errors="coerce")
    df_mod["ID"] = pd.to_numeric(df_mod["ID"], errors="coerce")
    df_merged = pd.merge(
        df_gt,
        df_mod[["ID", "original_label", "moderation_flagged"]],
        left_on="Original_ID",
        right_on="ID",
        how="left",
    )
    df_merged["original_label"] = df_merged["original_label"].fillna(0)
    df_merged["moderation_flagged"] = df_merged["moderation_flagged"].fillna(False)
    return df_merged


def check_phase1_filtering(row):
    """Phase 1: Moderationによるフィルタリング（旧コードと同じ）"""
    original_label = row["original_label"]
    moderation_flagged = row["moderation_flagged"]
    if isinstance(moderation_flagged, str):
        moderation_flagged = moderation_flagged.lower() == "true"
    if (original_label == 1) and moderation_flagged:
        return True
    return False


def generate_prompt(prompt_template, text, label):  # ★ label引数を追加
    """LLM-JP用のプロンプト作成。テンプレートに文章とラベルを埋め込む。"""

    # 0/1 の数値ラベルを "許容できる" / "許容できない" に変換
    label_text = "許容できる" if label == 0 else "許容できない"

    # テンプレート内の "{text}」" と "{label}" を対象の文章/ラベルに置換する
    instruction = prompt_template.replace("「{text}」", f"「{text}」")
    instruction = instruction.replace("{label}", label_text)  # ★ ラベルを埋め込む

    # LLM-JP系の標準的なプロンプトフォーマット
    prompt = f"### 指示:\n{instruction}\n\n### 応答:\n"
    return prompt


def get_llm_prediction(
    model, tokenizer, prompt_template, text, label
):  # ★ label引数を追加
    """モデルを用いた推論実行"""

    prompt = generate_prompt(prompt_template, text, label)  # ★ labelを渡す

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
        model.device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,  # ★ グローバル定数を使用
            do_sample=False,  # ★ グローバル定数を使用 (TEMPERATURE=0.1のため)
            temperature=TEMPERATURE,  # ★ グローバル定数を使用
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.replace(prompt, "").strip()

    if "### 応答:" in response:
        response = response.split("### 応答:")[-1].strip()

    upper_res = response.upper()
    if "TRUE" in upper_res:
        return True, response
    elif "FALSE" in upper_res:
        return False, response
    else:
        # 解析不能時はFalse扱い
        return False, response


def run_experiment(args):  # ★ 引数argsを受け取る
    """実験の実行と結果の集計"""
    print(
        f"=== 実験開始: LLM Ambiguity Detection (Model: {MODEL_ID}) ==="
    )  # ★ グローバル定数を使用

    # 1. パスの設定とプロンプトの読み込み
    # ★ 外部から読み込んだプロンプトテンプレートを使用
    prompt_template = load_prompt_template(args.prompt_path)

    # 結果の保存先パスは data/04_ambiguity_detection/results/サブフォルダ名 となる
    # ★ args.output_dirを使用
    full_output_dir = os.path.join(DATA_DIR, "results", args.output_dir)

    # 2. モデルとデータのロード
    model, tokenizer = load_model()  # ★ 引数なし
    df = load_and_merge_data()
    print(f"対象データ数: {len(df)}件")

    results = []

    # 3. 推論実行
    print("推論実行中...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row["sent"]
        label = row["original_label"]  # ★ original_labelを取得

        is_target_c = check_phase1_filtering(row)

        raw_response = ""
        if is_target_c:
            prediction = False
            phase = "Phase 1 (Filter)"
            raw_response = "Filtered by Moderation"
        else:
            # Phase 2 (LLM)
            prediction, raw_response = get_llm_prediction(
                model, tokenizer, prompt_template, text, label
            )  # ★ labelを渡す
            phase = "Phase 2 (LLM)"

        results.append(
            {
                "Original_ID": row["Original_ID"],
                "sent": text,
                "gt_is_ambiguous": row["gt_is_ambiguous"],
                "pred_is_ambiguous": prediction,
                "phase": phase,
                "llm_raw_response": raw_response,
            }
        )

    # 4. 結果集計 (レポート生成はargsに依存しない)
    df_res = pd.DataFrame(results)

    # --- ★追加: Phase 1 フィルタリングの評価指標の計算 ★ ---
    df_phase1_filtered = df_res[df_res["phase"] == "Phase 1 (Filter)"]
    total_filtered = len(df_phase1_filtered)

    # GTで「情報欠損あり」(True) と判定されたが、Phase 1で排除された事例の数 (Filter Miss)
    # GT_is_ambiguous (True) の合計を計算
    filtered_but_ambiguous = df_phase1_filtered["gt_is_ambiguous"].sum()

    # 排除された中で、GTで情報十分だった割合 (Filter Success Rate: 排除の成功度)
    if total_filtered > 0:
        filter_success_rate = (total_filtered - filtered_but_ambiguous) / total_filtered
    else:
        filter_success_rate = 0.0

    # 排除された中で、GTで曖昧だった割合 (Filter Risk Rate: 排除の危険度)
    if total_filtered > 0:
        filter_risk_rate = filtered_but_ambiguous / total_filtered
    else:
        filter_risk_rate = 0.0
    # --- ★追加ここまで ★ ---

    y_true = df_res["gt_is_ambiguous"]
    y_pred = df_res["pred_is_ambiguous"]
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    tp, fn = cm[0]
    fp, tn = cm[1]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=True, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=True, zero_division=0)

    # 5. レポート作成・保存
    report = []
    report.append("=== 実験結果レポート ===")
    report.append(f"Model: {MODEL_ID}")
    report.append(f"Prompt: {os.path.basename(args.prompt_path)}")  # ★ argsを使用
    report.append(f"Temperature: {TEMPERATURE}")  # ★ グローバル定数を使用
    report.append(f"Total Samples: {len(df)}")
    report.append(
        f"Phase 1 Filtered: {len(df_res[df_res['phase'] == 'Phase 1 (Filter)'])}"
    )
    report.append("-" * 30)

    # --- ★レポートにPhase 1の評価指標を追加 ★ ---
    report.append("Phase 1 Filtering Analysis:")
    report.append(f" Total Samples Filtered (Phase 1): {total_filtered}")
    report.append(
        f" Ambiguous Samples among Filtered (GT=True): {filtered_but_ambiguous} (見逃し)"
    )
    report.append(
        f" Filter Success Rate (GT=False among filtered): {filter_success_rate:.4f}"
    )
    report.append(f" Filter Risk Rate (GT=True among filtered): {filter_risk_rate:.4f}")
    report.append("-" * 30)
    # --- ★追加ここまで ★ ---

    report.append("Confusion Matrix (Target: Ambiguous/True):")
    report.append(f" TP: {tp}, FN: {fn}")
    report.append(f" FP: {fp}, TN: {tn}")
    report.append("-" * 30)
    report.append(f"Accuracy : {accuracy}")
    report.append(f"Precision: {precision}")
    report.append(f"Recall   : {recall}")
    report.append(f"F1 Score : {f1}")

    report_text = "\n".join(report)
    print("\n" + report_text)

    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    output_csv = os.path.join(full_output_dir, "ambiguity_detection_results_llmjp.csv")
    df_res.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n結果CSVを保存しました: {output_csv}")

    report_path = os.path.join(full_output_dir, "experiment_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"結果レポートを保存しました: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLMを用いた情報欠損文章の自動検出実験 (最小限の引数)"
    )

    # ★ 必須引数はプロンプトパスと出力ディレクトリのみ
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="使用するプロンプトファイルのパス",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="結果を保存するサブディレクトリ名 (data/04_ambiguity_detection/results/ の下に作成されます)",
    )

    args = parser.parse_args()
    run_experiment(args)
