# 情報欠損文章の検出実験スクリプト (統合版)
import argparse  # ★追加
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

# --- 静的な設定 (引数で上書き可能) ---
# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "04_ambiguity_detection")
MODERATION_RESULTS_PATH = os.path.join(
    BASE_DIR, "data", "02_method_moderation", "moderation_full_results.csv"
)

# 生成パラメータのデフォルト値
DEFAULT_MODEL_ID = "llm-jp/llm-jp-3.1-13b-instruct4"
MAX_NEW_TOKENS = 16


def load_prompt_template(prompt_path):
    """プロンプトファイルを読み込む"""
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            # テンプレート全体を読み込む
            return f.read().strip()
    except FileNotFoundError:
        print(f"エラー: プロンプトファイルが見つかりません: {prompt_path}")
        exit(1)


def load_model(model_id):
    """モデルとトークナイザーを読み込む"""
    print(f"Loading model: {model_id} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # low_cpu_mem_usage=True,
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        exit(1)


def load_and_merge_data(gt_file_path):
    """GTデータとModeration結果をIDベースで統合する"""
    print(f"Loading GT data from: {gt_file_path}")
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError(
            f"GTファイルが見つかりません: {gt_file_path}\nパスやファイル名を確認してください。"
        )

    df_gt = pd.read_csv(gt_file_path)
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
    """Phase 1: Moderationによるフィルタリング (Target C判定)"""
    original_label = row["original_label"]
    moderation_flagged = row["moderation_flagged"]

    if isinstance(moderation_flagged, str):
        moderation_flagged = moderation_flagged.lower() == "true"

    # Target C (NGかつFlagged) は「情報十分(False)」とみなし、LLM推論をスキップする
    if (original_label == 1) and moderation_flagged:
        return True
    return False


def generate_prompt(prompt_template, text):
    """LLM-JP用のプロンプト作成。テンプレートに文章を埋め込む。"""
    # テンプレート内の "{text}" を対象の文章に置換する
    instruction = prompt_template.replace("「{text}」", f"「{text}」")

    # LLM-JP系の標準的なプロンプトフォーマット
    prompt = f"### 指示:\n{instruction}\n\n### 応答:\n"
    return prompt


def get_llm_prediction(model, tokenizer, prompt_template, text, temperature):
    """モデルを用いた推論実行"""
    prompt = generate_prompt(prompt_template, text)

    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
        model.device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            # temperatureが0.0より大きい場合にdo_sampleを有効にする
            do_sample=(temperature > 0.0),
            temperature=temperature,
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


def run_experiment(args):
    """実験の実行と結果の集計"""
    print(f"=== 実験開始: LLM Ambiguity Detection (Model: {args.model_id}) ===")

    # 1. パスの設定とプロンプトの読み込み
    gt_file_path = os.path.join(DATA_DIR, args.gt_file)
    # 結果の保存先パスは data/04_ambiguity_detection/results/サブフォルダ名 となる
    full_output_dir = os.path.join(DATA_DIR, "results", args.output_dir)

    prompt_template = load_prompt_template(args.prompt_path)

    # 2. モデルとデータのロード
    model, tokenizer = load_model(args.model_id)
    df = load_and_merge_data(gt_file_path)
    print(f"対象データ数: {len(df)}件")

    results = []

    # 3. 推論実行
    print("推論実行中...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row["sent"]

        is_target_c = check_phase1_filtering(row)

        raw_response = ""
        if is_target_c:
            prediction = False
            phase = "Phase 1 (Filter)"
            raw_response = "Filtered by Moderation"
        else:
            prediction, raw_response = get_llm_prediction(
                model, tokenizer, prompt_template, text, args.temperature
            )
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

    # 4. 結果集計
    df_res = pd.DataFrame(results)
    y_true = df_res["gt_is_ambiguous"]
    y_pred = df_res["pred_is_ambiguous"]

    # Metrics
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
    report.append(f"Model: {args.model_id}")
    report.append(f"Prompt: {os.path.basename(args.prompt_path)}")
    report.append(f"Temperature: {args.temperature}")
    report.append(f"Total Samples: {len(df)}")
    report.append(
        f"Phase 1 Filtered: {len(df_res[df_res['phase'] == 'Phase 1 (Filter)'])}"
    )
    report.append("-" * 30)
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
        description="LLMを用いた情報欠損文章の自動検出実験"
    )

    # 必須引数: プロンプトと出力ディレクトリの指定を必須化
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="使用するプロンプトファイルのパス (例: src/04_ambiguity_detection/prompts/v01_base.txt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="結果を保存するサブディレクトリ名 (data/04_ambiguity_detection/results/ の下に作成されます。例: v01_base)",
    )

    # オプション引数: パラメータ調整時に使用
    parser.add_argument(
        "--model_id", type=str, default=DEFAULT_MODEL_ID, help="使用するLLMのID"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="LLMの生成温度 (0.0で決定論的)"
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        default="JCM_random_1000_sample_evaluated.csv",
        help="Ground Truth ファイル名 (data/04_ambiguity_detection/ にあるもの)",
    )

    args = parser.parse_args()
    run_experiment(args)
