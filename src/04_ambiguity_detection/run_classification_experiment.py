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

# --- 設定 ---
# 使用するモデルID (Hugging Face)
MODEL_ID = "llm-jp/llm-jp-3.1-13b-instruct4"

# パス設定
# プロジェクトルート: src/04_.../run_...py から3階層上
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# データディレクトリ
DATA_DIR = os.path.join(BASE_DIR, "data", "04_ambiguity_detection")

# 入力ファイル1: 人手評価済みGTデータ
GT_FILE_PATH = os.path.join(DATA_DIR, "JCM_random_1000_sample_evaluated.csv")

# 入力ファイル2: Moderation全量結果 (Phase 1フィルタリング用)
MODERATION_RESULTS_PATH = os.path.join(
    BASE_DIR, "data", "02_method_moderation", "moderation_full_results.csv"
)

# 出力先
OUTPUT_DIR = DATA_DIR  # data/04_ambiguity_detection に出力

# 生成パラメータ
MAX_NEW_TOKENS = 16
TEMPERATURE = 0.1  # 決定論的にするため低めに設定


def load_model():
    """
    モデルとトークナイザーを読み込む
    """
    print(f"Loading model: {MODEL_ID} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,  # VRAM節約のためbfloat16推奨（GPUが対応していれば）
            device_map="auto",  # 空きGPUに自動配置
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        exit(1)


def load_and_merge_data():
    """
    GTデータとModeration結果をIDベースで統合する
    """
    print(f"Loading GT data from: {GT_FILE_PATH}")
    if not os.path.exists(GT_FILE_PATH):
        raise FileNotFoundError(
            f"GTファイルが見つかりません: {GT_FILE_PATH}\nパスやファイル名を確認してください。"
        )

    # 1. GTデータの読み込み
    df_gt = pd.read_csv(GT_FILE_PATH)

    # フラグ処理
    if "Aのフラグ" in df_gt.columns:
        df_gt["gt_is_ambiguous"] = df_gt["Aのフラグ"].map(
            lambda x: True if str(x).strip().upper() == "TRUE" else False
        )
    else:
        raise ValueError("GTファイルに 'Aのフラグ' カラムが見つかりません。")

    # 2. Moderation結果の読み込み
    print(f"Loading Moderation results from: {MODERATION_RESULTS_PATH}")
    if not os.path.exists(MODERATION_RESULTS_PATH):
        raise FileNotFoundError(
            f"Moderation結果ファイルが見つかりません: {MODERATION_RESULTS_PATH}"
        )

    df_mod = pd.read_csv(MODERATION_RESULTS_PATH)

    # 3. マージ
    print("Merging data...")
    # 型合わせ
    df_gt["Original_ID"] = pd.to_numeric(df_gt["Original_ID"], errors="coerce")
    df_mod["ID"] = pd.to_numeric(df_mod["ID"], errors="coerce")

    df_merged = pd.merge(
        df_gt,
        df_mod[["ID", "original_label", "moderation_flagged"]],
        left_on="Original_ID",
        right_on="ID",
        how="left",
    )

    # 欠損値補完
    df_merged["original_label"] = df_merged["original_label"].fillna(0)
    df_merged["moderation_flagged"] = df_merged["moderation_flagged"].fillna(False)

    return df_merged


def check_phase1_filtering(row):
    """
    Phase 1: Moderationによるフィルタリング
    Target C (NGかつFlagged) は「情報十分(False)」とみなす
    """
    original_label = row["original_label"]
    moderation_flagged = row["moderation_flagged"]

    if isinstance(moderation_flagged, str):
        moderation_flagged = moderation_flagged.lower() == "true"

    if (original_label == 1) and moderation_flagged:
        return True
    return False


def generate_prompt(text):
    """
    LLM-JP用のプロンプト作成
    """
    instruction = f"""
あなたは日本語の道徳的判断データセットの品質管理者です。
以下の「対象の文章」だけを読んで、「道徳的に許容できる/できない」を判断するために、**必要な前提条件や状況説明が不足している（曖昧である）か**を判定してください。

対象の文章:
「{text}」

判定基準:
- TRUE: 文脈や前提条件（誰が、どこで、なぜ、など）が不明瞭で、これだけでは道徳的な判断を一意に下すことが難しい場合。
- FALSE: 文脈が明確である、あるいは常識的に考えて即座に判断が可能である場合。

回答は「TRUE」または「FALSE」の単語のみを出力してください。
"""
    # LLM-JP系の標準的なプロンプトフォーマット
    prompt = f"### 指示:\n{instruction}\n\n### 応答:\n"
    return prompt


def get_llm_prediction(model, tokenizer, text):
    """
    モデルを用いた推論実行（修正版）
    """
    prompt = generate_prompt(text)

    # 修正点: add_special_tokens=False はプロンプト形式によるが、ここでは標準動作でOK
    # 重要: return_token_type_ids=False を指定してエラーを回避
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
        model.device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # 決定論的生成
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # プロンプト部分を除去して応答部分のみ抽出
    response = generated_text.replace(prompt, "").strip()

    # 後処理：「### 応答:」などが残っている場合の対策
    if "### 応答:" in response:
        response = response.split("### 応答:")[-1].strip()

    # 判定
    upper_res = response.upper()
    if "TRUE" in upper_res:
        return True, response
    elif "FALSE" in upper_res:
        return False, response
    else:
        # 解析不能時はFalse扱い
        return False, response


def run_experiment():
    print(f"=== 実験開始: LLM Ambiguity Detection (Model: {MODEL_ID}) ===")

    # モデルロード
    model, tokenizer = load_model()

    # データ準備
    df = load_and_merge_data()
    print(f"対象データ数: {len(df)}件")

    results = []

    print("推論実行中...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row["sent"]

        # Phase 1
        is_target_c = check_phase1_filtering(row)

        raw_response = ""
        if is_target_c:
            prediction = False
            phase = "Phase 1 (Filter)"
            raw_response = "Filtered by Moderation"
        else:
            # Phase 2 (LLM)
            prediction, raw_response = get_llm_prediction(model, tokenizer, text)
            phase = "Phase 2 (LLM)"

        results.append(
            {
                "Original_ID": row["Original_ID"],
                "sent": text,
                "gt_is_ambiguous": row["gt_is_ambiguous"],
                "pred_is_ambiguous": prediction,
                "phase": phase,
                "llm_raw_response": raw_response,  # デバッグ用に生の応答も保存
            }
        )

    # 結果集計
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

    # レポート作成（文字列として保持）
    report = []
    report.append("=== 実験結果レポート ===")
    report.append(f"Model: {MODEL_ID}")
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
    report.append(f"Recall   : {recall}")
    report.append(f"F1 Score : {f1}")

    report_text = "\n".join(report)

    # 標準出力
    print("\n" + report_text)

    # 保存ディレクトリ作成
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. CSV保存
    output_csv = os.path.join(OUTPUT_DIR, "ambiguity_detection_results_llmjp.csv")
    df_res.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n結果CSVを保存しました: {output_csv}")

    # 2. レポートテキスト保存 (要望箇所)
    report_path = os.path.join(OUTPUT_DIR, "experiment_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"結果レポートを保存しました: {report_path}")


if __name__ == "__main__":
    run_experiment()
