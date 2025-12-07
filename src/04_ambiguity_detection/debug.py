import logging
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
from transformers import logging as hf_logging

# --- ログ設定: 詳細情報を表示 ---
logging.basicConfig(level=logging.INFO)
hf_logging.set_verbosity_info()  # Transformersのロード状況を表示

# --- 設定 ---
MODEL_ID = "llm-jp/llm-jp-3.1-13b-instruct4"

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "04_ambiguity_detection")
GT_FILE_PATH = os.path.join(DATA_DIR, "JCM_random_1000_sample_evaluated.csv")
MODERATION_RESULTS_PATH = os.path.join(
    BASE_DIR, "data", "02_method_moderation", "moderation_full_results.csv"
)
OUTPUT_DIR = DATA_DIR

# 生成パラメータ
MAX_NEW_TOKENS = 16
TEMPERATURE = 0.1


def load_model():
    print(f"Loading model: {MODEL_ID} ...")

    if not torch.cuda.is_available():
        print("【警告】GPUが検出されませんでした。")
    else:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM Info: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # A6000なら bfloat16 で十分収まるはずです
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # 自動配置（通常はGPU0に全て乗るはず）
        )
        model.eval()

        # モデルの配置を確認
        print("-" * 20)
        print(f"Model Device Map: {model.hf_device_map}")
        print("-" * 20)

        return model, tokenizer
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        exit(1)


def load_and_merge_data():
    # (前回と同じため省略なしで記載しますが、変更はありません)
    print(f"Loading GT data from: {GT_FILE_PATH}")
    if not os.path.exists(GT_FILE_PATH):
        raise FileNotFoundError(f"GTファイルが見つかりません: {GT_FILE_PATH}")

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
    original_label = row["original_label"]
    moderation_flagged = row["moderation_flagged"]

    if isinstance(moderation_flagged, str):
        moderation_flagged = moderation_flagged.lower() == "true"

    if (original_label == 1) and moderation_flagged:
        return True
    return False


def generate_prompt(text):
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
    prompt = f"### 指示:\n{instruction}\n\n### 応答:\n"
    return prompt


def get_llm_prediction(model, tokenizer, text):
    prompt = generate_prompt(text)

    # deviceへ移動
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
        model.device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=TEMPERATURE,
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
        return False, response


def run_experiment():
    print(f"=== 実験開始: debug LLM Ambiguity Detection (Model: {MODEL_ID}) ===")

    model, tokenizer = load_model()

    df = load_and_merge_data()
    print(f"対象データ数: {len(df)}件")

    results = []

    print("推論実行中...")
    # tqdmで進捗を表示
    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row["sent"]

        is_target_c = check_phase1_filtering(row)

        raw_response = ""
        if is_target_c:
            prediction = False
            phase = "Phase 1 (Filter)"
            raw_response = "Filtered by Moderation"
        else:
            prediction, raw_response = get_llm_prediction(model, tokenizer, text)
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

    # 結果保存（省略なし）
    df_res = pd.DataFrame(results)

    # Metrics
    y_true = df_res["gt_is_ambiguous"]
    y_pred = df_res["pred_is_ambiguous"]
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    tp, fn = cm[0]
    fp, tn = cm[1]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=True, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=True, zero_division=0)

    print("\n=== 実験結果レポート ===")
    print(f"Model Device Map: {model.hf_device_map}")  # 最終確認
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_csv = os.path.join(OUTPUT_DIR, "experiment_1_2_results_llmjp.csv")
    df_res.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n結果を保存しました: {output_csv}")


if __name__ == "__main__":
    run_experiment()
