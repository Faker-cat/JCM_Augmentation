# プロンプトを調整して曖昧性検出の分類実験を実行するスクリプト
import argparse
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

# --- 基本パス設定 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "04_ambiguity_detection")
GT_FILE_PATH = os.path.join(DATA_DIR, "JCM_random_1000_sample_evaluated.csv")
MODERATION_RESULTS_PATH = os.path.join(
    BASE_DIR, "data", "02_method_moderation", "moderation_full_results.csv"
)

# 生成パラメータ
MAX_NEW_TOKENS = 16
TEMPERATURE = 0.1


def load_prompt_template(prompt_path):
    """プロンプトファイルを読み込む"""
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"エラー: プロンプトファイルが見つかりません: {prompt_path}")
        exit(1)


def load_model(model_id):
    """指定されたモデルIDをロードする"""
    print(f"Loading model: {model_id} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        exit(1)


def load_and_merge_data():
    """GTデータとModeration結果を統合する"""
    df_gt = pd.read_csv(GT_FILE_PATH)
    if "gt_majority" in df_gt.columns:
        df_gt["gt_is_ambiguous"] = df_gt["gt_majority"].map(
            lambda x: True if str(x).strip().upper() == "TRUE" else False
        )
    else:
        raise ValueError("GTファイルに 'gt_majority' カラムが見つかりません。")

    df_mod = pd.read_csv(MODERATION_RESULTS_PATH)
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
    """Phase 1: Moderationによるフィルタリング"""
    original_label = row["original_label"]
    moderation_flagged = row["moderation_flagged"]
    if isinstance(moderation_flagged, str):
        moderation_flagged = moderation_flagged.lower() == "true"
    # ラベルが 1 (許容できない) かつ Moderationでフラグが立っているものを除外
    if (original_label == 1) and moderation_flagged:
        return True
    return False


def generate_prompt(prompt_template, text, label):
    """プロンプトの組み立て"""
    label_text = "許容できる" if label == 0 else "許容できない"
    instruction = prompt_template.replace("{text}", text).replace("{label}", label_text)
    return f"### 指示:\n{instruction}\n\n### 応答:\n"


def get_llm_prediction(model, tokenizer, prompt_template, text, label):
    """推論の実行"""
    prompt = generate_prompt(prompt_template, text, label)
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

    response = (
        tokenizer.decode(outputs[0], skip_special_tokens=True)
        .replace(prompt, "")
        .strip()
    )
    if "TRUE" in response.upper():
        return True, response
    return False, response


def run_experiment(args):
    # 1. パスの自動生成
    safe_model_name = args.model_id.replace("/", "_")
    prompt_name = os.path.splitext(os.path.basename(args.prompt_path))[0]
    full_output_dir = os.path.join(
        DATA_DIR, "results", safe_model_name, "classification", prompt_name
    )
    os.makedirs(full_output_dir, exist_ok=True)

    # 2. ロード
    prompt_template = load_prompt_template(args.prompt_path)
    model, tokenizer = load_model(args.model_id)
    df = load_and_merge_data()

    results = []
    print(f"=== 実験開始: {args.model_id} / {prompt_name} ===")

    # 3. 推論
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if check_phase1_filtering(row):
            results.append(
                {
                    "Original_ID": row["Original_ID"],
                    "sent": row["sent"],
                    "gt_is_ambiguous": row["gt_is_ambiguous"],
                    "pred_is_ambiguous": False,
                    "phase": "Phase 1 (Filter)",
                    "llm_raw_response": "Filtered by Moderation",
                }
            )
        else:
            pred, res = get_llm_prediction(
                model, tokenizer, prompt_template, row["sent"], row["original_label"]
            )
            results.append(
                {
                    "Original_ID": row["Original_ID"],
                    "sent": row["sent"],
                    "gt_is_ambiguous": row["gt_is_ambiguous"],
                    "pred_is_ambiguous": pred,
                    "phase": "Phase 2 (LLM)",
                    "llm_raw_response": res,
                }
            )

    # 4. レポート作成
    df_res = pd.DataFrame(results)
    df_phase1 = df_res[df_res["phase"] == "Phase 1 (Filter)"]

    # 指標計算
    total_filtered = len(df_phase1)
    filtered_but_ambiguous = df_phase1["gt_is_ambiguous"].sum()
    filter_success_rate = (
        (total_filtered - filtered_but_ambiguous) / total_filtered
        if total_filtered > 0
        else 0.0
    )
    filter_risk_rate = (
        filtered_but_ambiguous / total_filtered if total_filtered > 0 else 0.0
    )

    y_true, y_pred = df_res["gt_is_ambiguous"], df_res["pred_is_ambiguous"]
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    tp, fn = cm[0]
    fp, tn = cm[1]

    report = [
        "=== 実験結果レポート ===",
        f"Model: {args.model_id}",
        f"Prompt: {os.path.basename(args.prompt_path)}",
        f"Temperature: {TEMPERATURE}",
        f"Total Samples: {len(df)}",
        f"Phase 1 Filtered: {total_filtered}",
        "-" * 30,
        "Phase 1 Filtering Analysis:",
        f" Total Samples Filtered (Phase 1): {total_filtered}",
        f" Ambiguous Samples among Filtered (GT=True): {filtered_but_ambiguous} (見逃し)",
        f" Filter Success Rate (GT=False among filtered): {filter_success_rate:.4f}",
        f" Filter Risk Rate (GT=True among filtered): {filter_risk_rate:.4f}",
        "-" * 30,
        "Confusion Matrix (Target: Ambiguous/True):",
        f" TP: {tp}, FN: {fn}",
        f" FP: {fp}, TN: {tn}",
        "-" * 30,
        f"Accuracy : {accuracy_score(y_true, y_pred)}",
        f"Precision: {precision_score(y_true, y_pred, pos_label=True, zero_division=0)}",
        f"Recall   : {recall_score(y_true, y_pred, pos_label=True, zero_division=0)}",
        f"F1 Score : {f1_score(y_true, y_pred, pos_label=True, zero_division=0)}",
    ]

    # 5. 保存
    report_text = "\n".join(report)
    print("\n" + report_text)
    df_res.to_csv(
        os.path.join(full_output_dir, "results.csv"), index=False, encoding="utf-8-sig"
    )
    with open(
        os.path.join(full_output_dir, "experiment_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args)
