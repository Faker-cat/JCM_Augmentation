# src/04_ambiguity_detection/categorical_detection/run_categorical_pipeline.py
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

# --- パス設定 (3つ遡ってプロジェクトルートへ) ---
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_DIR = os.path.join(BASE_DIR, "data", "04_ambiguity_detection")
MASTER_GT_PATH = "/home/faker/JCM_Augmentation/data/04_ambiguity_detection/JCM_random_1000_sample_evaluated.csv"
RESULTS_ROOT = os.path.join(DATA_DIR, "results")
PROMPT_SETS_DIR = os.path.join(
    BASE_DIR, "src", "04_ambiguity_detection", "categorical_detection", "prompt_sets"
)


def load_model(model_id):
    print(f"Loading model: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tokenizer


def get_llm_prediction(model, tokenizer, prompt_template, text, label):
    # --- 修正・確認箇所 ---
    # マニュアルの表記に基づき、LLMが理解しやすい日本語ラベルに変換
    label_text = "許容できる" if int(label) == 0 else "許容できない"

    # プロンプト内の {text} と {label} を置換
    # label_text だけでなく、文脈に合わせて「許容できる」という言葉そのものを埋め込む
    full_prompt = prompt_template.replace("{text}", str(text)).replace(
        "{label}", label_text
    )
    # ----------------------

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        # max_new_tokensは [TRUE] / [FALSE] を拾うのに十分な10に設定
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = (
        tokenizer.decode(outputs[0], skip_special_tokens=True)
        .replace(full_prompt, "")
        .strip()
    )

    # [TRUE] という文字列が含まれているかどうかで判定
    is_true = "[TRUE]" in response.upper()
    return is_true, response


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    tp, fn = cm[0]
    fp, tn = cm[1]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=True, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=True, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=True, zero_division=0),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
    }


def main(args):
    safe_model_name = args.model_id.replace("/", "_")
    # プロンプトセット名（v01等）をパスに含める
    output_base = os.path.join(
        RESULTS_ROOT, safe_model_name, "categorical", args.prompt_set_name
    )
    raw_out_dir = os.path.join(output_base, "raw_outputs")
    final_out_dir = os.path.join(output_base, "final_integration")

    os.makedirs(raw_out_dir, exist_ok=True)
    os.makedirs(final_out_dir, exist_ok=True)

    # データのロード
    master_df = pd.read_csv(MASTER_GT_PATH)
    model, tokenizer = load_model(args.model_id)

    categories = ["cat1_logic", "cat2_label", "cat3_info"]
    all_cat_results = master_df[["Original_ID", "sent", "label", "gt_majority"]].copy()

    # 1. カテゴリ別の順次推論
    current_prompt_set_path = os.path.join(PROMPT_SETS_DIR, args.prompt_set_name)

    for cat in categories:
        print(f"\n--- Processing Category: {cat} (Set: {args.prompt_set_name}) ---")
        prompt_path = os.path.join(current_prompt_set_path, f"{cat}.txt")
        if not os.path.exists(prompt_path):
            print(f"Warning: {prompt_path} not found. Skipping.")
            continue

        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read().strip()

        cat_preds = []
        for _, row in tqdm(master_df.iterrows(), total=len(master_df)):
            pred, _ = get_llm_prediction(
                model, tokenizer, template, row["sent"], row["label"]
            )
            cat_preds.append(pred)

        all_cat_results[f"pred_{cat}"] = cat_preds
        # 中間ファイルの保存
        cat_df = master_df[["Original_ID", "sent", "label", "gt_majority"]].copy()
        cat_df["pred_is_ambiguous"] = cat_preds
        cat_df.to_csv(os.path.join(raw_out_dir, f"{cat}_results.csv"), index=False)

    # 2. 統合 (いずれかがTrueならTrue)
    all_cat_results["pred_is_ambiguous"] = all_cat_results[
        [f"pred_{c}" for c in categories if f"pred_{c}" in all_cat_results.columns]
    ].any(axis=1)

    # 3. 最終評価レポート作成
    metrics = calculate_metrics(
        all_cat_results["gt_majority"], all_cat_results["pred_is_ambiguous"]
    )

    report_text = [
        f"=== 最終再評価レポート (Categorical: {args.prompt_set_name}) ===",
        f"Model: {args.model_id}",
        f"Prompt Set Location: {current_prompt_set_path}",
        "-" * 30,
        "Confusion Matrix (Target: Ambiguous/True):",
        f" TP: {metrics['TP']}, FN: {metrics['FN']}",
        f" FP: {metrics['FP']}, TN: {metrics['TN']}",
        "-" * 30,
        f"Accuracy : {metrics['Accuracy']:.4f}",
        f"Precision: {metrics['Precision']:.4f}",
        f"Recall   : {metrics['Recall']:.4f}",
        f"F1 Score : {metrics['F1']:.4f}",
        "-" * 30,
        "Category Detection Summary (True Count):",
    ]
    for cat in categories:
        if f"pred_{cat}" in all_cat_results.columns:
            report_text.append(f" {cat}: {all_cat_results[f'pred_{cat}'].sum()}")

    # レポートとCSVの保存
    with open(
        os.path.join(final_out_dir, "experiment_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(report_text))

    all_cat_results.to_csv(
        os.path.join(final_out_dir, "final_results.csv"), index=False
    )

    # 検出された文章（いずれかがTrue）のみ抽出したまとめCSV
    all_cat_results[all_cat_results["pred_is_ambiguous"]].to_csv(
        os.path.join(final_out_dir, "combined_detection.csv"), index=False
    )

    print(f"\nCompleted. Results saved in: {output_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, required=True, help="HuggingFace model ID"
    )
    parser.add_argument(
        "--prompt_set_name",
        type=str,
        default="v01",
        help="Name of the prompt set folder in prompt_sets/",
    )
    args = parser.parse_args()
    main(args)
