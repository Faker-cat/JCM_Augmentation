# src/04_ambiguity_detection/run_stepwise_experiment.py
# idea2: 段階判定による曖昧性検出実験を実行するスクリプト
import argparse
import os

import pandas as pd
import torch

# 既存の分類スクリプトから必要な関数や定数をインポート
from run_classification_experiment import (
    DATA_DIR,
    TEMPERATURE,
    check_phase1_filtering,
    generate_prompt,
    load_and_merge_data,
    load_model,
    load_prompt_template,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm


def get_stepwise_prediction(model, tokenizer, prompt_template, text, label):
    """
    【idea2: 段階判定】LLMに思考ステップを出力させ、最終判定を抽出する
    """
    prompt = generate_prompt(prompt_template, text, label)
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
        model.device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # 思考プロセスを出力させるため長めに設定
            do_sample=False,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.replace(prompt, "").strip()

    # パース処理：最後に出現する「TRUE」または「FALSE」を探す
    upper_res = response.upper()
    if "最終判定" in response:
        final_part = response.split("最終判定")[-1].upper()
        if "TRUE" in final_part:
            return True, response
        elif "FALSE" in final_part:
            return False, response

    if "TRUE" in upper_res:
        return True, response
    return False, response


def run_stepwise_experiment(args):
    """実験の実行ループ"""
    # 1. 保存用パスの自動生成
    safe_model_name = args.model_id.replace("/", "_")
    prompt_name = os.path.splitext(os.path.basename(args.prompt_path))[0]
    full_output_dir = os.path.join(
        DATA_DIR, "results", safe_model_name, "stepwise", prompt_name
    )
    os.makedirs(full_output_dir, exist_ok=True)

    print(f"=== 実験開始: Stepwise / {args.model_id} / {prompt_name} ===")

    # 2. モデルとデータのロード
    prompt_template = load_prompt_template(args.prompt_path)
    model, tokenizer = load_model(args.model_id)
    df = load_and_merge_data()

    # テスト時は件数を制限することも可能 (現在は全件対象)
    df = df.head(10)

    results = []
    print("推論実行中（段階判定）...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row["sent"]
        label = row["original_label"]

        if check_phase1_filtering(row):
            prediction = False
            phase = "Phase 1 (Filter)"
            raw_response = "Filtered by Moderation"
        else:
            prediction, raw_response = get_stepwise_prediction(
                model, tokenizer, prompt_template, text, label
            )
            phase = "Phase 2 (LLM-Stepwise)"

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

    # 3. 集計とレポート作成
    df_res = pd.DataFrame(results)
    df_phase1 = df_res[df_res["phase"] == "Phase 1 (Filter)"]

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
        "=== 実験結果レポート (Stepwise) ===",
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

    # 4. 保存
    report_text = "\n".join(report)
    print("\n" + report_text)

    output_csv = os.path.join(full_output_dir, "stepwise_results.csv")
    df_res.to_csv(output_csv, index=False, encoding="utf-8-sig")

    with open(
        os.path.join(full_output_dir, "experiment_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report_text)

    print(f"結果を保存しました: {full_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="idea2: 段階判定による情報欠損検出")
    parser.add_argument(
        "--model_id", type=str, required=True, help="Hugging Face model ID"
    )
    parser.add_argument(
        "--prompt_path", type=str, required=True, help="プロンプトのパス"
    )

    args = parser.parse_args()
    run_stepwise_experiment(args)
