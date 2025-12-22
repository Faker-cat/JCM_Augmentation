# src/04_ambiguity_detection/reevaluate_all.py
# 再評価スクリプト: 既存の推論結果をマスターGTに基づいて再評価し、指標を算出・保存する
import argparse
import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(BASE_DIR, "data", "04_ambiguity_detection", "results")
MASTER_GT_PATH = os.path.join(
    BASE_DIR, "data", "04_ambiguity_detection", "JCM_random_1000_sample_evaluated.csv"
)
OUTPUT_ROOT = os.path.join(BASE_DIR, "data", "04_ambiguity_detection", "reevaluated")


def calculate_classification_metrics(df, gt_col, pred_col):
    """二値分類指標の計算"""
    y_true = df[gt_col]
    y_pred = df[pred_col]
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


def generate_report_text(method, model, prompt, gt_type, metrics, extra_info=""):
    """レポートテキストの組み立て"""
    report = [
        f"=== 再評価レポート ({gt_type}) ===",
        f"Method: {method}",
        f"Model: {model}",
        f"Prompt: {prompt}",
        "-" * 30,
        "Confusion Matrix (Target: Ambiguous/True):",
        f" TP: {metrics['TP']}, FN: {metrics['FN']}",
        f" FP: {metrics['FP']}, TN: {metrics['TN']}",
        "-" * 30,
        f"Accuracy : {metrics['Accuracy']:.4f}",
        f"Precision: {metrics['Precision']:.4f}",
        f"Recall   : {metrics['Recall']:.4f}",
        f"F1 Score : {metrics['F1']:.4f}",
    ]
    if extra_info:
        report.append("-" * 30)
        report.append(extra_info)
    return "\n".join(report)


def main(args):
    # 1. マスターGTのロード
    master_gt = pd.read_csv(MASTER_GT_PATH)
    gt_cols = ["gt_majority", "gt_union"]

    # 2. Resultsフォルダ内を探索
    for root, dirs, files in os.walk(RESULTS_ROOT):
        # パスから情報を抽出 (例: results/MODEL/METHOD/PROMPT)
        rel_path = os.path.relpath(root, RESULTS_ROOT)
        path_parts = rel_path.split(os.sep)
        if len(path_parts) < 3:
            continue

        model_name, method, prompt_name = path_parts[0], path_parts[1], path_parts[2]

        # --- 引数によるフィルタリング ---
        if args.model and args.model != model_name:
            continue
        if args.method and args.method != method:
            continue
        if args.prompt and args.prompt != prompt_name:
            continue

        # ファイル特定
        target_file = None
        if method == "classification" and "results.csv" in files:
            target_file = "results.csv"
        elif method == "scoring" and "scoring_results.csv" in files:
            target_file = "scoring_results.csv"
        elif method == "stepwise" and "stepwise_results.csv" in files:
            target_file = "stepwise_results.csv"

        if not target_file:
            continue

        print(f"Re-evaluating: {model_name} / {method} / {prompt_name}")

        # 3. データの結合と再評価
        df_pred = pd.read_csv(os.path.join(root, target_file))
        df_merged = df_pred.merge(
            master_gt[["Original_ID"] + gt_cols], on="Original_ID", how="left"
        )

        for gt_type in gt_cols:
            out_dir = os.path.join(
                OUTPUT_ROOT, gt_type, model_name, method, prompt_name
            )
            os.makedirs(out_dir, exist_ok=True)

            extra_info = ""
            if method == "scoring":
                summary = pd.crosstab(
                    df_merged["predicted_score"], df_merged[gt_type]
                ).reindex(range(11), fill_value=0)
                extra_info = "Score Distribution Summary:\n" + summary.to_string()
                df_merged["binary_pred"] = (
                    df_merged["predicted_score"] >= 5
                )  # 閾値5で暫定判定
                metrics = calculate_classification_metrics(
                    df_merged, gt_type, "binary_pred"
                )
            else:
                metrics = calculate_classification_metrics(
                    df_merged, gt_type, "pred_is_ambiguous"
                )

            # 保存
            report_text = generate_report_text(
                method, model_name, prompt_name, gt_type, metrics, extra_info
            )
            with open(
                os.path.join(out_dir, "experiment_report.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(report_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="既存の推論結果を指定して再評価を行う")
    parser.add_argument(
        "--model", type=str, help="特定のモデル名 (例: llm-jp_llm-jp-3.1-13b-instruct4)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["classification", "scoring", "stepwise"],
        help="特定の手法名",
    )
    parser.add_argument(
        "--prompt", type=str, help="特定のプロンプトファイル名 (拡張子なし)"
    )

    args = parser.parse_args()
    main(args)
