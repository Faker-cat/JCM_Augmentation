# src/04_ambiguity_detection/reevaluate_all.py
# 既存の曖昧度検出実験結果をマスターGTに基づいて再評価するスクリプト
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def clean_columns(df):
    """BOMや余計な空白を除去してカラム名を正規化する (KeyError対策)"""
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df


def calculate_classification_metrics(df, gt_col, pred_col):
    """二値分類指標の計算"""
    y_true = df[gt_col].astype(bool)
    y_pred = df[pred_col].astype(bool)
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
    # 1. マスターGTのロード (多数決と1人以上基準を読み込む)
    master_gt = clean_columns(pd.read_csv(MASTER_GT_PATH, encoding="utf-8-sig"))
    gt_cols = ["gt_majority", "gt_union"]

    # 2. Resultsフォルダ内を探索
    for root, dirs, files in os.walk(RESULTS_ROOT):
        rel_path = os.path.relpath(root, RESULTS_ROOT)
        path_parts = rel_path.split(os.sep)
        if len(path_parts) < 3:
            continue
        model_name, method, prompt_name = path_parts[0], path_parts[1], path_parts[2]

        if args.model and args.model != model_name:
            continue
        if args.method and args.method != method:
            continue
        if args.prompt and args.prompt != prompt_name:
            continue

        # ファイル名の特定とメソッドの判定
        target_file = None
        if "results.csv" in files:
            target_file = "results.csv"
        elif "scoring_results.csv" in files:
            target_file = "scoring_results.csv"
        elif "stepwise_results.csv" in files:
            target_file = "stepwise_results.csv"

        if not target_file:
            continue
        print(f"Re-evaluating: {model_name} / {method} / {prompt_name}")

        # 3. 推論データのロードと正規化
        df_pred = clean_columns(
            pd.read_csv(os.path.join(root, target_file), encoding="utf-8-sig")
        )
        # 既存の古いGT列（もしあれば）を削除して、マスターGTから結合する
        cols_to_drop = [
            c for c in df_pred.columns if c in gt_cols or c == "gt_is_ambiguous"
        ]
        df_pred = df_pred.drop(columns=cols_to_drop)
        df_merged = df_pred.merge(
            master_gt[["Original_ID"] + gt_cols], on="Original_ID", how="left"
        )

        for gt_type in gt_cols:
            out_dir = os.path.join(
                OUTPUT_ROOT, gt_type, model_name, method, prompt_name
            )
            os.makedirs(out_dir, exist_ok=True)

            extra_info = ""
            metrics = {}

            if method == "scoring":
                # --- Scoring 特有の処理 ---
                summary = pd.crosstab(
                    df_merged["predicted_score"], df_merged[gt_type]
                ).reindex(range(11), fill_value=0)
                summary["Total"] = summary.sum(axis=1)
                summary["True_Ratio"] = summary[True] / summary["Total"]
                extra_info = "Score Distribution Summary:\n" + summary.to_string()

                # スコア集計表の保存
                summary.to_csv(
                    os.path.join(out_dir, "score_summary_table.csv"),
                    encoding="utf-8-sig",
                )

                # 可視化の保存
                plt.figure(figsize=(10, 6))
                sns.histplot(
                    data=df_merged,
                    x="predicted_score",
                    hue=gt_type,
                    multiple="stack",
                    bins=11,
                    binrange=(0, 11),
                )
                plt.title(f"Re-evaluated: {prompt_name} ({gt_type})")
                plt.savefig(os.path.join(out_dir, "score_distribution.png"))
                plt.close()

                # 二値評価としてのメトリクス (閾値5で判定)
                df_merged["binary_pred"] = df_merged["predicted_score"] >= 5
                metrics = calculate_classification_metrics(
                    df_merged, gt_type, "binary_pred"
                )
                df_merged.to_csv(
                    os.path.join(out_dir, "scoring_results.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )

            else:
                # --- Classification / Stepwise の処理 ---
                metrics = calculate_classification_metrics(
                    df_merged, gt_type, "pred_is_ambiguous"
                )
                filename = (
                    "results.csv"
                    if method == "classification"
                    else "stepwise_results.csv"
                )
                df_merged.to_csv(
                    os.path.join(out_dir, filename), index=False, encoding="utf-8-sig"
                )

            # レポート保存
            report_text = generate_report_text(
                method, model_name, prompt_name, gt_type, metrics, extra_info
            )
            with open(
                os.path.join(out_dir, "experiment_report.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(report_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="モデル名")
    parser.add_argument(
        "--method",
        type=str,
        choices=["classification", "scoring", "stepwise"],
        help="手法名",
    )
    parser.add_argument("--prompt", type=str, help="プロンプト名")
    main(parser.parse_args())
