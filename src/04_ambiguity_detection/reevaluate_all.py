# src/04_ambiguity_detection/reevaluate_all.py
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

# --- 基本パス設定 ---
# --- 修正後のパス設定 ---
# 1回目: src/04_ambiguity_detection/
# 2回目: src/
# 3回目: JCM_Augmentation/ (プロジェクトルート)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 以下はご提示の通りで正しく動作します
RESULTS_ROOT = os.path.join(BASE_DIR, "data", "04_ambiguity_detection", "results")
MASTER_GT_PATH = "/home/faker/JCM_Augmentation/data/04_ambiguity_detection/JCM_random_1000_sample_evaluated.csv"
OUTPUT_ROOT = os.path.join(BASE_DIR, "data", "04_ambiguity_detection", "reevaluated")


def calculate_classification_metrics(df, gt_col, pred_col):
    """二値分類指標（Accuracy, Precision, Recall, F1）の計算"""
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
    """評価結果のテキストレポートを生成"""
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


def process_scoring(df_merged, gt_type, out_dir, prompt_name):
    """Scoring手法特有の集計（スコア分布）と可視化"""
    summary = pd.crosstab(df_merged["predicted_score"], df_merged[gt_type]).reindex(
        range(11), fill_value=0
    )
    summary["Total"] = summary.sum(axis=1)
    summary["True_Ratio"] = summary[True] / summary["Total"]
    summary.to_csv(
        os.path.join(out_dir, "score_summary_table.csv"), encoding="utf-8-sig"
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_merged,
        x="predicted_score",
        hue=gt_type,
        multiple="stack",
        bins=11,
        binrange=(0, 11),
    )
    plt.title(f"Ambiguity Score Distribution\n({prompt_name} / {gt_type})")
    plt.savefig(os.path.join(out_dir, "score_distribution.png"))
    plt.close()

    df_merged["binary_pred"] = df_merged["predicted_score"] >= 5
    metrics = calculate_classification_metrics(df_merged, gt_type, "binary_pred")
    return metrics, "Score Distribution Summary:\n" + summary.to_string()


def main(args):
    # 最新の正解ラベルをロード
    master_gt = pd.read_csv(MASTER_GT_PATH)
    gt_cols = ["gt_majority", "gt_union"]

    for root, dirs, files in os.walk(RESULTS_ROOT):
        rel_path = os.path.relpath(root, RESULTS_ROOT)
        path_parts = rel_path.split(os.sep)
        if len(path_parts) < 3:
            continue

        model_name, method, prompt_name = path_parts[0], path_parts[1], path_parts[2]

        # コマンドライン引数によるフィルタリング
        if args.model and args.model != model_name:
            continue
        if args.method and args.method != method:
            continue
        if args.prompt and args.prompt != prompt_name:
            continue

        # --- 手法に応じた入力ファイル名の特定 ---
        target_file = None
        if method == "scoring" and "scoring_results.csv" in files:
            target_file = "scoring_results.csv"
        elif method == "stepwise" and "stepwise_results.csv" in files:
            target_file = "stepwise_results.csv"
        elif method == "fewshot" and "fewshot_results.csv" in files:
            target_file = "fewshot_results.csv"
        elif method == "classification" and "results.csv" in files:
            target_file = "results.csv"

        if not target_file:
            continue

        print(f"Processing: {model_name} / {method} / {prompt_name}")

        df_pred = pd.read_csv(os.path.join(root, target_file))

        # 不要な古いGT列を削除
        cols_to_drop = [
            c for c in df_pred.columns if c in gt_cols or c == "gt_is_ambiguous"
        ]
        df_pred = df_pred.drop(columns=cols_to_drop)

        # Original_IDをキーにして最新のGTを紐付け
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
                metrics, extra_info = process_scoring(
                    df_merged, gt_type, out_dir, prompt_name
                )
                df_merged.to_csv(
                    os.path.join(out_dir, target_file),
                    index=False,
                    encoding="utf-8-sig",
                )
            else:
                # classification, stepwise, fewshot は共通の二値評価ロジック
                metrics = calculate_classification_metrics(
                    df_merged, gt_type, "pred_is_ambiguous"
                )
                df_merged.to_csv(
                    os.path.join(out_dir, target_file),
                    index=False,
                    encoding="utf-8-sig",
                )

            # レポート（experiment_report.txt）の保存
            report_text = generate_report_text(
                method, model_name, prompt_name, gt_type, metrics, extra_info
            )
            with open(
                os.path.join(out_dir, "experiment_report.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(report_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="既存の推論結果を最新のGT基準で再評価する"
    )
    parser.add_argument("--model", type=str, help="モデル名")
    parser.add_argument(
        "--method", type=str, help="手法名 (classification, fewshot, scoring, stepwise)"
    )
    parser.add_argument("--prompt", type=str, help="プロンプト名")
    args = parser.parse_args()
    main(args)
