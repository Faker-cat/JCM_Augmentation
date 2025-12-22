# src/04_ambiguity_detection/merge_gt_evaluations.py
import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# --- パス設定 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "04_ambiguity_detection", "raw_evaluations")
OUTPUT_CSV_PATH = os.path.join(
    BASE_DIR, "data", "04_ambiguity_detection", "JCM_random_1000_sample_evaluated.csv"
)
OUTPUT_REPORT_PATH = os.path.join(
    BASE_DIR, "data", "04_ambiguity_detection", "gt_evaluation_report.txt"
)  # ★レポート保存先


def load_and_clean(filename):
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path)
    # 「Aのフラグ」をBooleanに変換
    df["flag"] = df["Aのフラグ"].map(
        lambda x: True if str(x).strip().upper() == "TRUE" else False
    )
    return df[["Original_ID", "flag"]]


def calculate_fleiss_kappa(matrix):
    """Fleiss' Kappaの計算"""
    n, k = matrix.shape
    n_true = matrix.sum(axis=1)
    n_false = k - n_true
    p_i = (n_true**2 + n_false**2 - k) / (k * (k - 1))
    p_mean = p_i.mean()
    p_j_true = n_true.sum() / (n * k)
    p_j_false = n_false.sum() / (n * k)
    p_e = p_j_true**2 + p_j_false**2
    return (p_mean - p_e) / (1 - p_e)


def main():
    # 1. データの読み込み
    boss = load_and_clean("boss.csv").rename(columns={"flag": "flag_boss"})
    faker = load_and_clean("faker.csv").rename(columns={"flag": "flag_faker"})
    mila = load_and_clean("mila.csv").rename(columns={"flag": "flag_mila"})

    # 2. 結合
    base_df = pd.read_csv(os.path.join(RAW_DIR, "boss.csv")).drop(columns=["Aのフラグ"])
    merged = (
        base_df.merge(boss, on="Original_ID")
        .merge(faker, on="Original_ID")
        .merge(mila, on="Original_ID")
    )

    # 3. 集計
    flags = merged[["flag_boss", "flag_faker", "flag_mila"]]
    true_counts = flags.sum(axis=1)

    # 指標の計算
    complete_agreement = (true_counts == 3) | (true_counts == 0)
    agreement_counts = true_counts.apply(lambda x: 3 if x == 3 or x == 0 else 2)
    avg_agreement_number = agreement_counts.mean()

    k_bf = cohen_kappa_score(merged["flag_boss"], merged["flag_faker"])
    k_fm = cohen_kappa_score(merged["flag_faker"], merged["flag_mila"])
    k_mb = cohen_kappa_score(merged["flag_mila"], merged["flag_boss"])
    avg_kappa = (k_bf + k_fm + k_mb) / 3
    f_kappa = calculate_fleiss_kappa(flags.values)

    merged["gt_majority"] = true_counts >= 2
    merged["gt_union"] = true_counts >= 1

    # 4. レポート作成 [★修正ポイント]
    report_lines = [
        "=== 評価者合意度レポート (Ground Truth Analysis) ===",
        f"Total Samples: {len(merged)}",
        "-" * 40,
        "Agreement Metrics:",
        f" Complete Agreement Rate : {complete_agreement.mean():.4f}",
        f" Average Agreement Number: {avg_agreement_number:.4f} / 3.0",
        "-" * 40,
        "Reliability Metrics (Kappa):",
        f" Average Pairwise Kappa  : {avg_kappa:.4f}",
        f" Fleiss' Kappa           : {f_kappa:.4f}",
        "-" * 40,
        "GT Label Distribution:",
        f" Majority (>=2) True    : {merged['gt_majority'].sum()}",
        f" Union (>=1) True       : {merged['gt_union'].sum()}",
        "-" * 40,
        "Calculation Details:",
        f" Boss True Count        : {merged['flag_boss'].sum()}",
        f" Faker True Count       : {merged['flag_faker'].sum()}",
        f" Mila True Count        : {merged['flag_mila'].sum()}",
    ]

    report_text = "\n".join(report_lines)

    # ターミナルに表示
    print(report_text)

    # テキストファイルに保存
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    # 5. CSVの保存
    merged.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved Report: {OUTPUT_REPORT_PATH}")
    print(f"Saved Merged CSV: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
