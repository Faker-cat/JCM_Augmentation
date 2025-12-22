# src/04_ambiguity_detection/merge_gt_evaluations.py
# 複数評価データの結合とGT定義の更新
import os

import pandas as pd
from sklearn.metrics import cohen_kappa_score

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "04_ambiguity_detection", "raw_evaluations")
OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "04_ambiguity_detection", "JCM_random_1000_sample_evaluated.csv"
)


def load_and_clean(filename):
    path = os.path.join(RAW_DIR, filename)
    df = pd.read_csv(path)
    # 「Aのフラグ」をBooleanに変換
    df["flag"] = df["Aのフラグ"].map(
        lambda x: True if str(x).strip().upper() == "TRUE" else False
    )
    return df[["Original_ID", "flag"]]


def calculate_fleiss_kappa(matrix):
    """3人以上の評価者に対応したFleiss' Kappaの簡易実装"""
    n, k = matrix.shape  # n: サンプル数, k: 評価者数
    # 各行のTrue(1)の数をカウント
    n_true = matrix.sum(axis=1)
    n_false = k - n_true

    # 評価者の一致度 P_i
    p_i = (n_true**2 + n_false**2 - k) / (k * (k - 1))
    p_mean = p_i.mean()

    # 偶然の一致度 P_e
    p_j_true = n_true.sum() / (n * k)
    p_j_false = n_false.sum() / (n * k)
    p_e = p_j_true**2 + p_j_false**2

    return (p_mean - p_e) / (1 - p_e)


def main():
    # 1. データの読み込み
    print("Loading evaluations...")
    boss = load_and_clean("boss.csv").rename(columns={"flag": "flag_boss"})
    faker = load_and_clean("faker.csv").rename(columns={"flag": "flag_faker"})
    mila = load_and_clean("mila.csv").rename(columns={"flag": "flag_mila"})

    # 2. 結合 (Original_IDベース)
    # ベースとなるファイル（文章データを含むもの）を読み込む
    base_df = pd.read_csv(os.path.join(RAW_DIR, "boss.csv")).drop(columns=["Aのフラグ"])
    merged = (
        base_df.merge(boss, on="Original_ID")
        .merge(faker, on="Original_ID")
        .merge(mila, on="Original_ID")
    )

    # 3. 集計
    flags = merged[["flag_boss", "flag_faker", "flag_mila"]]
    true_counts = flags.sum(axis=1)

    # 完全一致率: 3人全員が同じ
    complete_agreement = (true_counts == 3) | (true_counts == 0)

    # 新しいGTの定義
    merged["gt_majority"] = true_counts >= 2  # 多数決 (2人以上)
    merged["gt_union"] = true_counts >= 1  # 1人でもTrueならTrue

    agreement_counts = true_counts.apply(lambda x: 3 if x == 3 or x == 0 else 2)
    avg_agreement_number = agreement_counts.mean()

    # 4. 指標の計算
    print("-" * 30)
    print(f"Total Samples: {len(merged)}")
    print(f"Complete Agreement Rate: {complete_agreement.mean():.4f}")
    print(f"Average Agreement Number: {avg_agreement_number:.4f} / 3.0")

    # Pairwise Cohen's Kappa (参考)
    k_bf = cohen_kappa_score(merged["flag_boss"], merged["flag_faker"])
    k_fm = cohen_kappa_score(merged["flag_faker"], merged["flag_mila"])
    k_mb = cohen_kappa_score(merged["flag_mila"], merged["flag_boss"])
    avg_kappa = (k_bf + k_fm + k_mb) / 3
    print(f"Average Pairwise Kappa: {avg_kappa:.4f}")

    # Fleiss' Kappa
    f_kappa = calculate_fleiss_kappa(flags.values)
    print(f"Fleiss' Kappa: {f_kappa:.4f}")

    # GT分布の確認
    print("-" * 30)
    print("GT Distribution:")
    print(f" Majority (>=2): {merged['gt_majority'].sum()} samples")
    print(f" Union    (>=1): {merged['gt_union'].sum()} samples")

    # 5. 保存
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nSaved merged GT to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
