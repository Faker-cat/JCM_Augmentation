import os

import pandas as pd

# 基本パスの設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT_FILE = os.path.join(
    BASE_DIR, "data", "04_ambiguity_detection", "JCM_random_1000_sample_evaluated.csv"
)
RAW_FILE = os.path.join(BASE_DIR, "data", "00_raw", "JCM_original.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "05_context_addition", "gt_trial_58")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "input_with_pairs_58.csv")


def main():
    # 1. 保存先ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. GT評価データの読み込み
    print(f"Loading evaluated data from: {GT_FILE}")
    if not os.path.exists(GT_FILE):
        print(f"Error: GT file not found at {GT_FILE}")
        return

    df_gt = pd.read_csv(GT_FILE)

    # gt_majorityがTrue（情報欠損あり）の行を抽出
    df_ambiguous = df_gt[df_gt["gt_majority"].astype(str).str.upper() == "TRUE"].copy()
    print(
        f"Found {len(df_ambiguous)} ambiguous samples based on 'gt_majority == True'."
    )

    # 3. 元データの読み込み（ペア文取得用）
    print(f"Loading raw data from: {RAW_FILE}")
    if not os.path.exists(RAW_FILE):
        print(f"Error: Raw file not found at {RAW_FILE}")
        return

    df_raw = pd.read_csv(RAW_FILE)
    # 【修正箇所】 カラム名を 'Original_ID' に変更
    raw_lookup = df_raw.set_index("Original_ID")[["sent", "label"]].to_dict("index")

    # 4. ペア情報の紐付け
    results = []
    for _, row in df_ambiguous.iterrows():
        target_id = int(row["Original_ID"])

        # ペアIDの計算 (1-2, 3-4, ... のペアリングルール)
        if target_id % 2 != 0:
            pair_id = target_id + 1
        else:
            pair_id = target_id - 1

        # 元データの辞書からペア文の情報を取得
        if pair_id in raw_lookup:
            pair_info = raw_lookup[pair_id]
            results.append(
                {
                    "Original_ID": target_id,
                    "target_sent": row["sent"],
                    "target_label": int(row["label"]),
                    "Pair_ID": pair_id,
                    "pair_sent": pair_info["sent"],
                    "pair_label": int(pair_info["label"]),
                }
            )
        else:
            print(
                f"Warning: Pair ID {pair_id} for Original_ID {target_id} not found in raw data."
            )

    # 5. 結果の保存
    df_result = pd.DataFrame(results)
    # ラベルの意味を付加（プロンプトでの理解を助けるため）
    df_result["target_label_str"] = df_result["target_label"].map(
        {0: "許容できる", 1: "許容できない"}
    )
    df_result["pair_label_str"] = df_result["pair_label"].map(
        {0: "許容できる", 1: "許容できない"}
    )

    df_result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Successfully created input dataset with pairs: {OUTPUT_FILE}")
    print(f"Total processed rows: {len(df_result)}")


if __name__ == "__main__":
    main()
