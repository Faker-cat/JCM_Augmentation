# JCM_Augmentation/src/00_original_data/combine_data.py
# "data_train.csv", "data_val.csv", "data_test.csv"から、JCM_original.csvを作成するスクリプト

import os

import pandas as pd

# 1. プロジェクトルートの特定
# スクリプトの場所が "src/01_preprocessing/script.py" の場合、
# ルート(JCM_Augmentation)に行くには3回遡る必要があります。
# (配置場所に合わせて dirname の回数を調整してください)
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/01_preprocessing
src_dir = os.path.dirname(current_dir)  # src
root_dir = os.path.dirname(src_dir)  # JCM_Augmentation

# 2. パスの設定
# 新しい構造: data/00_raw/ に元データと作成するJCM_original.csvを配置
raw_data_dir = os.path.join(root_dir, "data", "00_raw")
output_file = os.path.join(raw_data_dir, "JCM_original.csv")

# 結合対象のファイル名
files_to_combine = ["data_train.csv", "data_val.csv", "data_test.csv"]
all_data = []

# --- 読み込みと結合処理の例 ---
print(f"Reading files from: {raw_data_dir}")

for file_name in files_to_combine:
    file_path = os.path.join(raw_data_dir, file_name)

    if os.path.exists(file_path):
        print(f"  - Loading {file_name}...")
        try:
            # 読み込み
            df = pd.read_csv(file_path)
            all_data.append(df)
        except Exception as e:
            print(f"    Error loading {file_name}: {e}")
    else:
        print(f"    Warning: File not found {file_path}")

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)

    # --- ユーザー要望対応: IDの刷新 ---

    # 1. 元のファイルに含まれていた古いID列("Unnamed: 0")があれば削除
    if "Unnamed: 0" in combined_df.columns:
        combined_df.drop(columns=["Unnamed: 0"], inplace=True)
        print("  - Removed old 'Unnamed: 0' column.")

    # 2. 新しく 'Original_ID' 列を作成し、全件通し番号(1から開始)を付与
    # insert(0, ...) で一番左の列に追加します
    combined_df.insert(0, "Original_ID", range(1, 1 + len(combined_df)))
    print(
        f"  - Added new 'Original_ID' column with sequential numbers (1 to {len(combined_df)})."
    )

    # 保存に必要なディレクトリがなければ作成
    os.makedirs(raw_data_dir, exist_ok=True)

    combined_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n✅ Created: {output_file}")
    print(f"   Total rows: {len(combined_df)}")
    print(f"   Columns: {list(combined_df.columns)}")
else:
    print("\n❌ No data combined.")
