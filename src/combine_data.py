# JCM_Augmentation/src/combine_data.py
import os
import sys

import pandas as pd

# スクリプトの実行ディレクトリを基準にパスを構築
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root_dir, "data")
original_data_dir = os.path.join(root_dir, "original_data")
output_file = os.path.join(original_data_dir, "JCM_original.csv")

# 結合対象のファイル名
files_to_combine = ["data_train.csv", "data_val.csv", "data_test.csv"]
all_data = []

print("Starting data combination process...")

# データの読み込みと結合
for file_name in files_to_combine:
    file_path = os.path.join(data_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Error: Required file not found at {file_path}")
        print(
            "Please ensure your data_train.csv, data_val.csv, and data_test.csv are inside the 'data' directory."
        )
        sys.exit(1)

    # CSVファイルを読み込む。最初の無名カラム（元のインデックス）は無視する
    try:
        df = pd.read_csv(file_path, header=0, index_col=0)
        # 必要なカラム（'sent'と'label'）のみを保持
        if "sent" not in df.columns or "label" not in df.columns:
            print(
                f"Error: {file_name} does not contain expected columns 'sent' and 'label'. Columns found: {df.columns.tolist()}"
            )
            sys.exit(1)
        all_data.append(df[["sent", "label"]])
        print(f"Loaded {file_name} with {len(df)} rows.")

    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        sys.exit(1)

# すべてのデータを縦に結合し、インデックスを振り直す
combined_df = pd.concat(all_data, ignore_index=True)

# 出力ディレクトリが存在しない場合は作成
os.makedirs(original_data_dir, exist_ok=True)

# 結合したデータフレームを新しいCSVファイルとして保存（インデックスなし）
combined_df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Successfully combined {len(files_to_combine)} files into {output_file}")
print(f"Total rows in combined file: {len(combined_df)}")

print(
    "\nNext step: You can now proceed to the data augmentation process using src/augment_data.py."
)
