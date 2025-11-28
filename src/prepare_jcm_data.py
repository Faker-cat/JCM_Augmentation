# JCM_original.csvを読み込み、評価用のCSVファイルを作成するスクリプト
import os

import pandas as pd

# 入力ファイルと出力ファイルを、スクリプトの場所を基準に指定
input_file = "/home/faker/JCM_Augmentation/data/01_raw/JCM_original.csv"
output_file = (
    "/home/faker/JCM_Augmentation/data/02_prepared_for_eval/JCM_for_evaluation.csv"
)

# --- 💡 変更点: 評価者リストを定義 ---
# 評価者が追加された場合は、このリストに名前を追記してください。
EVALUATORS = ["A", "B", "C", "D", "E"]  # 初期設定
# ------------------------------------


def prepare_jcm_data(input_file, output_file, evaluators):
    print(f"1. ファイル '{input_file}' を読み込みます...")

    # 文字化け対策として、一般的な日本語エンコーディングを順に試行
    encodings_to_try = ["utf-8", "shift_jis", "cp932", "euc-jp"]
    df = None
    successful_encoding = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            successful_encoding = encoding
            break
        except Exception:
            continue

    if df is None:
        print(
            "エラー: どのエンコーディングでもファイルを正常に読み込めませんでした。ファイル名を確認してください。"
        )
        return

    # 2. 評価に必要な列を追加します

    # ID列の追加
    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, 1 + len(df)))

    # 評価者フラグ列の動的な追加
    evaluation_columns = ["ID", "sent"]
    for evaluator in evaluators:
        col_name = f"{evaluator}のフラグ"
        df[col_name] = ""
        evaluation_columns.append(col_name)

    print(f"2. {len(evaluators)}名分のフラグ列を追加しました。")

    # 3. 評価に必要な列のみを選択 (元のラベル列 'label' は除外)
    # 選択する列は、リスト評価者名に基づいて動的に生成されます
    try:
        evaluation_df = df[evaluation_columns].copy()
    except KeyError as e:
        print(f"エラー: データに '{e.args[0]}' という列名が見つかりません。")
        print("元のCSVの文章列名が 'sent' 以外の場合は、コードを修正してください。")
        return

    # 4. UTF-8エンコーディングでCSVファイルとして保存
    evaluation_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"\n✅ 完了: 評価用ファイル '{output_file}' が作成されました。")


if os.path.exists(input_file):
    prepare_jcm_data(input_file, output_file, EVALUATORS)
else:
    print(f"エラー: ファイル '{input_file}' が見つかりません。")
