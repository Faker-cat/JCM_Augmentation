# JCM_original.csvを読み込み、評価用のCSVファイルを作成するスクリプト
import os

import pandas as pd

# 入力ファイルと出力ファイルを、スクリプトの場所を基準に指定
input_file = "/home/faker/JCM_Augmentation/data/01_raw/JCM_original.csv"
output_file = (
    "/home/faker/JCM_Augmentation/data/02_prepared_for_eval/JCM_for_evaluation.csv"
)

# 評価者リスト
EVALUATORS = ["A", "B", "C", "D", "E"]


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

    # --- 💡 変更点1: シャッフル前に「元のID」を付与 ---
    # これにより、後で元の並び順に戻すことが可能になります。
    if "Original_ID" not in df.columns:
        df["Original_ID"] = range(1, 1 + len(df))
    # ---------------------------------------------

    # --- 💡 変更点2: データをランダムにシャッフル ---
    # frac=1 で全データを抽出（シャッフル）。
    # random_state=42 を指定（再現性の確保）。
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("   -> データをランダムに並べ替えました。")
    # ---------------------------------------------

    # 2. 評価に必要な列を追加します

    # ID列（評価用ID）の追加
    # シャッフル後にIDを振ることで、評価用IDは1から順に並びますが、中身はランダムになります。
    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, 1 + len(df)))

    # 評価者フラグ列の動的な追加
    # Original_IDを含めることで、分析時にソート可能にします。
    evaluation_columns = ["ID", "Original_ID", "sent"]
    for evaluator in evaluators:
        col_name = f"{evaluator}のフラグ"
        df[col_name] = ""
        evaluation_columns.append(col_name)

    print(f"2. {len(evaluators)}名分のフラグ列を追加しました。")

    # 3. 評価に必要な列のみを選択 (元のラベル列 'label' は除外)
    try:
        evaluation_df = df[evaluation_columns].copy()
    except KeyError as e:
        print(f"エラー: データに '{e.args[0]}' という列名が見つかりません。")
        print("元のCSVの文章列名が 'sent' 以外の場合は、コードを修正してください。")
        return

    # 4. UTF-8エンコーディングでCSVファイルとして保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    evaluation_df.to_csv(output_file, index=False, encoding="utf-8")

    print(
        "\n✅ 完了: ランダム順に並べ替え、Original_IDを付与した評価用ファイルを作成しました。"
    )
    print(f"出力先: '{output_file}'")


if os.path.exists(input_file):
    prepare_jcm_data(input_file, output_file, EVALUATORS)
else:
    print(f"エラー: ファイル '{input_file}' が見つかりません。")
