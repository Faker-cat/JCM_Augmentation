import os

import pandas as pd

# --- 設定 ---
# プロジェクトルートからの相対パスで定義
EVAL_DIR = "data/02_prepared_for_eval"
OUTPUT_DIR = "data/03_merged_results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "JCM_MERGED_EVALUATIONS.csv")

# 💡 評価者が増えたら、このリストに名前を追加
EVALUATORS = ["A", "B", "C"]
TOTAL_EVALUATORS = len(EVALUATORS)
MAJORITY_THRESHOLD = (TOTAL_EVALUATORS // 2) + 1  # 過半数 (例: 3人中 2人, 5人中 3人)

# 統合対象ファイルをフルパスで構成
EVALUATION_FILES = [
    os.path.join(EVAL_DIR, f"{evaluator}_evaluation.csv") for evaluator in EVALUATORS
]

ID_COLUMN = "ID"
SENTENCE_COLUMN = "sent"
# ------------


def safe_merge_evaluations(file_list, output_file, id_col, sent_col, evaluators):
    print("1. 評価データの統合を開始します...")

    base_df = None

    # 全てのファイルをIDをキーに結合していく
    for i, file_name in enumerate(file_list):
        if not os.path.exists(file_name):
            print(
                f"警告: ファイル '{file_name}' が見つかりません。パスを確認してください。スキップします。"
            )
            continue

        try:
            # 💡 Google SheetsからダウンロードされたCSVは余分な改行コードやバイト列を含む場合があるため、low_memory=False
            df_new = pd.read_csv(file_name, encoding="utf-8", low_memory=False)
        except Exception as e:
            print(
                f"エラー: ファイル '{file_name}' の読み込み中に問題が発生しました。エンコーディングを確認してください。エラー: {e}"
            )
            continue

        # 評価者のフラグ列を特定
        # NOTE: 結合処理によって新しいフラグ列名 (例: Aのフラグ, Bのフラグ) が作成される可能性があるため、
        # ここではベースファイルが持つ可能性のあるすべてのフラグ列名パターンを除外して、対象の列を識別しています。
        flag_col = [
            col for col in df_new.columns if "のフラグ" in col and col != f"{id_col}_x"
        ]

        if not flag_col:
            print(
                f"警告: ファイル '{file_name}' にフラグ列が見つかりません。スキップします。"
            )
            continue

        # 保持する列を選択: ID, 文章(最初のみ), フラグ列
        cols_to_keep = [id_col] + flag_col

        if base_df is None:
            # 最初のファイルの場合、IDと文章列をベースとして使用
            if sent_col not in df_new.columns:
                print(
                    f"エラー: ベースファイル '{file_name}' に文章列 '{sent_col}' が見つかりません。処理を中断します。"
                )
                return
            cols_to_keep.append(sent_col)
            base_df = df_new[cols_to_keep].copy()
            print(f"   -> ベースファイルとして '{file_name}' を設定しました。")

        else:
            # 2つ目以降のファイルは、IDをキーにベースデータに結合
            df_to_merge = df_new[cols_to_keep].copy()
            base_df = pd.merge(base_df, df_to_merge, on=id_col, how="left")
            print(f"   -> '{file_name}' の評価結果を統合しました。")

    if base_df is None:
        print("エラー: 統合できるファイルが見つかりませんでした。")
        return

    # --- 2. 分析項目の追加 (最も重要な処理) ---
    print("\n2. 分析項目（TRUE判定数、過半数フラグなど）を計算します...")

    # 💡 修正箇所: フラグ列名からアンダースコア '_' を削除
    # 実際のファイル内の列名 ('Aのフラグ', 'Bのフラグ'...) に合わせます。
    flag_columns = [f"{evaluator}のフラグ" for evaluator in evaluators]

    # 実際にbase_dfに存在するフラグ列名のみを抽出します
    existing_flag_columns = [
        col
        for col in base_df.columns
        if "のフラグ" in col and col.replace("_x", "").replace("_y", "") in flag_columns
    ]

    if len(existing_flag_columns) != TOTAL_EVALUATORS:
        print(
            f"警告: 統合された列数 ({len(existing_flag_columns)}) が評価者数 ({TOTAL_EVALUATORS}) と一致しません。"
        )
        print("列名を確認してください。続行しますが、結果が不正確な可能性があります。")

    # 💡 TRUE/FALSE文字列を数値 (1/0) に変換
    # Google SheetsのチェックボックスはCSVで "TRUE" / "FALSE" になる
    for col in existing_flag_columns:
        # 大文字・小文字を無視してTRUEを1、FALSEを0、その他（空欄など）を0に変換
        base_df[col] = base_df[col].astype(str).str.upper().str.strip()
        base_df[col] = base_df[col].apply(lambda x: 1 if x == "TRUE" else 0)

    # 1. 曖昧判定数 (TRUEと判断した評価者の合計人数) を計算
    base_df["TRUE判定数"] = base_df[existing_flag_columns].sum(axis=1)

    # 2. 評価者数
    base_df["評価者数"] = TOTAL_EVALUATORS

    # 3. TRUE判定割合 (合意率の指標)
    base_df["TRUE判定割合"] = base_df["TRUE判定数"] / base_df["評価者数"]

    # 4. 過半数TRUEフラグ
    # 必要な情報が欠落している (TRUE) と判断した評価者が過半数か (例: 3人中2人以上)
    base_df["過半数TRUEフラグ"] = (base_df["TRUE判定数"] >= MAJORITY_THRESHOLD).astype(
        int
    )

    # 5. 最低1人TRUEフラグ (一人でも欠落と判断)
    base_df["最低1人TRUEフラグ"] = (base_df["TRUE判定数"] >= 1).astype(int)

    print(f"   -> 判定数の計算が完了しました。過半数しきい値: {MAJORITY_THRESHOLD}人。")

    # --- 3. 最終列順の整理と保存 ---

    # 💡 新規追加: 理想的な列順を定義
    # IDと文章
    primary_cols = [id_col, sent_col]

    # 評価者フラグ列
    eval_flag_cols = existing_flag_columns

    # 分析・集計列
    analysis_cols = [
        "TRUE判定数",
        "TRUE判定割合",
        "評価者数",
        "過半数TRUEフラグ",
        "最低1人TRUEフラグ",
    ]

    # 最終的な列順を構築し、DataFrameを並び替え
    final_cols = primary_cols + eval_flag_cols + analysis_cols

    # DataFrameにある列だけを選択（エラー防止のため）
    final_cols_safe = [col for col in final_cols if col in base_df.columns]

    base_df = base_df[final_cols_safe]

    # 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    base_df.to_csv(output_file, index=False, encoding="utf-8")

    print(
        f"\n✅ 完了: 統合および分析項目が追加されたファイル '{output_file}' が作成されました。"
    )


# スクリプト実行
safe_merge_evaluations(
    EVALUATION_FILES, OUTPUT_FILE, ID_COLUMN, SENTENCE_COLUMN, EVALUATORS
)
