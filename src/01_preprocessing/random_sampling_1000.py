import os

import pandas as pd

# --- 設定 ---
INPUT_FILE = "/home/faker/JCM_Augmentation/data/00_raw/JCM_original.csv"
OUTPUT_FILE = (
    "/home/faker/JCM_Augmentation/data/01_ground_truth/JCM_random_1000_sample_ex.csv"
)

# サンプリングする件数
SAMPLE_SIZE = 1000

# 評価者リスト (必要に応じて変更してください)
EVALUATORS = ["A", "B", "C", "D", "E"]
# -----------


def create_random_sample_csv():
    print(f"1. ファイル '{INPUT_FILE}' を読み込みます...")

    # 文字化け対策
    encodings_to_try = ["utf-8", "shift_jis", "cp932", "euc-jp"]
    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(INPUT_FILE, encoding=encoding)
            break
        except Exception:
            continue

    if df is None:
        print("エラー: ファイルを読み込めませんでした。")
        return

    print(f"   全データ数: {len(df)}件")

    # --- 💡 ポイント1: サンプリング前に「元のID」を付与 ---
    # これにより、後で元の並び順（または元のファイルの場所）を特定できます
    if "Original_ID" not in df.columns:
        df["Original_ID"] = range(1, 1 + len(df))

    # --- 💡 ポイント2: ランダムに1000件サンプリング ---
    if len(df) < SAMPLE_SIZE:
        print(
            f"警告: データ数({len(df)})がサンプリング数({SAMPLE_SIZE})より少ないため、全データを使用します。"
        )
        sampled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        sampled_df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    print(f"2. ランダムに {len(sampled_df)} 件を抽出しました。")

    # --- ポイント3: 評価用IDの振り直しと列の整理 ---

    # 新しい評価用ID (1〜1000) を先頭に追加
    sampled_df.insert(0, "ID", range(1, 1 + len(sampled_df)))

    # 保存したい列のリスト作成 (ID, Original_ID, 文章)
    # ※ここで 'label' を含めないことで、元ラベルを除外します
    target_columns = ["ID", "Original_ID", "sent"]

    # 評価者の入力欄を追加
    for evaluator in EVALUATORS:
        col_name = f"{evaluator}のフラグ"
        sampled_df[col_name] = ""  # 空欄で作成
        target_columns.append(col_name)

    # 必要な列だけを抽出
    try:
        final_df = sampled_df[target_columns].copy()
    except KeyError as e:
        print(f"エラー: 必要な列が見つかりません: {e}")
        return

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\n✅ 完了: サンプリングされた評価用ファイルを作成しました。")
    print(f"出力先: {OUTPUT_FILE}")
    print(f"データ構成:\n{final_df.head(3)}")


if __name__ == "__main__":
    create_random_sample_csv()
