import math
import os
import time

import pandas as pd
from dotenv import load_dotenv  # 追加
from openai import APIError, OpenAI, RateLimitError
from tqdm import tqdm

# .envファイルを読み込む
load_dotenv()

# APIキーの設定
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# キーが読み込めていない場合の安全策
if not client.api_key:
    print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。")
    print(".envファイルを作成するか、exportコマンドで設定してください。")
    exit()

# データの読み込み
input_file = "data/01_raw/JCM_original.csv"
if not os.path.exists(input_file):
    print(f"エラー: 入力ファイルが見つかりません: {input_file}")
    exit()

df = pd.read_csv(input_file)
print(f"全データ数: {len(df)}件")

# --- 設定 (有料アカウント向けに高速化) ---
# Moderation APIは無料ですが、有料アカウント(Tier1以上)ならレート制限が緩和されます。
BATCH_SIZE = 50  # 一度に送る件数を増やす (推奨: 20~50)
MAX_RETRIES = 5  # リトライ回数


# --- Moderation APIにかける関数 (バッチ処理 & リトライ付き) ---
def check_moderation_batch(texts, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            # 配列(リスト)をそのまま渡してバッチ処理
            response = client.moderations.create(input=texts)
            return response.results

        except RateLimitError:
            # レート制限時: 指数バックオフ
            wait_time = (2**attempt) + 1
            print(
                f"\n[Rate Limit] Waiting {wait_time}s... (Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_time)

        except APIError as e:
            # その他のAPIエラー
            print(
                f"\n[API Error] {e}. Waiting 5s... (Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(5)

        except Exception as e:
            print(f"\n[Unexpected Error] {e}")
            return None

    print(f"\nFailed to process batch after {max_retries} attempts.")
    return None


# --- バッチ処理ループ ---
results = []

# バッチ数の計算
num_batches = math.ceil(len(df) / BATCH_SIZE)

print(f"処理を開始します (全 {num_batches} バッチ / {len(df)} 件)...")

for i in tqdm(range(num_batches)):
    start_idx = i * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE

    # バッチ切り出し
    batch_df = df.iloc[start_idx:end_idx]
    batch_texts = batch_df["sent"].tolist()

    # API呼び出し
    batch_results = check_moderation_batch(batch_texts)

    if batch_results:
        # 結果の紐付け
        for j, mod_res in enumerate(batch_results):
            original_row = batch_df.iloc[j]
            label = original_row["label"]
            text = original_row["sent"]
            original_idx = batch_df.index[j]

            # --- 分類ロジック ---
            # Type A: JCM=1(NG) だが API=Safe -> 潜在的な曖昧/情報欠落候補
            is_type_a = (label == 1) and (not mod_res.flagged)

            # Type B: JCM=0(OK) だが API=Flagged -> 文脈依存の暴力/過激表現候補
            is_type_b = (label == 0) and (mod_res.flagged)

            results.append(
                {
                    "ID": original_idx + 1,
                    "sent": text,
                    "original_label": label,
                    "moderation_flagged": mod_res.flagged,
                    "is_ambiguous_candidate": is_type_a,  # ターゲットA
                    "is_contextual_safety": is_type_b,  # ターゲットB
                    "scores": mod_res.category_scores.model_dump(),
                }
            )
    else:
        print(f"Skipping batch {i} due to errors.")

    # 待機時間 (有料アカウントなら短くてOK)
    time.sleep(0.5)

# --- 結果の保存 ---
result_df = pd.DataFrame(results)

# 保存先
output_dir = "data/02_prepared_for_eval"
os.makedirs(output_dir, exist_ok=True)

# 1. 全結果 (moderation_full_results.csv)
result_df.to_csv(
    os.path.join(output_dir, "moderation_full_results.csv"),
    index=False,
    encoding="utf-8",
)

# 2. ターゲットA (candidates_ambiguous_ng.csv)
candidates_a = result_df[result_df["is_ambiguous_candidate"] == True]
candidates_a.to_csv(
    os.path.join(output_dir, "candidates_ambiguous_ng.csv"),
    index=False,
    encoding="utf-8",
)

# 3. ターゲットB (candidates_contextual_ok.csv)
candidates_b = result_df[result_df["is_contextual_safety"] == True]
candidates_b.to_csv(
    os.path.join(output_dir, "candidates_contextual_ok.csv"),
    index=False,
    encoding="utf-8",
)

print("\n✅ 処理完了:")
print(f" - 全データ処理数: {len(result_df)} / {len(df)}")
print(
    f" - ターゲットA（曖昧なNG候補）: {len(candidates_a)}件 -> {output_dir}/candidates_ambiguous_ng.csv"
)
print(
    f" - ターゲットB（文脈依存OK候補）: {len(candidates_b)}件 -> {output_dir}/candidates_contextual_ok.csv"
)
