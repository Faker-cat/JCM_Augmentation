import pandas as pd
import os
from src.llm_helper import generate_llm_prompt, parse_llm_response
# from src.llm_helper import load_llm_model, query_llm # 実際のLLM利用時にコメントアウトを解除

# --- 設定 ---
INPUT_FILE = "original_data/JCM_original.csv"
OUTPUT_DIR = "results"
THREE_W_TARGETS = ["when", "where", "who"]
# JCMデータセットのカラム名
TEXT_COLUMN = "sent"
LABEL_COLUMN = "label"
# ----------------


# --- ダミー関数 (llm_helper.pyに本来あるが、ここでは実行環境の都合でローカルに定義) ---
def query_llm_dummy(prompt: str) -> str:
    """LLMへの問い合わせの代わりにダミー応答を返す関数"""
    # 特定の例に対する応答 (例: 欠落していると判断されるケース)
    if (
        "「いつ」の情報が加わることで" in prompt
        and "信号が赤信号だったため車のアクセルを踏んだ" in prompt
    ):
        return "YES\n深夜"
    if (
        "「だれが」の情報が加わることで" in prompt
        and "友人の皿に料理を取り分けようとした時" in prompt
    ):
        return "YES\n衛生観念のない知人が"

    # 欠落判断が「NO」になるデフォルト (例: 欠落していないと判断されるケース)
    if "信号が赤信号だったため車のブレーキを踏んだ" in prompt:
        return "NO\nN/A"

    # その他（ランダムな付加情報を返すダミーロジック）
    if "YES" in prompt:
        if "いつ" in prompt:
            return "YES\n夜中の３時"
        if "どこで" in prompt:
            return "YES\n薄暗い路地裏で"
        if "だれが" in prompt:
            return "YES\n近所のおじさんが"

    return "NO\nN/A"


# ---------------------------------------------------------------------------------------


def load_data(file_path: str) -> pd.DataFrame:
    """データセットをロードする"""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        # 必要なカラムのみを抽出
        if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
            raise ValueError(
                f"Required columns '{TEXT_COLUMN}' or '{LABEL_COLUMN}' not found."
            )
        return df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None


def augment_data(df_original: pd.DataFrame, target_3w: str):
    """
    特定の3W情報のみを付加する処理を実行する。
    """
    df_augmented = df_original.copy()

    # 付加情報格納用の新しいカラム名
    augmentation_col = f"added_{target_3w}_info"
    df_augmented[augmentation_col] = None

    # LLMモデルのロード（実際の実装時に使用）
    # tokenizer, llm_model = load_llm_model("tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5")

    print(f"\n--- Starting augmentation for: {target_3w} ---")

    total_rows = len(df_augmented)
    augmented_count = 0

    for index, row in df_augmented.iterrows():
        text = row[TEXT_COLUMN]
        label = row[LABEL_COLUMN]

        # 1. プロンプト生成
        prompt = generate_llm_prompt(text, label, target_3w)

        # 2. LLMに問い合わせ
        # 実際の実装では: response_text = query_llm(prompt, tokenizer, llm_model)
        response_text = query_llm_dummy(prompt)  # ダミー応答を使用

        # 3. 応答の解析
        is_missing, augmentation_text = parse_llm_response(response_text)

        # 4. 欠落判断が「YES」の場合のみ付加情報を保存
        if is_missing == "YES" and augmentation_text != "N/A":
            df_augmented.loc[index, augmentation_col] = augmentation_text
            augmented_count += 1

        if (index + 1) % 100 == 0 or index + 1 == total_rows:
            print(
                f"Processed {index + 1}/{total_rows} rows. Added info count: {augmented_count}"
            )

    print(
        f"--- Augmentation for {target_3w} completed. Total added annotations: {augmented_count} ---"
    )
    return df_augmented


def save_data(df: pd.DataFrame, target_3w: str):
    """結果をCSVファイルに保存する"""
    output_filename = f"JCM_{target_3w}_augmented.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # output_dataディレクトリが存在しない場合は作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 元データに付加情報を加えた形で保存
    print(f"Saving augmented data to {output_path}...")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print("Save complete.")


def main():
    """メイン処理"""

    if not os.path.exists(INPUT_FILE):
        print(
            f"\n[CRITICAL ERROR] The required input file '{INPUT_FILE}' was not found."
        )
        print(
            "Please combine your JCM data into one file and save it as 'JCM_original.csv' inside the 'original_data' folder."
        )
        return

    df_original = load_data(INPUT_FILE)
    if df_original is None:
        return

    # 3つの情報（when, where, who）について独立して処理を実行
    for target_3w in THREE_W_TARGETS:
        # 元のデータセットに対して、特定の3W情報のみを付加
        df_result = augment_data(df_original, target_3w)

        # 結果を保存（例：results/JCM_when_augmented.csv）
        save_data(df_result, target_3w)

    print("\nAll augmentation processes completed.")
    print(
        "Next Step: Use the generated CSV files in the 'results' folder for human annotation and analysis."
    )


if __name__ == "__main__":
    main()  # 実行環境によっては、この行をコメントアウトし、手動で実行してください。
