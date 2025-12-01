import json
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import LukeForTokenClassification, MLukeTokenizer, pipeline

# --- 設定 ---
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(root_dir, "original_data", "JCM_original.csv")
OUTPUT_FILE = os.path.join(root_dir, "original_data", "vocabulary_pools_hf.json")
TEXT_COLUMN = "sent"

# 使用するNERモデル
MODEL_NAME = "Mizuiro-sakura/luke-japanese-base-finetuned-ner"


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        return None


def extract_vocabulary_hf(df):
    print(f"Loading NER model: {MODEL_NAME}...")
    try:
        # LUKEモデルとトークナイザーのロード
        tokenizer = MLukeTokenizer.from_pretrained(MODEL_NAME)
        model = LukeForTokenClassification.from_pretrained(MODEL_NAME)

        # GPUの設定
        device = 0 if torch.cuda.is_available() else -1

        # パイプラインの作成
        # aggregation_strategy="simple" で、分割されたトークンを結合して出力
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    extracted_terms = {"when": set(), "where": set(), "who": set()}

    print("Extracting entities...")
    texts = df[TEXT_COLUMN].tolist()

    # 処理の進捗を表示
    for text in tqdm(texts):
        try:
            # LUKEモデルは入力長制限があるため、必要に応じて切り詰める
            # ただしLUKEのトークナイザーは文字数とトークン数が異なるため、安全マージンをとる
            processed_text = text[:256]

            entities = ner_pipeline(processed_text)

            for entity in entities:
                label = entity["entity_group"]
                word = entity["word"]

                # ノイズ除去
                if len(word) < 1:
                    continue

                # ラベルのマッピング
                # Mizuiro-sakura/luke-japanese-base-finetuned-ner のラベルセット:
                # LOC: Location, PER: Person, ORG: Organization, MISC: Miscellaneous

                # 時間表現(TIM, DATE)はこのモデルでは抽出されない可能性が高いですが、念のためマッピングに残します
                if label in ["DATE", "TIME", "TIM"]:
                    extracted_terms["when"].add(word)
                elif label in ["LOC", "GPE", "FAC"]:
                    extracted_terms["where"].add(word)
                elif label in ["PER", "PSN"]:
                    extracted_terms["who"].add(word)
                elif label in ["ORG"]:
                    # 組織は文脈によるが、主体者(Who)として扱う
                    extracted_terms["who"].add(word)

        except Exception:
            # print(f"Error processing text: {e}")
            continue

    # セットをリストに変換してソート
    return {k: sorted(list(v)) for k, v in extracted_terms.items()}


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    df = load_data(INPUT_FILE)
    if df is None:
        return

    vocab_pools = extract_vocabulary_hf(df)

    if vocab_pools:
        # 結果のプレビュー
        print("\n--- Extracted Vocabulary Pools (Preview) ---")
        for category, vocab_list in vocab_pools.items():
            print(f"Category: {category.upper()} ({len(vocab_list)} unique items)")
            print(f"Top 10: {vocab_list[:10]}")

        # 保存
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(vocab_pools, f, ensure_ascii=False, indent=4)
        print(f"\nSuccessfully saved vocabulary pools to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
