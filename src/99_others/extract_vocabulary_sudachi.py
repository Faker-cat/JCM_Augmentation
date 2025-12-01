import json
import os

import pandas as pd
from sudachipy import dictionary, tokenizer

# --- 設定 ---
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(root_dir, "original_data", "JCM_original.csv")
OUTPUT_FILE = os.path.join(root_dir, "original_data", "vocabulary_pools_sudachi.json")
TEXT_COLUMN = "sent"


def extract_vocabulary_sudachi(df):
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C

    extracted_terms = {"when": set(), "where": set(), "who": set()}

    for text in df[TEXT_COLUMN]:
        tokens = tokenizer_obj.tokenize(text, mode)

        for token in tokens:
            pos = token.part_of_speech()
            surface = token.surface()

            # 品詞情報に基づく簡易的な分類
            # 注: 辞書の定義に依存するため、完璧ではありません

            # When: 名詞,普通名詞,副詞可能 / 名詞,時相名詞
            if pos[0] == "名詞" and (
                pos[1] == "時相名詞" or (pos[1] == "普通名詞" and pos[2] == "副詞可能")
            ):
                # ノイズ除去（短い語や記号などを除外）
                if len(surface) > 1:
                    extracted_terms["when"].add(surface)

            # Where: 名詞,固有名詞,地名 / 名詞,普通名詞,場所
            # Sudachiの品詞体系では「場所」が明示的でない場合があるため、キーワードマッチも併用推奨
            if pos[0] == "名詞" and (pos[1] == "固有名詞" and pos[2] == "地名"):
                extracted_terms["where"].add(surface)
            elif pos[0] == "名詞" and surface in [
                "学校",
                "家",
                "公園",
                "病院",
                "店",
                "部屋",
            ]:  # 一般的な場所名詞
                extracted_terms["where"].add(surface)

            # Who: 名詞,固有名詞,人名 / 代名詞
            if pos[0] == "名詞" and (pos[1] == "固有名詞" and pos[2] == "人名"):
                extracted_terms["who"].add(surface)
            elif pos[0] == "代名詞" and surface not in [
                "これ",
                "それ",
                "あれ",
                "どれ",
                "ここ",
                "そこ",
                "あそこ",
                "どこ",
            ]:  # 人称代名詞
                extracted_terms["who"].add(surface)
            elif surface in [
                "友達",
                "先生",
                "母",
                "父",
                "子供",
                "上司",
                "部下",
            ]:  # 一般的な役割名詞
                extracted_terms["who"].add(surface)

    return {k: sorted(list(v)) for k, v in extracted_terms.items()}


def main():
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    vocab_pools = extract_vocabulary_sudachi(df)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab_pools, f, ensure_ascii=False, indent=4)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
