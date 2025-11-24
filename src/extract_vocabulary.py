import json
import os
import re
import sys
from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm  # 処理の進捗を表示するためにtqdmをインポート

# --- GiNZA/spaCyのインポートとロードの試行 ---
# 環境に依存するため、try-exceptでエラーを回避
try:
    import spacy

    # GiNZAモデルのロード (環境に合わせてモデル名を変更してください)
    NLP = spacy.load("ja_ginza")
    print("✅ spaCy (GiNZA) model loaded successfully.")
    IS_NLP_READY = True
except Exception as e:
    print(f"❌ Error loading GiNZA/spaCy. Model loading error: {e}")
    print("⚠️ Falling back to Regex-based extraction (Lower precision).")
    IS_NLP_READY = False
# -----------------------------

# --- 設定 ---
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(root_dir, "original_data", "JCM_original.csv")
OUTPUT_FILE = os.path.join(root_dir, "original_data", "vocabulary_pools.json")
TEXT_COLUMN = "sent"
LABEL_COLUMN = "label"
PREVIEW_COUNT = 10
DEMO_COUNT = 100  # デモ用に処理する行数

# GiNZAのNERラベルと3Wカテゴリのマッピング
# 論文執筆時には、どのラベルがどの3Wに該当するかを明確に記述することを推奨
NER_MAPPING = {
    "when": ["DATE", "TIME", "EVENT"],
    "where": ["LOC", "FACILITY", "GPE"],
    "who": ["PERSON", "NORP", "ORG", "PRODUCT"],
}
# ----------------


def load_data(file_path: str) -> pd.DataFrame:
    """データセットをロードする"""
    try:
        df = pd.read_csv(file_path)
        if TEXT_COLUMN not in df.columns:
            print(f"Error: Required column '{TEXT_COLUMN}' not found.")
            sys.exit(1)
        return df
    except FileNotFoundError:
        print(
            f"Error: Input file not found at {file_path}. Please run src/combine_data.py first to create JCM_original.csv."
        )
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        sys.exit(1)


def extract_vocabulary_regex_fallback(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    GiNZAが使用できない場合のフォールバックロジック（正規表現ベース）。
    """
    when_vocab: Set[str] = set()
    where_vocab: Set[str] = set()
    who_vocab: Set[str] = set()

    # 抽出ルール
    when_keywords = r"(^|\s|[^、。])(夜中|深夜|朝|昼|夕方|午前|午後|年|月|日|分|週間|ヶ月|正月|クリスマス|今日|明日|先週|先月|昨日|明日|来年|祝日|連休)(に|で|の|には|まで|から|前|$)"
    where_keywords = r"(^|\s|[^、。])(学校|病院|道路|公園|自宅|家|ホテル|美術館|駅|職場|パチンコ|バー|店舗|店内|路上|会議室|教室内|山|海|空地|近所|風呂|浴槽|庭|デパート|テーブル)(で|に|へ|から|の|には|$)"
    who_keywords = r"(^|\s|[^、。])(友達|母親|父親|上司|部下|子供|生徒|老人|おばあさん|運転手|犯人|客|彼氏|妻|先生|同僚|誰|知人|娘|息子|家族|隣人|姉|兄|自分)(が|に|と|の|には|から|$)"

    for text in df[TEXT_COLUMN]:
        matches_when = re.findall(when_keywords, text)
        for match in matches_when:
            when_vocab.add(match[1].strip())

        matches_where = re.findall(where_keywords, text)
        for match in matches_where:
            where_vocab.add(match[1].strip())

        matches_who = re.findall(who_keywords, text)
        for match in matches_who:
            who_vocab.add(match[1].strip())

    if "誰" in who_vocab:
        who_vocab.remove("誰")

    return {
        "when": sorted(list(when_vocab)),
        "where": sorted(list(where_vocab)),
        "who": sorted(list(who_vocab)),
    }


def extract_vocabulary_nlp(df: pd.DataFrame) -> Dict[str, List[str]]:
    # GiNZAがロードできなかった場合のフォールバックロジックは省略（前回のコードのまま使用）
    if not IS_NLP_READY:
        print(
            "\n--- ⚠️ WARNING: GiNZA is unavailable. Executing Regex Fallback Extraction ---"
        )
        return extract_vocabulary_regex_fallback(df)  # フォールバックロジックの呼び出し

    # GiNZAが利用可能な場合のプライマリロジック
    extracted_terms: Dict[str, Set[str]] = {key: set() for key in NER_MAPPING.keys()}
    texts = df[TEXT_COLUMN].tolist()

    # 処理件数が1件の場合はtqdmを使用しない
    use_tqdm = len(texts) > 1
    iterable = (
        tqdm(NLP.pipe(texts, disable=["tagger", "parser"]), total=len(texts))
        if use_tqdm
        else NLP.pipe(texts, disable=["tagger", "parser"])
    )

    for doc in iterable:
        for ent in doc.ents:
            for category, labels in NER_MAPPING.items():
                if ent.label_ in labels:
                    clean_text = ent.text.strip()
                    if clean_text and len(clean_text) < 20 and not clean_text.isdigit():
                        extracted_terms[category].add(clean_text)

    final_pools: Dict[str, List[str]] = {}
    for category, terms in extracted_terms.items():
        final_pools[category] = sorted(list(terms))

    return final_pools


def main():
    print(f"--- Starting vocabulary extraction for '{TEXT_COLUMN}' column ---")

    df = load_data(INPUT_FILE)
    if df is None:
        return

    # 語彙の抽出
    vocabulary_pools = extract_vocabulary_nlp(df)

    # 結果の表示（確認用）
    print("\n--- Extracted Vocabulary Pools (Preview) ---")
    for category, vocab_list in vocabulary_pools.items():
        print(f"Category: {category.upper()} ({len(vocab_list)} unique items)")
        print(
            f"Top {PREVIEW_COUNT} terms: {vocab_list[: min(len(vocab_list), PREVIEW_COUNT)]}"
        )

    # JSONファイルとして保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(vocabulary_pools, f, ensure_ascii=False, indent=4)

    print(f"\n--- SUCCESS: Vocabulary pools saved to {OUTPUT_FILE} ---")


def run_demo(num_rows=3):
    """少数のデータ（GiNZA使用）に対して抽出ロジックの動作を確認するデモ関数"""

    df_full = load_data(INPUT_FILE)
    if df_full is None:
        return

    # デモ対象のデータセット（最初のnum_rows件）
    df_demo = df_full.head(num_rows).copy()

    print(f"\n--- Starting GiNZA NER Demo (First {num_rows} rows) ---")

    all_extracted_terms = {"when": set(), "where": set(), "who": set()}

    # 1行ずつ処理し、その結果を個別に表示
    for index, row in df_demo.iterrows():
        text = row[TEXT_COLUMN]
        label = row[LABEL_COLUMN]

        # 1行のデータフレームを渡し、NER抽出を実行
        df_single = pd.DataFrame({TEXT_COLUMN: [text]})
        # NOTE: GiNZAがロードされていればGiNZA、そうでなければフォールバックロジックを実行
        extracted_pools = extract_vocabulary_nlp(df_single)

        print(f"\nRow {index} (Label: {label}): '{text}'")
        for category, vocab_list in extracted_pools.items():
            if vocab_list:
                print(f"  -> {category.upper()} Found: {vocab_list}")
                all_extracted_terms[category].update(vocab_list)  # 統合プールへの追加
            # else:
            #     print(f"  -> {category.upper()} Found: []") # 抽出語彙がない場合

    # 全てのデータからの統合された抽出結果を表示
    print("\n--- Consolidated Vocabulary Pool Summary ---")
    for category, terms in all_extracted_terms.items():
        final_pools = sorted(list(terms))
        print(f"Category: {category.upper()} ({len(final_pools)} unique items)")
        print(f"Sample: {final_pools}")

    print("\n✅ Demo finished. GiNZA NER extraction result confirmed.")


if __name__ == "__main__":
    main()  # 通常の全件抽出
    # run_demo()  # 最初の5件をデモとして実行
