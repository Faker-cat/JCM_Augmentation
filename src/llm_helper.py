import re
from typing import Tuple

# --- LLMとの連携に関する設定とダミー関数 ---
# ⚠️ 注意: 実際には、以下の「query_llm_dummy」関数を削除し、
# ご使用のLLM（例: tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5）
# をロードして推論を行う関数（例: query_llm）を実装してください。
# --------------------------------------------


def generate_llm_prompt(text: str, current_label: int, target_3w: str) -> str:
    """
    LLMに「情報欠落の判断」と「付加情報の生成」を依頼するプロンプトを生成する。

    Args:
        text (str): JCMの文章 (sent)
        current_label (int): 現在の道徳判断ラベル (0: 許容できる, 1: 許容できない)
        target_3w (str): 付加対象の情報 ('when', 'where', 'who')

    Returns:
        str: LLMに渡すプロンプト
    """
    if target_3w == "when":
        japanese_3w = "いつ"
        example_context = "朝食時、昼食時、帰宅時など"
    elif target_3w == "where":
        japanese_3w = "どこで"
        example_context = "学校の廊下、自宅のトイレ、デパートなど"
    elif target_3w == "who":
        japanese_3w = "だれが"
        example_context = "母親、知人、見知らぬ男性など"
    else:
        raise ValueError("Invalid target_3w")

    # ユーザーの要件を反映したプロンプト設計
    prompt = f"""
    あなたは、文章の道徳判断の文脈を詳細にするための情報（{japanese_3w}）を生成するAIです。

    以下の文章は、道徳判断データセット（JCM）からの抜粋です。
    文章: 「{text}」
    現在の道徳判断ラベル: {current_label}（0: 許容できる, 1: 許容できない）

    以下の2つのステップに従って、結果を改行区切りで出力してください。

    ---
    ステップ1: 欠落判断
    この文章に、「{japanese_3w}」の情報が加わることで、現在の道徳判断ラベル（{current_label}）が変化する可能性があるか（つまり、「{japanese_3w}」の情報が道徳判断に必要不可欠であるか）を判断してください。
    * 変化する可能性がある場合は **「YES」** と、変化する可能性が低い、または情報が既に含まれている場合は **「NO」** とのみ回答してください。

    ステップ2: 付加情報の生成
    * ステップ1の欠落判断が「YES」の場合、元の文章の道徳判断に影響を与えない、**最も中立的で一般的な「{japanese_3w}」の情報**を一つだけ生成してください。（例: {example_context}）
    * ステップ1の欠落判断が「NO」の場合、**「N/A」** と回答してください。

    【出力形式】
    欠落判断の結果 (YES または NO)
    付加情報 (中立的な「{japanese_3w}」の情報、または N/A)
    ---
    """
    return prompt


def parse_llm_response(response: str) -> Tuple[str, str]:
    """LLMの応答を解析して、欠落判断と付加情報を抽出する"""
    # 正規表現を使って、YES/NOとその次の行のテキストを抽出
    match = re.search(
        r"^(YES|NO)\s*\n\s*(.*)$", response.strip(), re.MULTILINE | re.IGNORECASE
    )

    if match:
        is_missing = match.group(1).strip().upper()
        augmentation_text = match.group(2).strip()
        # N/Aチェック
        if is_missing == "NO":
            augmentation_text = "N/A"

        return is_missing, augmentation_text

    # 解析失敗時は安全のため「付加しない」と判断
    return "NO", "N/A"
