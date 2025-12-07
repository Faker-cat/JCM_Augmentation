# 曖昧文章データを用いて情報付加を試すスクリプト（vLLM版）
import argparse
import os

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Augment JCM data using LLM")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Hugging Face model path",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/03_model_selection/曖昧文章_ランダム10件.csv",
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/03_model_selection/augmentation_test",
        help="Directory to save results",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
    )
    # メモリ不足対策のためのフラグ
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable CUDA graph capturing to save memory",
    )
    return parser.parse_args()


def create_prompt(text, label, tokenizer):
    """
    QwenなどのInstructモデル用にチャットテンプレートを適用してプロンプトを作成する
    """

    # ラベルの意味を言語化
    if label == 0:
        label_str = "許容できる (OK)"
        instruction = "この行為が「道徳的に問題ない（許容される）」と誰でも納得できるような、正当な理由や背景事情を補ってください。"
    else:
        label_str = "許容できない (NG)"
        instruction = "この行為が「道徳的に悪い（許容されない）」と誰でも判断できるような、悪意や迷惑な状況を補ってください。"

    system_message = "あなたは日本語の編集者です。与えられた短い文章に対して、道徳的な判断が一意に定まるように情報を補足し、リライトしてください。"

    user_message = f"""以下の「元の文章」は、情報不足のため道徳的な判断が困難です。
元の文章の文脈を維持しつつ、指定された「正解ラベル」の通りに判断できるよう、必要な前提条件や状況を書き加えてください。

# 制約事項
1. 出力は「修正後の文章」のみにしてください。解説や「はい、承知しました」などの返事は不要です。
2. 元の文章の核となる行動は変えないでください。
3. 日本語として自然な文章にしてください。

# 入力情報
元の文章: {text}
正解ラベル: {label_str}
指示: {instruction}

# 修正後の文章:"""

    # チャットテンプレートの適用 (リスト形式で渡す)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    # tokenizerを使ってプロンプトを整形 (add_generation_prompt=Trueでアシスタントの開始タグまで付ける)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def main():
    args = parse_args()

    # 1. データ読み込み
    print(f"Loading data from {args.input_path}...")
    try:
        df = pd.read_csv(args.input_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. モデルとTokenizerの準備
    print(f"Initializing model: {args.model_path}...")

    # プロンプト作成用にTokenizerを先にロード (HuggingFaceのキャッシュディレクトリも考慮)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # プロンプトの構築
    # labelカラムとsentカラムを使用
    prompts = []
    for _, row in df.iterrows():
        prompt = create_prompt(row["sent"], row["label"], tokenizer)
        prompts.append(prompt)

    # vLLMの初期化
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        enforce_eager=args.enforce_eager,  # メモリ対策設定を反映
    )

    # サンプリングパラメータ (少し創造性を持たせるためtemperatureを入れるが、あまり高くしすぎない)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,  # 文脈付与なら少し長めの方が安全かもしれません
        stop=["<|eot_id|>", "<|end_of_text|>"],  # Llama-3用に修正
    )

    # 3. 推論実行
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # 4. 結果の整形
    generated_texts = [output.outputs[0].text.strip() for output in outputs]

    # 結果をデータフレームに追加
    df["augmented_text"] = generated_texts

    # 5. 保存
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(args.model_path))

    # ファイル名にモデル名を付与
    output_filename = f"{model_name}_augmented_results_10samples.csv"
    output_file = os.path.join(args.output_dir, output_filename)

    # 見やすいようにカラム順序を整理
    cols = ["Original_ID", "label", "sent", "augmented_text"]
    # もしOriginal_IDが無い場合のエラー回避
    available_cols = [c for c in cols if c in df.columns]
    # 足りないカラムがあれば追加したdf全体を保存
    if len(available_cols) < len(cols):
        df.to_csv(output_file, index=False)
    else:
        df[available_cols].to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
