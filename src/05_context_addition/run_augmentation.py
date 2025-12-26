import argparse
import os

import pandas as pd
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--prompt_set_name", type=str, required=True)
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/05_context_addition/gt_trial_58/inputs/input_with_pairs_58.csv",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # パスの設定
    base_dir = os.getcwd()
    prompt_path = os.path.join(
        base_dir, "src", "05_context_addition", "prompts", f"{args.prompt_set_name}.txt"
    )

    model_name_safe = args.model_id.replace("/", "_")
    output_dir = os.path.join(
        base_dir,
        "data",
        "05_context_addition",
        "gt_trial_58",
        "results",
        model_name_safe,
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.prompt_set_name}.csv")

    # 1. プロンプトテンプレートの読み込み
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    # 2. データの読み込み
    df = pd.read_csv(args.input_path)

    # 3. プロンプトの構築
    prompts = []
    for _, row in df.iterrows():
        p = template.format(
            target_sent=row["target_sent"],
            target_label_str=row["target_label_str"],
            pair_sent=row["pair_sent"],
            pair_label_str=row["pair_label_str"],
        )
        prompts.append(p)

    # 4. モデルのロードと生成 (vLLM)
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

    outputs = llm.generate(prompts, sampling_params)

    # 5. 結果の抽出と保存
    generated_texts = [
        output.outputs[0].text.strip().replace("シナリオA:", "").strip()
        for output in outputs
    ]
    df["augmented_sent"] = generated_texts  # 文章Aを補完した新しい文章

    # 不要なカラムを整理して保存
    output_cols = [
        "Original_ID",
        "target_sent",
        "target_label",
        "augmented_sent",
        "Pair_ID",
        "pair_sent",
        "pair_label",
    ]
    df[output_cols].to_csv(output_file, index=False, encoding="utf-8-sig")

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
