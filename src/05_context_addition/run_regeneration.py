# フィードバックに基づき、文章を再生成するスクリプト
import argparse
import os
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams


def run_regeneration():
    parser = argparse.ArgumentParser(
        description="人間からのフィードバックに基づき、文章を再生成します。"
    )
    parser.add_argument(
        "--prompt_method", type=str, required=True, help="手法名 (例: p01_pair_base)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="モデルID (例: tokyotech-llm/Llama-3.1-8B)",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="GPU数")
    args = parser.parse_args()

    # モデル名からディレクトリ名を作成 (例: tokyotech-llm/Llama-3.1-8B -> tokyotech-llm_Llama-3.1-8B)
    model_name_dir = args.model_id.replace("/", "_")

    # パスの設定 (ディレクトリ構造に準拠)
    base_dir = Path(os.getcwd())
    work_dir = (
        base_dir
        / "data/05_context_addition/gt_trial_58/outputs"
        / args.prompt_method
        / model_name_dir
    )
    input_csv = work_dir / "step2_feedback/annotation.csv"
    output_csv = work_dir / "step2_feedback/regenerated.csv"

    if not input_csv.exists():
        print(f"Error: 入力ファイルが見つかりません: {input_csv}")
        return

    # データの読み込み
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    # is_ok == 0 (要修正) の行のみを抽出
    target_df = df[df["is_ok"] == 0].copy()

    if len(target_df) == 0:
        print("再生成が必要なデータはありません（すべて is_ok == 1 です）。")
        return

    print(f"再生成対象: {len(target_df)} 件")

    # プロンプトの構築
    prompts = []
    for _, row in target_df.iterrows():
        # 骨子の制約 に基づくプロンプト
        refinement_prompt = f"""
あなたは道徳判断データセットの品質改善を行うエキスパートです。
以前、以下の文章を生成しましたが、人間から「{row["feedback_comment"]}」という修正指示がありました。

【元の文】: {row["target_sent"]}
【前回の生成】: {row["augmented_sent"]}
【目指すべきラベル】: {row["target_label_str"]}

以下の制約を「厳守」して、文章を修正してください：
1. 出力は40文字以内で出力することを厳守してください。
2. 句点（。）を含めず、必ず「一文のみ」で出力すること。
3. 主語（私、彼など）を入れないこと。
4. 元のラベル（{row["target_label_str"]}）が明確に定まる文脈にすること。

修正後の文章のみを出力し、ほかには一切のものを出力しないでください。
"""
        prompts.append(refinement_prompt.strip())

    # モデルのロードと推論 (run_augmentation.py と同様の設定)
    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

    outputs = llm.generate(prompts, sampling_params)

    # 生成結果の抽出
    regenerated_sents = [output.outputs[0].text.strip() for output in outputs]
    target_df["augmented_sent_step2"] = regenerated_sents

    # 保存先ディレクトリの作成（念のため）
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 結果の保存 (utf-8-sigでExcel等での文字化けを防止)
    target_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"再生成結果を保存しました: {output_csv}")


if __name__ == "__main__":
    run_regeneration()
