# src/04_ambiguity_detection/run_scoring_experiment.py
# idea1: 曖昧度スコアリング実験を実行するスクリプト
import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

# 既存の分類スクリプトから共通ロジックをインポート
from run_classification_experiment import (
    DATA_DIR,
    TEMPERATURE,
    check_phase1_filtering,
    generate_prompt,
    load_and_merge_data,
    load_model,
    load_prompt_template,
)
from tqdm import tqdm


def get_llm_score(model, tokenizer, prompt_template, text, label):
    """LLMからスコア(0-10)を取得する"""
    prompt = generate_prompt(prompt_template, text, label)
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
        model.device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,  # スコアリング用に少し長めに設定
            do_sample=False,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = (
        tokenizer.decode(outputs[0], skip_special_tokens=True)
        .replace(prompt, "")
        .strip()
    )

    # 正規表現で数値部分を抽出
    match = re.search(r"(\d+)", response)
    if match:
        score = int(match.group(1))
        return min(max(score, 0), 10), response
    return None, response


def run_experiment(args):
    # 1. パスの自動生成
    safe_model_name = args.model_id.replace("/", "_")
    prompt_name = os.path.splitext(os.path.basename(args.prompt_path))[0]
    full_output_dir = os.path.join(
        DATA_DIR, "results", safe_model_name, "scoring", prompt_name
    )
    os.makedirs(full_output_dir, exist_ok=True)

    # 2. ロード
    prompt_template = load_prompt_template(args.prompt_path)
    model, tokenizer = load_model(args.model_id)
    df = load_and_merge_data()

    results = []
    phase1_count = 0
    print(f"=== 実験開始: Scoring / {args.model_id} / {prompt_name} ===")

    # 3. 推論実行
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if check_phase1_filtering(row):
            phase1_count += 1
            continue  # Phase 1で除外されたものはスコアリング対象外

        score, raw_res = get_llm_score(
            model, tokenizer, prompt_template, row["sent"], row["original_label"]
        )
        results.append(
            {
                "Original_ID": row["Original_ID"],
                "sent": row["sent"],
                "gt_is_ambiguous": row["gt_is_ambiguous"],
                "predicted_score": score,
                "llm_raw_response": raw_res,
            }
        )

    df_res = pd.DataFrame(results).dropna(subset=["predicted_score"])

    # 4. 集計処理
    summary_table = pd.crosstab(
        df_res["predicted_score"], df_res["gt_is_ambiguous"], dropna=False
    ).reindex(range(11), fill_value=0)

    summary_table["Total"] = summary_table.sum(axis=1)
    summary_table["True_Ratio"] = summary_table[True] / summary_table["Total"]

    # 5. レポート作成
    report = [
        "=== 曖昧度スコアリング実験レポート ===",
        f"Model: {args.model_id}",
        f"Prompt: {os.path.basename(args.prompt_path)}",
        f"Temperature: {TEMPERATURE}",
        f"Total Samples: {len(df)}",
        f"Phase 1 Filtered (Excluded): {phase1_count}",
        f"Scored Samples: {len(df_res)}",
        "-" * 30,
        "Score Distribution Summary (Count of Ground Truth Labels):",
        summary_table.to_string(),
        "-" * 30,
    ]

    report_text = "\n".join(report)
    print("\n" + report_text)

    # 6. 保存処理
    df_res.to_csv(
        os.path.join(full_output_dir, "scoring_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    summary_table.to_csv(
        os.path.join(full_output_dir, "score_summary_table.csv"), encoding="utf-8-sig"
    )
    with open(
        os.path.join(full_output_dir, "experiment_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report_text)

    # 7. 可視化
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_res,
        x="predicted_score",
        hue="gt_is_ambiguous",
        multiple="stack",
        bins=11,
        binrange=(0, 11),
    )
    plt.title(f"Ambiguity Score Distribution\n({args.model_id} / {prompt_name})")
    plt.xlabel("LLM Predicted Ambiguity Score (0-10)")
    plt.ylabel("Sample Count")
    plt.savefig(os.path.join(full_output_dir, "score_distribution.png"))

    print(f"結果を保存しました: {full_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args)
