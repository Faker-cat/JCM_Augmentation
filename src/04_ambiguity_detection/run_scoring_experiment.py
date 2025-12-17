# src/04_ambiguity_detection/run_scoring_experiment.py
import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

# 既存の分類スクリプトから関数をインポート
from run_classification_experiment import (
    check_phase1_filtering,
    generate_prompt,
    load_and_merge_data,
    load_model,
    load_prompt_template,
)
from tqdm import tqdm

# 設定
MODEL_ID = "llm-jp/llm-jp-3.1-13b-instruct4"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "04_ambiguity_detection")
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.1


def get_llm_score(model, tokenizer, prompt_template, text, label):
    """LLMからスコア(0-10)を取得する"""
    prompt = generate_prompt(prompt_template, text, label)
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
        model.device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = (
        tokenizer.decode(outputs[0], skip_special_tokens=True)
        .replace(prompt, "")
        .strip()
    )

    # 正規表現で「スコア: 7」あるいは単なる数値部分を抽出
    match = re.search(r"(\d+)", response)
    if match:
        score = int(match.group(1))
        return min(max(score, 0), 10), response  # 0-10の範囲に収める
    return None, response


def run_experiment(args):
    full_output_dir = os.path.join(DATA_DIR, "results", args.output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    prompt_template = load_prompt_template(args.prompt_path)
    model, tokenizer = load_model()
    df = load_and_merge_data()

    results = []
    print("スコアリング実行中...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Phase 1フィルタを通過したものは除外
        if check_phase1_filtering(row):
            continue

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

    # --- 追加機能: スコアごとのTRUE/FALSE集計 ---
    # クロス集計表の作成 (行: スコア, 列: 地上実況ラベル)
    summary_table = pd.crosstab(
        df_res["predicted_score"], df_res["gt_is_ambiguous"], dropna=False
    ).reindex(range(11), fill_value=0)  # 0-10まで全てのスコアを網羅

    # 各スコアにおけるTRUEの割合（精度/相関の確認用）
    summary_table["Total"] = summary_table.sum(axis=1)
    summary_table["True_Ratio"] = summary_table[True] / summary_table["Total"]

    # --- レポート作成 ---
    report = []
    report.append("=== 曖昧度スコアリング実験レポート ===")
    report.append(f"Model: {MODEL_ID}")
    report.append(f"Prompt: {os.path.basename(args.prompt_path)}")
    report.append("-" * 30)
    report.append("Score Distribution Summary (Count of Ground Truth Labels):")
    report.append(summary_table.to_string())  # 表をテキスト形式で追加
    report.append("-" * 30)

    report_text = "\n".join(report)
    print("\n" + report_text)

    # --- 保存処理 ---
    # 1. 詳細な全件結果CSV
    output_csv = os.path.join(full_output_dir, "scoring_results.csv")
    df_res.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # 2. 集計表CSV (卒論の表として貼り付けやすい)
    summary_csv = os.path.join(full_output_dir, "score_summary_table.csv")
    summary_table.to_csv(summary_csv, encoding="utf-8-sig")

    # 3. テストレポート
    report_path = os.path.join(full_output_dir, "experiment_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # 4. 可視化 (ヒストグラム)
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_res,
        x="predicted_score",
        hue="gt_is_ambiguous",
        multiple="stack",
        bins=11,
        binrange=(0, 11),
    )
    plt.title("Ambiguity Score Distribution\n(Stacked by Ground Truth)")
    plt.xlabel("LLM Predicted Ambiguity Score (0-10)")
    plt.ylabel("Sample Count")
    plot_path = os.path.join(full_output_dir, "score_distribution.png")
    plt.savefig(plot_path)

    print(f"結果を {full_output_dir} に保存しました。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args)
