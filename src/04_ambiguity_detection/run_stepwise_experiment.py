# /home/faker/JCM_Augmentation/src/04_ambiguity_detection/run_stepwise_experiment.py
#  idea2: 段階判定による情報欠損検出実験スクリプト
import argparse
import os

import pandas as pd
import torch

# 既存のスクリプトから必要な関数や定数をインポート
from run_classification_experiment import (
    DATA_DIR,
    MODEL_ID,
    TEMPERATURE,
    check_phase1_filtering,
    generate_prompt,
    load_and_merge_data,
    load_model,
    load_prompt_template,
)
from tqdm import tqdm


def get_stepwise_prediction(model, tokenizer, prompt_template, text, label):
    """
    【idea2: 段階判定】
    LLMに思考ステップを出力させ、最終判定を抽出する
    """
    prompt = generate_prompt(prompt_template, text, label)
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
        model.device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # 思考プロセスを出力させるため、長めに設定
            do_sample=False,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.replace(prompt, "").strip()

    # パース処理：最後に出現する「TRUE」または「FALSE」を探す
    upper_res = response.upper()
    # 3. 最終判定：の後ろを確認
    if "最終判定" in response:
        final_part = response.split("最終判定")[-1].upper()
        if "TRUE" in final_part:
            return True, response
        elif "FALSE" in final_part:
            return False, response

    # 見つからない場合は全体からキーワードマッチ
    if "TRUE" in upper_res:
        return True, response
    return False, response


def run_stepwise_experiment(args):
    """実験の実行ループ（run_classification_experiment.py の run_experiment を idea2 用に調整）"""
    print(f"=== 実験開始: Stepwise Detection (Model: {MODEL_ID}) ===")

    prompt_template = load_prompt_template(args.prompt_path)
    full_output_dir = os.path.join(DATA_DIR, "results", args.output_dir)

    model, tokenizer = load_model()
    df = load_and_merge_data()

    # ★ ここを追加：最初の10件のみに制限する
    df = df.head(10)

    results = []
    print("推論実行中（段階判定）...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row["sent"]
        label = row["original_label"]

        # Phase 1: Moderationフィルタリングは既存ロジックを継承
        if check_phase1_filtering(row):
            prediction = False
            phase = "Phase 1 (Filter)"
            raw_response = "Filtered"
        else:
            # Phase 2: 今回の目玉である段階判定
            prediction, raw_response = get_stepwise_prediction(
                model, tokenizer, prompt_template, text, label
            )
            phase = "Phase 2 (LLM-Stepwise)"

        results.append(
            {
                "Original_ID": row["Original_ID"],
                "sent": text,
                "gt_is_ambiguous": row["gt_is_ambiguous"],
                "pred_is_ambiguous": prediction,
                "phase": phase,
                "llm_raw_response": raw_response,
            }
        )

    # 結果の保存とレポート生成
    df_res = pd.DataFrame(results)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    output_csv = os.path.join(full_output_dir, "stepwise_detection_results.csv")
    df_res.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"結果を保存しました: {output_csv}")


# --- 直接実行するための main ブロック ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="idea2: 段階判定による情報欠損検出")
    parser.add_argument(
        "--prompt_path", type=str, required=True, help="プロンプトのパス"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="結果保存ディレクトリ名"
    )

    args = parser.parse_args()
    run_stepwise_experiment(args)
