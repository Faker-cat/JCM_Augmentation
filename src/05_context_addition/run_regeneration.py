from pathlib import Path

import pandas as pd

# 既存のヘルパー関数（llm_helperなど）があると仮定
# from src.99_others.llm_helper import call_llm


def run_regeneration(prompt_method, model_name):
    # パスの設定
    base_path = Path(
        f"data/05_context_addition/gt_trial_58/outputs/{prompt_method}/{model_name}"
    )
    input_csv = base_path / "step2_feedback/annotation.csv"
    output_csv = base_path / "step2_feedback/regenerated.csv"

    # データの読み込み
    df = pd.read_csv(input_csv)

    # is_ok == 0 (要修正) のものだけを抽出
    target_df = df[df["is_ok"] == 0].copy()
    print(f"再生成対象: {len(target_df)} 件")

    regenerated_sents = []

    for _, row in target_df.iterrows():
        # フィードバック内容を強調したプロンプトの構成
        # 骨子の制約 [cite: 37-46] をベースに、個別の指示を追加
        refinement_prompt = f"""
あなたは道徳判断データセットの品質改善を行うエキスパートです。
以前、以下の文章を生成しましたが、人間から「{row["feedback_comment"]}」という修正指示がありました。

【元の文】: {row["target_sent"]}
【前回の生成】: {row["augmented_sent"]}
【目指すべきラベル】: {row["target_label_str"]}

以下の制約を「厳守」して、文章を修正してください：
1. 40文字以内の日本語であること。
2. 句点（。）を含めず、必ず「一文のみ」で出力すること。
3. 主語（私、彼など）を入れないこと。
4. 元のラベル（{row["target_label_str"]}）が明確に定まる文脈にすること。

修正後の文章のみを出力してください。
"""
        # LLM呼び出し（仮の関数名）
        # new_sent = call_llm(model_name, refinement_prompt)
        new_sent = "LLMからの回答例"  # 実際にはここをLLM呼び出しに置換
        regenerated_sents.append(new_sent)

    target_df["augmented_sent_step2"] = regenerated_sents

    # 結果の保存
    target_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"再生成結果を保存しました: {output_csv}")


if __name__ == "__main__":
    # 実行例
    run_regeneration("p01_pair_base", "tokyotech-llm_Llama-3.1-8B")
