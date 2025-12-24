# src/04_ambiguity_detection/fewshot_experiment/run_fewshot.py
import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- パス設定 ---
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_DIR = os.path.join(BASE_DIR, "data", "04_ambiguity_detection")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
FEWSHOT_DATA_DIR = os.path.join(DATA_DIR, "fewshot")


def load_model(model_id):
    print(f"Loading model: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tokenizer


def create_fewshot_prefix(train_df, n_true=3, n_false=3):
    """
    100件の学習用データから例示（Shot）を抽出
    """
    shots_true = train_df[train_df["gt_majority"] == True].sample(
        n=n_true, random_state=42
    )
    shots_false = train_df[train_df["gt_majority"] == False].sample(
        n=n_false, random_state=42
    )
    shots = pd.concat([shots_true, shots_false]).sample(frac=1, random_state=42)

    prefix = "\n### 判定の参考例:\n"
    for _, row in shots.iterrows():
        label_text = "許容できる" if row["label"] == 0 else "許容できない"
        ans = "[TRUE]" if row["gt_majority"] else "[FALSE]"
        prefix += f"文章: 「{row['sent']}」\nラベル: {label_text}\n判定: {ans}\n---\n"
    return prefix


def run_experiment(args):
    # 出力パスの構築 (results/[MODEL]/fewshot/[PROMPT])
    safe_model_name = args.model_id.replace("/", "_")
    prompt_name = os.path.splitext(os.path.basename(args.prompt_path))[0]
    output_dir = os.path.join(RESULTS_DIR, safe_model_name, "fewshot", prompt_name)
    os.makedirs(output_dir, exist_ok=True)

    # データのロード
    train_df = pd.read_csv(os.path.join(FEWSHOT_DATA_DIR, "train_100.csv"))
    test_df = pd.read_csv(os.path.join(FEWSHOT_DATA_DIR, "test_900.csv"))

    # モデルのロード
    model, tokenizer = load_model(args.model_id)

    method_folder = os.path.basename(os.path.dirname(args.prompt_path))
    prompt_file_name = os.path.splitext(os.path.basename(args.prompt_path))[0]

    # 保存用のファイル名を作成 (例: classification_v01_base.txt)
    save_filename = f"{method_folder}_{prompt_file_name}.txt"

    # 指示文と例示の準備
    with open(args.prompt_path, "r", encoding="utf-8") as f:
        instruction = f.read().strip()
    fewshot_prefix = create_fewshot_prefix(train_df)

    # --- 追加: 最終的なプロンプト(ベース+例示)を保存する処理 ---
    fewshot_prompt_dir = os.path.join(
        BASE_DIR, "src", "04_ambiguity_detection", "prompts", "fewshot"
    )
    os.makedirs(fewshot_prompt_dir, exist_ok=True)

    # 最終的なプロンプトの構成を確認用に作成 (判定対象部分はダミー)
    full_prompt_template = (
        f"{instruction}\n"
        f"{fewshot_prefix}\n"
        f"### 判定対象:\n"
        f"文章: 「[ここに評価対象の文章が入ります]」\n"
        f"ラベル: [許容できる/許容できない]\n"
        f"判定:"
    )

    # 指定されたファイル名で保存
    prompt_save_path = os.path.join(fewshot_prompt_dir, save_filename)
    with open(prompt_save_path, "w", encoding="utf-8") as f:
        f.write(full_prompt_template)

    print(f"Few-shot prompt template saved to: {prompt_save_path}")
    # ---------------------------------------------------

    results = []
    print(f"Starting Few-shot inference: {prompt_name} (Test size: {len(test_df)})")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        label_text = "許容できる" if row["label"] == 0 else "許容できない"

        full_prompt = (
            f"{instruction}\n"
            f"{fewshot_prefix}\n"
            f"### 判定対象:\n"
            f"文章: 「{row['sent']}」\n"
            f"ラベル: {label_text}\n"
            f"判定:"
        )

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        # token_type_ids エラー対策
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = (
            tokenizer.decode(outputs[0], skip_special_tokens=True)
            .replace(full_prompt, "")
            .strip()
        )

        pred = True if "TRUE" in response.upper() else False

        results.append(
            {
                "Original_ID": row["Original_ID"],
                "sent": row["sent"],
                "gt_is_ambiguous": row["gt_majority"],
                "pred_is_ambiguous": pred,
                "llm_raw_response": response,
            }
        )

    # CSV保存
    res_df = pd.DataFrame(results)
    res_df.to_csv(
        os.path.join(output_dir, "fewshot_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    args = parser.parse_args()
    run_experiment(args)
