import argparse
import os
from pathlib import Path

import pandas as pd


def finalize_dataset_cat12():
    parser = argparse.ArgumentParser(
        description="Category 1&2限定の統合と統計レポート作成テスト"
    )
    parser.add_argument(
        "--prompt_method", type=str, required=True, help="手法名 (例: p01_pair_base)"
    )
    parser.add_argument("--model_id", type=str, required=True, help="モデルID")
    args = parser.parse_args()

    # パス設定
    model_name_dir = args.model_id.replace("/", "_")
    base_dir = (
        Path(os.getcwd())
        / "data/05_context_addition/gt_trial_58/outputs"
        / args.prompt_method
        / model_name_dir
    )
    input_csv = (
        Path(os.getcwd())
        / "data/05_context_addition/gt_trial_58/inputs/input_with_pairs_58.csv"
    )
    input_dir = base_dir / "step2_feedback"
    report_dir = Path(os.getcwd()) / "data/05_context_addition/gt_trial_58/reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. データの読み込み
    df_input = pd.read_csv(input_csv, encoding="utf-8-sig")
    # Category 1 & 2 のみを抽出 (ambiguity_type カラムが追加されている前提)
    target_oids = df_input[df_input["ambiguity_type"].isin([1, 2])][
        "Original_ID"
    ].tolist()

    df_step1 = pd.read_csv(input_dir / "annotation.csv", encoding="utf-8-sig")

    reg_path = input_dir / "regenerated.csv"
    df_step2 = (
        pd.read_csv(reg_path, encoding="utf-8-sig") if reg_path.exists() else None
    )

    man_path = input_dir / "manual_fix_needed.csv"
    df_step3 = (
        pd.read_csv(man_path, encoding="utf-8-sig") if man_path.exists() else None
    )

    # 2. 集計ロジック (Category 1 & 2 の ID のみを対象にする)
    counts = {"step1": 0, "step2": 0, "step3": 0}

    # フィルタリングされた行のみでループ
    df_filtered = df_step1[df_step1["Original_ID"].isin(target_oids)]

    for _, row in df_filtered.iterrows():
        oid = row["Original_ID"]

        # A. 人手修正（Step 3）
        if df_step3 is not None and oid in df_step3["Original_ID"].values:
            counts["step3"] += 1

        # B. 再生成（Step 2）
        elif df_step2 is not None and oid in df_step2["Original_ID"].values:
            s2_row = df_step2[df_step2["Original_ID"] == oid].iloc[0]
            if s2_row.get("is_ok_step2") == 1:
                counts["step2"] += 1
            else:
                counts["step1"] += 1  # 再生成失敗によりStep1の状態

        # C. 初期生成（Step 1）
        else:
            counts["step1"] += 1

    # 3. 統計レポートの作成
    report_file_name = f"report_cat12_only_{args.prompt_method}_{model_name_dir}.txt"
    report_path = report_dir / report_file_name

    total_targeted = len(df_filtered)

    report_content = f"""JCM Context Addition Report (Category 1 & 2 Only)
==================================================
Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {args.model_id}
Prompt Method: {args.prompt_method}
Total Targeted Records (Cat 1&2): {total_targeted}

【Step-by-Step Breakdown】
- Step 1 (Initial LLM OK): {counts["step1"]} records
- Step 2 (Regenerated LLM OK): {counts["step2"]} records
- Step 3 (Manual Final Fix): {counts["step3"]} records
==================================================
Note: This report excludes Category 3 (Information Gap).
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"作成完了: {report_path}")
    print("\n" + report_content)


if __name__ == "__main__":
    finalize_dataset_cat12()
