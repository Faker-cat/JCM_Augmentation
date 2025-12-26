import argparse
import os
from pathlib import Path

import pandas as pd


def finalize_dataset():
    parser = argparse.ArgumentParser(
        description="JCM文脈付加データの最終統合と統計レポート作成を行います。"
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
    args = parser.parse_args()

    # パス設定
    model_name_dir = args.model_id.replace("/", "_")
    base_dir = (
        Path(os.getcwd())
        / "data/05_context_addition/gt_trial_58/outputs"
        / args.prompt_method
        / model_name_dir
    )
    input_dir = base_dir / "step2_feedback"
    output_dir = base_dir / "step3_final"
    report_dir = Path(os.getcwd()) / "data/05_context_addition/gt_trial_58/reports"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. 各ステップのファイルを読み込み
    # Step 1の結果が含まれるベースファイル
    df_step1 = pd.read_csv(input_dir / "annotation.csv", encoding="utf-8-sig")

    # Step 2の再生成結果（is_ok_step2フラグが含まれている想定）
    reg_path = input_dir / "regenerated.csv"
    df_step2 = (
        pd.read_csv(reg_path, encoding="utf-8-sig") if reg_path.exists() else None
    )

    # Step 3の人手修正結果
    man_path = input_dir / "manual_fix_needed.csv"
    df_step3 = (
        pd.read_csv(man_path, encoding="utf-8-sig") if man_path.exists() else None
    )

    # 2. 最終文章（final_sent）の決定ロジック
    final_rows = []
    counts = {"step1": 0, "step2": 0, "step3": 0}

    # annotation.csvに含まれる58件をベースに回す
    for _, row in df_step1.iterrows():
        oid = row["Original_ID"]
        final_sent = ""
        source = ""

        # A. 人手修正（Step 3）があるか確認
        if df_step3 is not None and oid in df_step3["Original_ID"].values:
            final_sent = df_step3.loc[
                df_step3["Original_ID"] == oid, "final_manual_sent"
            ].values[0]
            source = "step3"
            counts["step3"] += 1

        # B. 再生成（Step 2）で合格したか確認
        elif df_step2 is not None and oid in df_step2["Original_ID"].values:
            s2_row = df_step2[df_step2["Original_ID"] == oid].iloc[0]
            if s2_row.get("is_ok_step2") == 1:
                final_sent = s2_row["augmented_sent_step2"]
                source = "step2"
                counts["step2"] += 1
            else:
                # ここに来る場合はmanual_fix_neededにあるはずだが、念のため
                final_sent = row["augmented_sent"]
                source = "step1"
                counts["step1"] += 1

        # C. 初期生成（Step 1）で合格していた場合
        else:
            final_sent = row["augmented_sent"]
            source = "step1"
            counts["step1"] += 1

        # 指定された4つのカラムのみを保持
        final_rows.append(
            {
                "Original_ID": oid,
                "target_sent": row["target_sent"],
                "target_label_str": row["target_label_str"],
                "final_sent": final_sent,
            }
        )

    # DataFrame化
    final_df = pd.DataFrame(final_rows)

    # 3. CSVの保存
    final_csv_path = output_dir / "final.csv"
    final_df.to_csv(final_csv_path, index=False, encoding="utf-8-sig")

    # 4. 統計レポートの作成
    report_content = f"""JCM Context Addition Report
==================================================
Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {args.model_id}
Prompt Method: {args.prompt_method}
Total Records: {len(final_df)}

【Step-by-Step Breakdown】
- Step 1 (Initial LLM OK): {counts["step1"]} records
- Step 2 (Regenerated LLM OK): {counts["step2"]} records
- Step 3 (Manual Final Fix): {counts["step3"]} records
==================================================
Output file: {final_csv_path}
"""
    report_file_path = report_dir / f"report_{args.prompt_method}_{model_name_dir}.txt"
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"作成完了: {final_csv_path}")
    print(f"レポート出力: {report_file_path}")
    print("\n" + report_content)


if __name__ == "__main__":
    finalize_dataset()
