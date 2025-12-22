# 混合行列の各要素ごとにデータを分割し、個別のCSVファイルとして保存するスクリプト
from pathlib import Path

import pandas as pd

# ===== パス設定 =====
base_dir = Path(
    "/home/faker/JCM_Augmentation/data/04_ambiguity_detection/results/tokyotech-llm_Llama-3.3-Swallow-70B-Instruct-v0.4/classification/v02_base/"
)

input_csv = base_dir / "results.csv"
output_dir = base_dir / "confusion_matrix_split"
output_dir.mkdir(exist_ok=True)

# ===== CSV 読み込み =====
df = pd.read_csv(input_csv)

# ===== 混合行列の分割 =====
tp = df[(df["gt_is_ambiguous"] == True) & (df["pred_is_ambiguous"] == True)]
fp = df[(df["gt_is_ambiguous"] == False) & (df["pred_is_ambiguous"] == True)]
fn = df[(df["gt_is_ambiguous"] == True) & (df["pred_is_ambiguous"] == False)]
tn = df[(df["gt_is_ambiguous"] == False) & (df["pred_is_ambiguous"] == False)]

# ===== 保存 =====
tp.to_csv(output_dir / "TP_gt_true_pred_true.csv", index=False)
fp.to_csv(output_dir / "FP_gt_false_pred_true.csv", index=False)
fn.to_csv(output_dir / "FN_gt_true_pred_false.csv", index=False)
tn.to_csv(output_dir / "TN_gt_false_pred_false.csv", index=False)

# ===== 中身の確認（件数＋先頭数件）=====
print("=== Confusion Matrix Split Summary ===")
for name, subset in {
    "TP": tp,
    "FP": fp,
    "FN": fn,
    "TN": tn,
}.items():
    print(f"\n[{name}] count = {len(subset)}")
    print(subset.head(3))
