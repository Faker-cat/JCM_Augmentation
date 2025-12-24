# src/04_ambiguity_detection/fewshot_experiment/make_fewshot_data.py
# few-shot 学習用のデータセットを作成するスクリプト
import os

import pandas as pd

MASTER_GT = "data/04_ambiguity_detection/JCM_random_1000_sample_evaluated.csv"
OUT_DIR = "data/04_ambiguity_detection/fewshot"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(MASTER_GT)

# gt_majority を基準に True 5件を確保
df_true = df[df["gt_majority"] == True].sample(n=5, random_state=42)
df_false = df[df["gt_majority"] == False].sample(n=95, random_state=42)

df_train = pd.concat([df_true, df_false]).sample(frac=1, random_state=42)  # シャッフル
df_test = df.drop(df_train.index)

df_train.to_csv(os.path.join(OUT_DIR, "train_100.csv"), index=False)
df_test.to_csv(os.path.join(OUT_DIR, "test_900.csv"), index=False)

print(
    f"Split completed: Train={len(df_train)} (True={df_train['gt_majority'].sum()}), Test={len(df_test)}"
)
