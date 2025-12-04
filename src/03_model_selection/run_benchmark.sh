#!/bin/bash

# スクリプトのエラーで停止しないように設定（必要に応じて set -e に変更）
set +e

# ソースコードのパス
SRC_PATH="/home/faker/JCM_Augmentation/src/03_model_selection/eval_zeroshot_jcm.py"

# --- 実行設定 ---

# 1. llm-jp-3.1-13b (GPU 1枚で動作)
# echo "Running evaluation for llm-jp-3.1-13b..."
# python $SRC_PATH \
#     --model_path "llm-jp/llm-jp-3.1-13b-instruct4" \
#     --tensor_parallel_size 1
#     --limit 5

# 2. Qwen3-Coder-30B (GPU 1枚で動作)
# echo "Running evaluation for Qwen3-Coder-30B..."
# python $SRC_PATH \
#     --model_path "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
#     --tensor_parallel_size 1 \
#     --gpu_memory_utilization 0.95 \
#     --limit 5 Qwen/Qwen2.5-Coder-32B-Instruct

# CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 poetry run python $SRC_PATH \
#     --model_path "Qwen/Qwen2.5-32B-Instruct-AWQ" \
#     --tensor_parallel_size 2 \
#     --gpu_memory_utilization 0.95 \
#     --limit 5

# # 3. Llama-3.1-Swallow-70B (GPU 2枚推奨)
echo "Running evaluation for Llama-3.1-Swallow-70B..."
python $SRC_PATH \
    --model_path "tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3" \
    --tensor_parallel_size 2 \
    --limit 5

# # 4. Qwen3-Next-80B (GPU 2~3枚推奨)
# echo "Running evaluation for Qwen3-Next-80B..."
# python $SRC_PATH \
#     --model_path "Qwen/Qwen3-Next-80B-A3B-Instruct" \
#     --tensor_parallel_size 2

# # 5. gpt-oss-120b (GPU 3枚推奨 ※モデルサイズ次第で2枚でもいける可能性あり)
# echo "Running evaluation for gpt-oss-120b..."
# python $SRC_PATH \
#     --model_path "openai/gpt-oss-120b" \
#     --tensor_parallel_size 3

# 新しいモデルを追加
# echo "Running evaluation for NEW-MODEL..."
# python $SRC_PATH \
#     --model_path "新しいモデルのパス" \
#     --tensor_parallel_size 1

echo "All evaluations finished! Check data/03_model_selection/*_debug folder."