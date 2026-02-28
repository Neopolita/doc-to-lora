#!/bin/bash
set -euo pipefail

# =============================================================================
# End-to-end training entrypoint for Ministral-3-3B hypernetwork on HF Jobs
# =============================================================================
# This script handles:
#   1. Base data download & compact dataset building
#   2. FineWeb QA pair generation (using gemma-3-12b-it via vLLM)
#   3. Self-generated response data (using Ministral-3-3B via vLLM)
#   4. Hypernetwork training (8 GPU, accelerate)
#   5. Pushing trained model to HF Hub
# =============================================================================

MODEL="mistralai/Ministral-3-3B-Instruct-2512"
QA_GEN_MODEL="google/gemma-3-12b-it"
CONFIG="configs/main_exp/ministral-3b/self_gen_lv1_closed_qa_1_l2l.yaml"
HF_REPO="neopolita/doc-to-lora-ministral-3b-2512"
PORT=29051

echo "=========================================="
echo "Phase 0: Install dependencies"
echo "=========================================="

bash install.sh

# Force upgrade mistral_common for Ministral-3-3B tokenizer v13 support
# Needs >=1.9.0, but vLLM 0.8.5 pins an older version — force it
# Use pip directly to bypass uv's lockfile constraints
.venv/bin/pip install "mistral-common>=1.9.0" --force-reinstall --no-deps

echo "=========================================="
echo "Phase 1: Base data setup"
echo "=========================================="

# Download SQuAD
if [ ! -d "data/raw_datasets/squad" ]; then
    echo "Downloading SQuAD..."
    uv run huggingface-cli download --repo-type dataset rajpurkar/squad --local-dir data/raw_datasets/squad
fi

# Build compact datasets
echo "Building compact datasets..."
uv run data/build_drop_compact.py
uv run data/build_pwc_compact.py
uv run data/build_ropes_compact.py
uv run data/build_squad_compact.py

echo "=========================================="
echo "Phase 2: FineWeb QA pair generation"
echo "=========================================="

# Download FineWeb Edu
if [ ! -d "data/raw_datasets/fineweb_edu" ]; then
    echo "Downloading FineWeb Edu..."
    uv run data/download_fineweb_edu.py
fi

# Generate QA pairs from FineWeb using gemma-3-12b-it
echo "Generating FineWeb QA pairs (level 0 and 1)..."
for shard_id in $(seq -f "%03g" 0 13); do
    if [ ! -f "data/raw_datasets/fw_qa_v2/min_0_to_2000/${shard_id}"*level_1*.parquet ] 2>/dev/null; then
        uv run data/generate_fw_edu_qa_v2.py \
            --shard_pattern "${shard_id}_00000" \
            --n_qa_pairs=5 \
            --vllm_model="${QA_GEN_MODEL}" \
            --max_length=2000 \
            --max_model_length=2048
        uv run data/generate_fw_edu_qa_v2_repeat.py \
            --shard_pattern "min_0_to_2000/${shard_id}*level_0" \
            --n_qa_pairs=5 \
            --vllm_model="${QA_GEN_MODEL}"
    fi
done

echo "=========================================="
echo "Phase 3: Self-generated response data"
echo "=========================================="

# Self-gen FineWeb QA responses using Ministral-3-3B
echo "Generating self-gen responses for FineWeb QA..."
for shard_id in $(seq -f "%03g" 0 13); do
    uv run data/self_generate_qa.py \
        --vllm_model "${MODEL}" \
        --glob_pattern "data/raw_datasets/fw_qa_v2/min_0_to_2000/${shard_id}*_level_1*" \
        --closed_qa_prob 1.0
done

# Validation split
echo "Generating self-gen responses for validation..."
uv run data/self_generate_qa.py \
    --vllm_model "${MODEL}" \
    --glob_pattern 'data/raw_datasets/fw_qa_v2/min_0_to_2000/*_level_0_val.parquet'

# Self-gen for other datasets
echo "Generating self-gen responses for compact datasets..."
uv run data/self_generate_qa.py \
    --vllm_model "${MODEL}" \
    --ds_names squad_compact ropes_compact drop_compact \
    --split train --closed_qa_prob 1.0

uv run data/self_generate_qa.py \
    --vllm_model "${MODEL}" \
    --ds_names pwc_compact \
    --split train --closed_qa_prob 0.0

echo "=========================================="
echo "Phase 4: Training (with auto-push to HF Hub)"
echo "=========================================="

# HF_PUSH_REPO triggers automatic upload after training completes
export HF_PUSH_REPO="${HF_REPO}"

uv run accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_port $PORT \
    --num_processes=8 --gpu_ids all \
    train.py \
    "${CONFIG}" \
    --model_name_or_path="${MODEL}" \
    --target_modules=down_proj --lora_r=8 \
    --eval_strategy=no --max_qas_len=2048 --max_qas_per_sample=1 \
    --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 \
    --max_steps=80000 --gradient_accumulation_steps=8 --max_packed_inp_len=4096 \
    --max_packed_ctx_len=4096 --use_per_ctx_average_loss=True --use_kl_loss=True \
    --quantize_ctx_encoder=True

echo "=========================================="
echo "Done! Model pushed to https://huggingface.co/${HF_REPO}"
echo "=========================================="
