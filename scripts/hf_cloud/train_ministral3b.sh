#!/bin/bash
set -euo pipefail

# =============================================================================
# End-to-end training entrypoint for Ministral-3-3B hypernetwork on HF Jobs
# =============================================================================
# This script handles:
#   1. Base data download & compact dataset building
#   2. Self-generated response data (using Ministral-3-3B via vLLM)
#   3. Hypernetwork training (4 GPU, accelerate)
#   4. Pushing trained model to HF Hub
# =============================================================================

MODEL="mistralai/Ministral-3-3B-Instruct-2512"
CONFIG="configs/main_exp/ministral-3b/compact_only.yaml"
HF_REPO="neopolita/doc-to-lora-ministral-3b-2512"
PORT=29051

echo "=========================================="
echo "Phase 0: Install dependencies"
echo "=========================================="

bash install.sh

# Upgrade mistral_common for Ministral-3-3B tokenizer v13 support
# Use pip directly to bypass uv's lockfile constraints (uv run re-syncs)
.venv/bin/pip install "mistral-common>=1.9.0" --force-reinstall --no-deps
.venv/bin/pip install pydantic-extra-types

# Patch vLLM 0.8.5 for Ministral-3-3B compatibility
.venv/bin/python << 'PATCH'
import os

# 1. Skip fake_quantizer keys in llama weight loader
p = '.venv/lib/python3.10/site-packages/vllm/model_executor/models/llama.py'
lines = open(p).readlines()
out = []
for line in lines:
    if line.strip() == 'param = params_dict[name]':
        indent = len(line) - len(line.lstrip())
        sp = ' ' * indent
        out.append(sp + 'if name not in params_dict:\n')
        out.append(sp + '    continue\n')
    out.append(line)
open(p, 'w').writelines(out)
print('Patched llama.py: skip unknown weight keys')

# 2. Remove skip_special_tokens=False assert in Mistral tokenizer
p = '.venv/lib/python3.10/site-packages/vllm/transformers_utils/tokenizers/mistral.py'
t = open(p).read()
old = '''    assert (
            skip_special_tokens
        ), "skip_special_tokens=False is not supported for Mistral tokenizers."'''
t = t.replace(old, '    skip_special_tokens = True  # patched: force True for compat')
open(p, 'w').write(t)
print('Patched mistral.py: allow skip_special_tokens=False')
PATCH

echo "=========================================="
echo "Phase 1: Base data setup"
echo "=========================================="

# Use .venv/bin directly (not uv run) to prevent uv from
# re-syncing the environment and reverting the mistral_common upgrade

# Download SQuAD
if [ ! -d "data/raw_datasets/squad" ]; then
    echo "Downloading SQuAD..."
    .venv/bin/huggingface-cli download --repo-type dataset rajpurkar/squad --local-dir data/raw_datasets/squad
fi

# Build compact datasets
echo "Building compact datasets..."
.venv/bin/python data/build_drop_compact.py
.venv/bin/python data/build_pwc_compact.py
.venv/bin/python data/build_ropes_compact.py
.venv/bin/python data/build_squad_compact.py

echo "=========================================="
echo "Phase 1.5: Extract text-only language model"
echo "=========================================="

# Ministral-3-3B is packaged as a multimodal model (Mistral3ForConditionalGeneration).
# vLLM 0.8.5 can't load it directly. Extract the text-only MistralForCausalLM.
BF16_MODEL="models/ministral-3b-bf16"
.venv/bin/python scripts/extract_language_model.py \
    --model "${MODEL}" --output "${BF16_MODEL}"

echo "=========================================="
echo "Phase 2: Self-generated response data"
echo "=========================================="

echo "Generating self-gen responses for compact datasets (10% subset)..."
.venv/bin/python data/self_generate_qa.py \
    --vllm_model "${MODEL}" \
    --vllm_model_path "${BF16_MODEL}" \
    --ds_names squad_compact ropes_compact drop_compact \
    --split train --closed_qa_prob 1.0 --max_samples 1500

.venv/bin/python data/self_generate_qa.py \
    --vllm_model "${MODEL}" \
    --vllm_model_path "${BF16_MODEL}" \
    --ds_names pwc_compact \
    --split train --closed_qa_prob 0.0 --max_samples 500

echo "=========================================="
echo "Phase 3: Training (with auto-push to HF Hub)"
echo "=========================================="

# HF_PUSH_REPO triggers automatic upload after training completes
export HF_PUSH_REPO="${HF_REPO}"

.venv/bin/accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_port $PORT \
    --num_processes=4 --gpu_ids all \
    train.py \
    "${CONFIG}" \
    --model_name_or_path="${MODEL}" \
    --target_modules=down_proj --lora_r=8 \
    --eval_strategy=no --max_qas_len=2048 --max_qas_per_sample=1 \
    --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 \
    --max_steps=4000 --gradient_accumulation_steps=8 --max_packed_inp_len=2048 \
    --max_packed_ctx_len=2048 --use_per_ctx_average_loss=True --use_kl_loss=True \
    --quantize_ctx_encoder=True \
    --save_steps=250 --save_total_limit=5

echo "=========================================="
echo "Done! Model pushed to https://huggingface.co/${HF_REPO}"
echo "=========================================="
