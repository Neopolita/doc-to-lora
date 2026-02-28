#!/bin/bash
set -euo pipefail

# =============================================================================
# Dry run — verify everything works before full training
# Single A100, ~10 training steps, compact datasets only
# =============================================================================

MODEL="mistralai/Ministral-3-3B-Instruct-2512"
CONFIG="configs/main_exp/ministral-3b/dry_run.yaml"
PORT=29051

echo "=========================================="
echo "Phase 0: Install dependencies"
echo "=========================================="

bash install.sh

# Upgrade mistral_common for Ministral-3-3B tokenizer v13 support
# Use pip directly to bypass uv's lockfile constraints (uv run re-syncs)
.venv/bin/pip install "mistral-common>=1.9.0" --force-reinstall --no-deps
.venv/bin/pip install pydantic-extra-types

# Patch vLLM 0.8.5's llama weight loader to skip fake_quantizer keys
# (Ministral-3-3B FP8 checkpoint has calibration weights vLLM doesn't know about)
.venv/bin/python << 'PATCH'
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
print('Patched vLLM llama.py to skip unknown weight keys')
PATCH

echo "=========================================="
echo "Phase 1: Generate minimal self-gen data"
echo "=========================================="

# Use .venv/bin directly (not uv run) to prevent uv from
# re-syncing the environment and reverting the mistral_common upgrade
.venv/bin/python data/self_generate_qa.py \
    --vllm_model "${MODEL}" \
    --ds_names squad_compact drop_compact \
    --split train --closed_qa_prob 1.0 --debug

echo "=========================================="
echo "Phase 2: Training (10 steps, 1 GPU)"
echo "=========================================="

.venv/bin/accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_port $PORT \
    --num_processes=1 --gpu_ids 0 \
    train.py \
    "${CONFIG}" \
    --model_name_or_path="${MODEL}" \
    --target_modules=down_proj --lora_r=8 \
    --eval_strategy=no --max_qas_len=2048 --max_qas_per_sample=1 \
    --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 \
    --max_steps=10 --gradient_accumulation_steps=1 --max_packed_inp_len=4096 \
    --max_packed_ctx_len=4096 --use_per_ctx_average_loss=True --use_kl_loss=True \
    --quantize_ctx_encoder=True

echo "=========================================="
echo "Dry run complete! Everything works."
echo "=========================================="
