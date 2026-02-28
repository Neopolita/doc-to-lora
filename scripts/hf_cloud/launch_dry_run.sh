#!/bin/bash
set -euo pipefail

# =============================================================================
# Dry run on single A100 — verify setup before full training
# Estimated cost: ~$3-5 (a100-large ~$6.30/hr, should finish in <30 min)
# =============================================================================

IMAGE="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
REPO_URL="https://github.com/Neopolita/doc-to-lora"

echo "Launching dry run..."
hf jobs run \
    --namespace neopolita \
    --flavor a100x8 \
    --timeout 1h \
    --secrets WANDB_API_KEY \
    --secrets HF_TOKEN \
    --env WANDB_PROJECT=doc-to-lora \
    --env HF_HUB_ENABLE_HF_TRANSFER=1 \
    "${IMAGE}" \
    bash -c "apt-get update && apt-get install -y git curl build-essential && curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=/root/.local/bin:\$PATH && git clone ${REPO_URL} /app && cd /app && bash scripts/hf_cloud/dry_run.sh"

echo ""
echo "Dry run submitted! Monitor with:"
echo "  hf jobs ps"
echo "  hf jobs logs <job_id>"
