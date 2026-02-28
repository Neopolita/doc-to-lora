#!/bin/bash
set -euo pipefail

# =============================================================================
# Launch Ministral-3-3B hypernetwork training on HF Jobs
# Billed to organization: mistral-hackaton-2026
# =============================================================================
#
# Prerequisites:
#   1. Install hf CLI: curl -LsSf https://hf.co/cli/install.sh | bash
#   2. Login: hf auth login
#   3. Ensure WANDB_API_KEY is set in your environment
#   4. Ensure you have write access to org "mistral-hackaton-2026"
#
# Usage:
#   bash scripts/hf_cloud/launch_job.sh
#
# Monitor:
#   hf jobs ps
#   hf jobs logs <job_id>
#
# =============================================================================

# Stock PyTorch image — no Docker account needed.
# Dependencies are installed at job startup via install.sh
IMAGE="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
REPO_URL="https://github.com/Neopolita/doc-to-lora"

echo "Launching HF Jobs training..."
hf jobs run \
    --namespace neopolita \
    --flavor a100x4 \
    --timeout 72h \
    --secrets WANDB_API_KEY \
    --secrets HF_TOKEN \
    --env WANDB_PROJECT=doc-to-lora \
    --env HF_HUB_ENABLE_HF_TRANSFER=1 \
    --env HF_PUSH_REPO=neopolita/doc-to-lora-ministral-3b-2512 \
    "${IMAGE}" \
    bash -c "apt-get update && apt-get install -y git curl build-essential && curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=/root/.local/bin:\$PATH && git clone ${REPO_URL} /app && cd /app && bash scripts/hf_cloud/train_ministral3b.sh"

echo ""
echo "Job submitted! Monitor with:"
echo "  hf jobs ps"
echo "  hf jobs logs <job_id>"
