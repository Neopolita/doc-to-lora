"""Extract text-only language model from a multimodal Mistral3 checkpoint.

Ministral-3-3B is packaged as Mistral3ForConditionalGeneration (multimodal).
vLLM 0.8.5 can't load multimodal models without a vision preprocessor.
This script extracts just the MistralForCausalLM language model and saves it
as standard bf16 safetensors that vLLM can load directly.

Usage:
    python scripts/extract_language_model.py \
        --model mistralai/Ministral-3-3B-Instruct-2512 \
        --output models/ministral-3b-bf16
"""

import argparse
import os

import torch
from transformers import MistralConfig, Mistral3ForConditionalGeneration
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# Register ministral3 text config for transformers 4.51.3 compatibility
# (Ministral-3-3B uses model_type="ministral3" which isn't in this version)
if "ministral3" not in CONFIG_MAPPING:
    CONFIG_MAPPING.register("ministral3", MistralConfig)

# Use BF16 variants to avoid FP8 dequantization issues in transformers 4.51.3
BF16_VARIANTS = {
    "mistralai/Ministral-3-3B-Instruct-2512": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model ID")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.output, "config.json")):
        print(f"Extracted model already exists at {args.output}, skipping.")
        return

    download_name = BF16_VARIANTS.get(args.model, args.model)
    print(f"Loading {download_name} ...")

    model = Mistral3ForConditionalGeneration.from_pretrained(
        download_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # Extract just the language model (MistralForCausalLM)
    lang_model = model.language_model
    print(f"Extracted language model: {type(lang_model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in lang_model.parameters()) / 1e9:.2f}B")

    # Save as standard HF model
    os.makedirs(args.output, exist_ok=True)
    lang_model.save_pretrained(args.output, safe_serialization=True)
    print(f"Saved bf16 model to {args.output}")


if __name__ == "__main__":
    main()
