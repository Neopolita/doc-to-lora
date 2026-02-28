"""Convert FP8 Ministral-3-3B multimodal model to bf16 text-only model.

vLLM 0.8.5's load_format="mistral" doesn't properly dequantize FP8 weights,
producing all-zero outputs (<unk> tokens). This script:
1. Loads via transformers Mistral3ForConditionalGeneration (handles FP8→bf16)
2. Strips the vision tower, keeping only the language model
3. Saves as standard bf16 safetensors that vLLM can load directly

Usage:
    python scripts/convert_fp8_to_bf16.py \
        --model mistralai/Ministral-3-3B-Instruct-2512 \
        --output models/ministral-3b-bf16
"""

import argparse
import os

import torch
from transformers import AutoConfig, Mistral3ForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model ID")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.output, "config.json")):
        print(f"Converted model already exists at {args.output}, skipping.")
        return

    print(f"Loading {args.model} (this downloads FP8 weights and converts to bf16)...")

    # Strip FP8 quantization config so transformers loads in bf16
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    if hasattr(getattr(config, "text_config", None), "quantization_config"):
        delattr(config.text_config, "quantization_config")

    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.model,
        config=config,
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

    # Verify config is MistralForCausalLM (not the multimodal wrapper)
    saved_config = AutoConfig.from_pretrained(args.output)
    print(f"Saved config model_type: {saved_config.model_type}")


if __name__ == "__main__":
    main()
