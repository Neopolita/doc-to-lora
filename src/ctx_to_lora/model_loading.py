import logging
import os

import torch
from peft import PeftModel
from peft import get_peft_config as _get_peft_config
from peft.utils import PeftType
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    MistralConfig,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# Register ministral3 text config for transformers 4.51.3 compatibility
# (Ministral-3-3B uses model_type="ministral3" which isn't in this version)
if "ministral3" not in CONFIG_MAPPING:
    CONFIG_MAPPING.register("ministral3", MistralConfig)

logger = logging.getLogger()

GEMMA_VISION_MODELS = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]

# Multimodal models that use Mistral3ForConditionalGeneration (Pixtral-like)
MISTRAL3_VISION_MODELS = [
    "mistralai/Ministral-3-3B-Instruct-2512",
]


def check_is_vision_model(model_name):
    return model_name in GEMMA_VISION_MODELS or model_name in MISTRAL3_VISION_MODELS


def get_model_and_tokenizer(
    model_name_or_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    tokenizer_kwargs=None,
    use_q_lora=False,
    device="cuda",
    dtype=torch.bfloat16,
):
    model = get_model(
        model_name_or_path,
        train,
        requires_grad,
        use_flash_attn,
        peft_config,
        model_kwargs,
        use_q_lora,
        device,
        dtype,
    )
    tokenizer = get_tokenizer(model_name_or_path, tokenizer_kwargs, peft_config, train)
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def get_tokenizer(
    model_name_or_path, tokenizer_kwargs=None, peft_config=None, train=False
):
    padding_side = "left" if not train else "right"
    truncation_side = "left"

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            add_bos_tokens=False,
            add_eos_tokens=False,
            padding_side=padding_side,
            truncation_side=truncation_side,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )
    except ValueError:
        # Fallback for models whose tokenizer class isn't in this transformers version
        # (e.g. Ministral-3-3B uses TokenizersBackend, unknown to transformers 4.51.3)
        from transformers import PreTrainedTokenizerFast
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        # Patch: transformers 4.51.3 expects extra_special_tokens as dict, but
        # newer model configs may provide a list
        _orig = PreTrainedTokenizerBase._set_model_specific_special_tokens
        def _patched(self, special_tokens=None):
            if isinstance(special_tokens, list):
                special_tokens = {}
            return _orig(self, special_tokens=special_tokens)
        PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_name_or_path,
            add_bos_tokens=False,
            add_eos_tokens=False,
            padding_side=padding_side,
            truncation_side=truncation_side,
            trust_remote_code=True,
            **tokenizer_kwargs,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    template_path = f"chat_templates/{model_name_or_path}.jinja"
    if not os.path.exists(template_path):
        logger.warning(
            f"Chat template not found at {template_path}. Using default template."
        )
        return tokenizer

    logger.info(f"Using chat template from {template_path}")
    chat_template = open(template_path).read()
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    tokenizer.chat_template = chat_template
    return tokenizer


def get_model(
    model_name_or_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    use_q_lora=False,
    device="cuda",
    dtype=torch.bfloat16,
):
    model_init_kwargs = dict(
        pretrained_model_name_or_path=model_name_or_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=None,
    )
    is_vision_model = check_is_vision_model(model_name_or_path)
    if model_kwargs is not None:
        model_init_kwargs.update(model_kwargs)

    is_bidir_model = (
        "bert" in model_name_or_path.lower() or "gte" in model_name_or_path.lower()
    )

    if use_flash_attn:
        if "gte" not in model_name_or_path:
            model_init_kwargs["attn_implementation"] = "flash_attention_2"
        elif "gte" in model_name_or_path:
            model_init_kwargs["attn_implementation"] = "sdpa"

    if is_vision_model:
        # always use sdpa for vision models
        # model_init_kwargs["attn_implementation"] = "sdpa"
        model_init_kwargs.pop("use_cache")
    elif is_bidir_model:
        model_init_kwargs["torch_dtype"] = torch.float32
        model_init_kwargs.pop("use_cache")

    if use_q_lora:
        # https://huggingface.co/blog/4bit-transformers-bitsandbytes
        # https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing
        # see bitsandbytes for the quantization implementation https://github.com/bitsandbytes-foundation/bitsandbytes
        # see unsloth https://huggingface.co/docs/trl/v0.7.11/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth
        # does work currently bc it modifies the forward pass call of Linear
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_init_kwargs["quantization_config"] = bnb_config

    logger.debug(f"Model init kwargs: {model_init_kwargs}")
    if not is_vision_model:
        if is_bidir_model:
            model = AutoModel.from_pretrained(**model_init_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(**model_init_kwargs)
    elif model_name_or_path in MISTRAL3_VISION_MODELS:
        from transformers import AutoConfig, Mistral3ForConditionalGeneration
        # Strip FP8 quantization config — we want bf16 for training, and
        # transformers 4.51.3 doesn't support activation_scheme="static"
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        if hasattr(getattr(config, "text_config", None), "quantization_config"):
            delattr(config.text_config, "quantization_config")
        model_init_kwargs["config"] = config
        # PixtralVisionModel doesn't support flash_attention_2 — use eager
        # (we only need the language_model anyway, vision tower is discarded)
        model_init_kwargs["attn_implementation"] = "eager"
        model = Mistral3ForConditionalGeneration.from_pretrained(**model_init_kwargs)
        model = model.language_model
        # Restore name_or_path (lost when extracting sub-model from multimodal wrapper)
        model.config.name_or_path = model_name_or_path
    else:
        model = Gemma3ForConditionalGeneration.from_pretrained(**model_init_kwargs)
        model = model.language_model
    if peft_config is not None:
        model = PeftModel(model, peft_config)
    model.train(train)
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad
    return model


def get_lora_config(model_dir, **kwargs):
    if "target_modules" not in kwargs or kwargs["target_modules"] is None:
        logger.info("No target modules specified for LoRA.")
        return None
    r = kwargs.pop("lora_r", 8)
    peft_conf_kwargs = dict(
        r=r,
        peft_type=PeftType.LORA,
        base_model_name_or_path=model_dir,
        task_type="CAUSAL_LM",
        lora_dropout=kwargs.get("lora_dropout", 0.0),
        lora_alpha=r ** (3 / 2) * 2,
    )

    peft_conf_kwargs.update(kwargs)
    peft_config = _get_peft_config(peft_conf_kwargs)
    return peft_config
