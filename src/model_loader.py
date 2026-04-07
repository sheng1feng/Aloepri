from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.defaults import DEFAULT_SYSTEM_PROMPT


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype: str) -> torch.dtype | str:
    normalized = dtype.lower()
    mapping: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[normalized]


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cpu",
    dtype: str = "auto",
) -> tuple[Any, Any]:
    torch_dtype = resolve_torch_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    if device != "cpu":
        model.to(device)
    return tokenizer, model


def format_chat_prompt(
    tokenizer: Any,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    text = format_chat_prompt(tokenizer, prompt, system_prompt=system_prompt)
    encoded = tokenizer(text, return_tensors="pt")
    return move_batch_to_device(encoded, device)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}

