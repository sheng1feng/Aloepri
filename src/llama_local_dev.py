from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from src.model_loader import set_global_seed


def load_llama_local_tokenizer(model_dir: str | Path):
    return AutoTokenizer.from_pretrained(Path(model_dir), use_fast=True, trust_remote_code=True)


def load_llama_local_config(model_dir: str | Path):
    return AutoConfig.from_pretrained(Path(model_dir), trust_remote_code=True)


def build_mock_llama_from_local_metadata(
    model_dir: str | Path,
    *,
    seed: int = 20260323,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 2,
    max_position_embeddings: int = 2048,
) -> tuple[Any, Any]:
    model_dir = Path(model_dir)
    set_global_seed(seed)
    tokenizer = load_llama_local_tokenizer(model_dir)
    base_config = load_llama_local_config(model_dir)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rope_theta=float(getattr(base_config, "rope_theta", 500000.0) or 500000.0),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=bool(getattr(base_config, "tie_word_embeddings", True)),
    )
    model = LlamaForCausalLM(config).eval()
    return tokenizer, model


def mock_llama_summary(tokenizer: Any, model: Any) -> dict[str, int | str | bool]:
    return {
        "model_type": str(model.config.model_type),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_length": int(len(tokenizer)),
        "model_vocab_size": int(model.config.vocab_size),
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
    }


def tokenize_llama_prompt(tokenizer: Any, prompt: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    encoded = tokenizer(text, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}
