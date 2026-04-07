from __future__ import annotations

from typing import Any

from src.aloepri.adapters.llama import LlamaArchitectureAdapter, build_llama_config, is_llama_compatible_model
from src.aloepri.adapters.qwen import QwenArchitectureAdapter, build_qwen_config, is_qwen_compatible_model


def get_architecture_adapter(model: Any):
    model_type = str(getattr(getattr(model, "config", object()), "model_type", "")).lower()
    if model_type == "llama":
        return LlamaArchitectureAdapter.from_model(model)
    if "qwen" in model_type:
        return QwenArchitectureAdapter.from_model(model)

    qwen_error = None
    try:
        return QwenArchitectureAdapter.from_model(model)
    except Exception as exc:
        qwen_error = exc

    try:
        return LlamaArchitectureAdapter.from_model(model)
    except Exception as exc:
        raise ValueError(
            f"Unsupported model architecture; qwen check failed with {qwen_error!r}, llama check failed with {exc!r}"
        ) from exc


def build_architecture_config(model: Any, **overrides):
    adapter = get_architecture_adapter(model)
    if isinstance(adapter, QwenArchitectureAdapter):
        return build_qwen_config(model, **overrides)
    if isinstance(adapter, LlamaArchitectureAdapter):
        return build_llama_config(model, **overrides)
    raise TypeError(f"Unsupported adapter type: {type(adapter)!r}")

__all__ = [
    "LlamaArchitectureAdapter",
    "build_llama_config",
    "is_llama_compatible_model",
    "QwenArchitectureAdapter",
    "build_qwen_config",
    "is_qwen_compatible_model",
    "get_architecture_adapter",
    "build_architecture_config",
]
