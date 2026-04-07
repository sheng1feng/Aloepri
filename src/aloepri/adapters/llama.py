from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_attr(obj: Any, path: str):
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            raise ValueError(f"Missing required Llama-architecture attribute: {path}")
        current = getattr(current, part)
    return current


@dataclass(frozen=True)
class LlamaArchitectureAdapter:
    """
    Structural adapter for Llama-style decoder-only causal LM models.

    Unlike the Qwen adapter, this adapter intentionally validates that the
    checkpoint identifies itself as the Llama family, so that similarly-shaped
    decoder architectures do not get misclassified.
    """

    model: Any

    @classmethod
    def from_model(cls, model: Any) -> "LlamaArchitectureAdapter":
        config = _require_attr(model, "config")
        model_type = str(getattr(config, "model_type", "")).lower()
        if model_type != "llama":
            raise ValueError(f"Expected model_type='llama', got {model_type!r}")

        _require_attr(model, "config.hidden_size")
        _require_attr(model, "config.num_hidden_layers")
        _require_attr(model, "config.num_attention_heads")
        _require_attr(model, "model.embed_tokens")
        _require_attr(model, "model.layers")
        _require_attr(model, "model.norm")
        _require_attr(model, "lm_head")
        layers = _require_attr(model, "model.layers")
        if len(layers) == 0:
            raise ValueError("Llama-architecture model must have at least one decoder layer.")
        layer0 = layers[0]
        _require_attr(layer0, "self_attn.q_proj")
        _require_attr(layer0, "self_attn.k_proj")
        _require_attr(layer0, "self_attn.v_proj")
        _require_attr(layer0, "self_attn.o_proj")
        _require_attr(layer0, "input_layernorm")
        _require_attr(layer0, "post_attention_layernorm")
        _require_attr(layer0, "mlp.gate_proj")
        _require_attr(layer0, "mlp.up_proj")
        _require_attr(layer0, "mlp.down_proj")
        return cls(model=model)

    @property
    def config(self):
        return self.model.config

    @property
    def layers(self):
        return self.model.model.layers

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.config.num_hidden_layers)

    @property
    def num_attention_heads(self) -> int:
        return int(self.config.num_attention_heads)

    @property
    def num_key_value_heads(self) -> int:
        return int(getattr(self.config, "num_key_value_heads", self.config.num_attention_heads))

    @property
    def head_dim(self) -> int:
        explicit = getattr(self.config, "head_dim", None)
        if explicit is not None:
            return int(explicit)
        return self.hidden_size // self.num_attention_heads

    @property
    def rope_theta(self) -> float:
        rope_theta = getattr(self.config, "rope_theta", None)
        if rope_theta is None:
            return 10000.0
        return float(rope_theta)

    @property
    def model_type(self) -> str:
        return str(getattr(self.config, "model_type", "llama"))

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
            "is_tied_word_embeddings": bool(
                self.model.get_input_embeddings().weight.data_ptr()
                == self.model.get_output_embeddings().weight.data_ptr()
            ),
        }


def is_llama_compatible_model(model: Any) -> bool:
    try:
        LlamaArchitectureAdapter.from_model(model)
        return True
    except Exception:
        return False


def build_llama_config(
    model: Any,
    *,
    expansion_size: int = 128,
    lam: float = 0.3,
    alpha_e: float = 1.0,
    alpha_h: float = 0.2,
    attention_profile: str = "rqk_hqk_block_taukv_taugroup",
    qk_scale_range: tuple[float, float] = (0.95, 1.05),
    beta: int = 8,
    gamma: float = 1e3,
    ffn_scale_range: tuple[float, float] = (0.9, 1.1),
    seed: int = 20260323,
    device: str = "cpu",
    dtype: str = "float32",
    adapted_layers: list[int] | None = None,
):
    from src.aloepri.config import AloePriConfig

    adapter = LlamaArchitectureAdapter.from_model(model)
    return AloePriConfig(
        hidden_size=adapter.hidden_size,
        num_hidden_layers=adapter.num_hidden_layers,
        num_attention_heads=adapter.num_attention_heads,
        num_key_value_heads=adapter.num_key_value_heads,
        head_dim=adapter.head_dim,
        rope_theta=adapter.rope_theta,
        expansion_size=expansion_size,
        lam=lam,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
        attention_profile=attention_profile,
        qk_scale_range=qk_scale_range,
        beta=beta,
        gamma=gamma,
        ffn_scale_range=ffn_scale_range,
        seed=seed,
        device=device,
        dtype=dtype,
        adapted_layers=list(range(adapter.num_hidden_layers)) if adapted_layers is None else adapted_layers,
        architecture_family="llama_decoder",
        model_type=adapter.model_type,
    )
