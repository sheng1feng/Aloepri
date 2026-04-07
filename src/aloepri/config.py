from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

@dataclass(frozen=True)
class AloePriConfig:
    # Model parameters
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float = 10000.0
    architecture_family: str = "qwen_decoder"
    model_type: str = "qwen"
    
    # Obfuscation parameters
    expansion_size: int = 128
    lam: float = 0.3
    alpha_e: float = 1.0
    alpha_h: float = 0.2
    
    # Attention specific
    attention_profile: str = "rqk_hqk_block_taukv_taugroup"
    qk_scale_range: Tuple[float, float] = (0.95, 1.05)
    beta: int = 8
    gamma: float = 1e3
    
    # FFN specific
    ffn_scale_range: Tuple[float, float] = (0.9, 1.1)
    
    # Execution parameters
    seed: int = 20260323
    device: str = "cpu"
    dtype: str = "float32"
    
    # Adapted layers (list of layer indices)
    adapted_layers: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.adapted_layers:
            # Default to all layers if not specified
            object.__setattr__(self, 'adapted_layers', list(range(self.num_hidden_layers)))

    @classmethod
    def from_model(cls, model, **overrides) -> "AloePriConfig":
        from src.aloepri.adapters import build_architecture_config

        return build_architecture_config(model, **overrides)
