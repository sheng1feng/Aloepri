from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from src.keymat import KeyMatTransform
from src.attention_keys import AttentionComplexConfig
from src.aloepri.layers.base import ObfuscatedLayer
from src.stage_h_attention_static import StaticizedQwen2Attention, build_staticized_attention

class AloePriAttention(ObfuscatedLayer):
    """
    Standardized AloePri Attention (staticized/fused).
    """
    def __init__(
        self,
        attention_module: nn.Module,
        keymat_transform: KeyMatTransform,
        input_norm_weight: torch.Tensor,
        attention_profile: str,
        seed: int,
        qk_scale_range: tuple[float, float],
        beta: int,
        gamma: float,
        rope_base: float,
        layer_idx: int,
        recorder: Optional[object] = None,
        record_name: Optional[str] = None,
    ) -> None:
        super().__init__(recorder=recorder, record_name=record_name)
        
        # We wrap the existing StaticizedQwen2Attention for now, but standardize its creation
        self.attn = build_staticized_attention(
            attention_module=attention_module,
            keymat_transform=keymat_transform,
            input_norm_weight=input_norm_weight,
            recorder=recorder or object(), # Need a recorder to avoid error if not provided
            layer_idx=layer_idx,
            attention_profile=attention_profile,
            seed=seed,
            qk_scale_range=qk_scale_range,
            beta=beta,
            gamma=gamma,
            rope_base=rope_base,
        )

    def forward(self, *args, **kwargs):
        # StaticizedQwen2Attention returns (attn_output, attn_weights)
        # Note: past_key_value is handled internally if past_key_values is passed in kwargs
        outputs = self.attn(*args, **kwargs)
        if isinstance(outputs, tuple):
            output = outputs[0]
        else:
            output = outputs
        self.record(output)
        return outputs

def wrap_attention(
    attention_module: nn.Module,
    keymat_transform: KeyMatTransform,
    input_norm_weight: torch.Tensor,
    attention_profile: str,
    seed: int,
    qk_scale_range: tuple[float, float],
    beta: int,
    gamma: float,
    rope_base: float,
    layer_idx: int,
    recorder: Optional[object] = None,
    record_name: Optional[str] = None,
) -> AloePriAttention:
    return AloePriAttention(
        attention_module=attention_module,
        keymat_transform=keymat_transform,
        input_norm_weight=input_norm_weight,
        attention_profile=attention_profile,
        seed=seed,
        qk_scale_range=qk_scale_range,
        beta=beta,
        gamma=gamma,
        rope_base=rope_base,
        layer_idx=layer_idx,
        recorder=recorder,
        record_name=record_name,
    )
