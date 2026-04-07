from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from src.keymat import KeyMatTransform
from src.obfuscate_ffn import FFNTransform
from src.aloepri.layers.base import ObfuscatedLayer
from src.stage_g_ffn import KeyMatFusedQwen2MLP

class AloePriMLP(ObfuscatedLayer):
    """
    Standardized AloePri MLP for Qwen-style models.
    Wraps KeyMatFusedQwen2MLP from stage_g_ffn.
    """
    def __init__(
        self,
        mlp_module: nn.Module,
        keymat_transform: KeyMatTransform,
        ffn_transform: FFNTransform,
        input_norm_weight: torch.Tensor,
        recorder: Optional[object] = None,
        record_name: Optional[str] = None,
    ) -> None:
        super().__init__(recorder=recorder, record_name=record_name)
        self.inner = KeyMatFusedQwen2MLP(
            mlp_module=mlp_module,
            keymat_transform=keymat_transform,
            input_norm_weight=input_norm_weight,
            ffn_transform=ffn_transform,
            recorder=recorder,
            record_name=record_name,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Note: KeyMatFusedQwen2MLP handles recording inside its forward
        return self.inner(hidden_states)

def wrap_mlp(
    mlp_module: nn.Module,
    keymat_transform: KeyMatTransform,
    ffn_transform: FFNTransform,
    input_norm_weight: torch.Tensor,
    recorder: Optional[object] = None,
    record_name: Optional[str] = None,
) -> AloePriMLP:
    return AloePriMLP(
        mlp_module=mlp_module,
        keymat_transform=keymat_transform,
        ffn_transform=ffn_transform,
        input_norm_weight=input_norm_weight,
        recorder=recorder,
        record_name=record_name,
    )
