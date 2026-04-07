from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from src.keymat import KeyMatTransform
from src.aloepri.layers.base import ObfuscatedLayer
from src.stage_g_norm import KeyMatFusedRMSNorm

class AloePriRMSNorm(ObfuscatedLayer):
    """
    Standardized AloePri RMSNorm.
    Wraps KeyMatFusedRMSNorm from stage_g_norm.
    """
    def __init__(
        self,
        norm_layer: nn.Module,
        keymat_transform: KeyMatTransform,
        kappa: float,
        recorder: Optional[object] = None,
        record_name: Optional[str] = None,
    ) -> None:
        super().__init__(recorder=recorder, record_name=record_name)
        self.inner = KeyMatFusedRMSNorm(
            norm_layer=norm_layer,
            keymat_transform=keymat_transform,
            kappa=kappa,
            recorder=recorder,
            record_name=record_name,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # KeyMatFusedRMSNorm handles recording inside its forward
        return self.inner(hidden_states)

def wrap_norm(
    norm_layer: nn.Module,
    keymat_transform: KeyMatTransform,
    kappa: float,
    recorder: Optional[object] = None,
    record_name: Optional[str] = None,
) -> AloePriRMSNorm:
    return AloePriRMSNorm(
        norm_layer=norm_layer,
        keymat_transform=keymat_transform,
        kappa=kappa,
        recorder=recorder,
        record_name=record_name,
    )
