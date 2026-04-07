from __future__ import annotations

import torch
from torch import nn

from src.keymat import KeyMatTransform, apply_inverse_keymat_transform, apply_keymat_transform


def estimate_kappa_for_keymat(
    keymat_transform: KeyMatTransform,
    hidden_size: int | None = None,
    num_samples: int = 1000,
    seed: int = 0,
) -> float:
    hidden_size = hidden_size or keymat_transform.hidden_size
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    samples = torch.randn((num_samples, hidden_size), generator=generator, dtype=torch.float32)
    transformed = apply_keymat_transform(samples, keymat_transform)
    ratio = torch.linalg.vector_norm(transformed, dim=-1) / torch.linalg.vector_norm(samples, dim=-1)
    return float(ratio.mean().item())


class KeyMatRMSNormBridge(nn.Module):
    def __init__(
        self,
        norm_layer: nn.Module,
        keymat_transform: KeyMatTransform,
        kappa: float,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.norm_layer = norm_layer
        self.keymat_transform = keymat_transform
        self.kappa = float(kappa)
        self.recorder = recorder
        self.record_name = record_name

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_hidden = apply_inverse_keymat_transform(hidden_states, self.keymat_transform)
        normalized = self.norm_layer(base_hidden)
        if self.recorder is not None and self.record_name is not None:
            self.recorder.record(self.record_name, normalized)
        return apply_keymat_transform(normalized, self.keymat_transform)


def build_keymat_rmsnorm_wrapper(
    norm_layer: nn.Module,
    keymat_transform: KeyMatTransform,
    kappa: float,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatRMSNormBridge:
    return KeyMatRMSNormBridge(
        norm_layer=norm_layer,
        keymat_transform=keymat_transform,
        kappa=kappa,
        recorder=recorder,
        record_name=record_name,
    )


def fuse_keymat_norm_into_adjacent_linear(
    linear_weight: torch.Tensor,
    left_inverse: torch.Tensor,
    right_key: torch.Tensor,
) -> torch.Tensor:
    left_inverse = left_inverse.to(device=linear_weight.device, dtype=linear_weight.dtype)
    right_key = right_key.to(device=linear_weight.device, dtype=linear_weight.dtype)
    return left_inverse @ linear_weight @ right_key
