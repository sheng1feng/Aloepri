from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.hidden_keys import HiddenTransform
from src.transforms import apply_hidden_transform, apply_inverse_hidden_transform


@dataclass(frozen=True)
class FFNTransform:
    perm: torch.Tensor
    inv_perm: torch.Tensor
    scale: torch.Tensor
    inv_scale: torch.Tensor

    @property
    def dim(self) -> int:
        return int(self.perm.numel())


def generate_ffn_permutation(intermediate_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(intermediate_size, generator=generator, dtype=torch.long)


def generate_ffn_scaling(
    intermediate_size: int,
    scale_range: tuple[float, float],
    seed: int,
) -> torch.Tensor:
    low, high = scale_range
    if low <= 0 or high <= 0 or low > high:
        raise ValueError(f"Invalid FFN scale range: {scale_range}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.empty(intermediate_size, dtype=torch.float32).uniform_(low, high, generator=generator)


def build_ffn_transform(perm: torch.Tensor, scale: torch.Tensor) -> FFNTransform:
    perm = torch.as_tensor(perm, dtype=torch.long).clone()
    scale = torch.as_tensor(scale, dtype=torch.float32).clone()
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), dtype=torch.long)
    inv_scale = 1.0 / scale
    return FFNTransform(
        perm=perm,
        inv_perm=inv_perm,
        scale=scale,
        inv_scale=inv_scale,
    )


def apply_ffn_permutation(hidden: torch.Tensor, ffn_transform: FFNTransform) -> torch.Tensor:
    perm = ffn_transform.perm.to(hidden.device)
    transformed = torch.empty_like(hidden)
    transformed[..., perm] = hidden
    return transformed


def apply_ffn_up_transform(hidden: torch.Tensor, ffn_transform: FFNTransform) -> torch.Tensor:
    permuted = apply_ffn_permutation(hidden, ffn_transform)
    scale = ffn_transform.scale.to(device=hidden.device, dtype=hidden.dtype)
    return permuted * scale


def invert_ffn_product_transform(hidden: torch.Tensor, ffn_transform: FFNTransform) -> torch.Tensor:
    inv_scale = ffn_transform.inv_scale.to(device=hidden.device, dtype=hidden.dtype)
    scaled = hidden * inv_scale
    perm = ffn_transform.perm.to(hidden.device)
    return scaled.index_select(dim=-1, index=perm)


class ObfuscatedQwen2MLP(nn.Module):
    def __init__(
        self,
        mlp_module: nn.Module,
        hidden_transform: HiddenTransform,
        ffn_transform: FFNTransform,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = mlp_module.gate_proj
        self.up_proj = mlp_module.up_proj
        self.down_proj = mlp_module.down_proj
        self.act_fn = mlp_module.act_fn
        self.hidden_transform = hidden_transform
        self.ffn_transform = ffn_transform
        self.recorder = recorder
        self.record_name = record_name

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_hidden = apply_inverse_hidden_transform(hidden_states, self.hidden_transform)
        gate_base = self.gate_proj(base_hidden)
        up_base = self.up_proj(base_hidden)

        gate_mid = apply_ffn_permutation(gate_base, self.ffn_transform)
        up_mid = apply_ffn_up_transform(up_base, self.ffn_transform)
        product_mid = self.act_fn(gate_mid) * up_mid
        product_base = invert_ffn_product_transform(product_mid, self.ffn_transform)
        down_base = self.down_proj(product_base)
        output = apply_hidden_transform(down_base, self.hidden_transform)
        if self.recorder is not None and self.record_name is not None:
            self.recorder.record(self.record_name, output)
        return output


def obfuscate_ffn_block(
    mlp_module: nn.Module,
    hidden_transform: HiddenTransform,
    ffn_transform: FFNTransform,
    recorder=None,
    record_name: str | None = None,
) -> ObfuscatedQwen2MLP:
    return ObfuscatedQwen2MLP(
        mlp_module=mlp_module,
        hidden_transform=hidden_transform,
        ffn_transform=ffn_transform,
        recorder=recorder,
        record_name=record_name,
    )
