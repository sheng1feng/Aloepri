from __future__ import annotations

import torch
from torch import nn

from src.hidden_keys import HiddenTransform


def permute_feature_weight(weight: torch.Tensor, hidden_transform: HiddenTransform) -> torch.Tensor:
    permuted = weight.clone()
    perm = hidden_transform.perm.to(weight.device)
    permuted[perm] = weight
    return permuted


def estimate_kappa(
    hidden_transform: HiddenTransform,
    hidden_size: int,
    num_samples: int = 1000,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    samples = torch.randn((num_samples, hidden_size), generator=generator, dtype=torch.float32)

    perm = hidden_transform.perm
    scale = hidden_transform.scale
    transformed = torch.empty_like(samples)
    transformed[:, perm] = samples
    transformed = transformed * scale

    ratio = torch.linalg.vector_norm(transformed, dim=-1) / torch.linalg.vector_norm(samples, dim=-1)
    return float(ratio.mean().item())


class ObfuscatedRMSNorm(nn.Module):
    def __init__(
        self,
        norm_layer: nn.Module,
        hidden_transform: HiddenTransform,
        kappa: float,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_transform = hidden_transform
        self.variance_epsilon = norm_layer.variance_epsilon
        self.register_buffer(
            "permuted_weight",
            permute_feature_weight(norm_layer.weight.detach().to(torch.float32), hidden_transform),
            persistent=False,
        )
        self.kappa = float(kappa)
        self.recorder = recorder
        self.record_name = record_name

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        normalized = hidden_states_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        output = self.permuted_weight.to(device=hidden_states.device, dtype=torch.float32) * normalized
        output = output * self.kappa
        output = output.to(input_dtype)
        if self.recorder is not None and self.record_name is not None:
            self.recorder.record(self.record_name, output)
        return output


def apply_rmsnorm_obfuscation(
    norm_layer: nn.Module,
    hidden_transform: HiddenTransform,
    kappa: float,
    recorder=None,
    record_name: str | None = None,
) -> ObfuscatedRMSNorm:
    return ObfuscatedRMSNorm(
        norm_layer=norm_layer,
        hidden_transform=hidden_transform,
        kappa=kappa,
        recorder=recorder,
        record_name=record_name,
    )
