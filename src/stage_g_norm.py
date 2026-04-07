from __future__ import annotations

import torch
from torch import nn

from src.keymat import KeyMatTransform


class KeyMatFusedRMSNorm(nn.Module):
    def __init__(
        self,
        norm_layer: nn.Module,
        keymat_transform: KeyMatTransform,
        kappa: float,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.variance_epsilon = float(norm_layer.variance_epsilon)
        self.kappa = float(kappa)
        self.recorder = recorder
        self.record_name = record_name

        q = keymat_transform.inverse.detach().to(torch.float32)
        norm_weight = norm_layer.weight.detach().to(torch.float32)
        restore = q * norm_weight.unsqueeze(0)
        metric_matrix = q @ q.T
        self.register_buffer("restore_matrix", restore, persistent=False)
        self.register_buffer("metric_matrix", metric_matrix, persistent=False)
        self.hidden_size = int(keymat_transform.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_fp32 = hidden_states.to(torch.float32)
        metric_matrix = self.metric_matrix.to(device=hidden_states.device, dtype=torch.float32)
        quadratic = torch.matmul(hidden_fp32, metric_matrix)
        variance = (quadratic * hidden_fp32).sum(dim=-1, keepdim=True) / float(self.hidden_size)
        output = hidden_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        if self.recorder is not None and self.record_name is not None:
            restored = torch.matmul(
                output,
                self.restore_matrix.to(device=hidden_states.device, dtype=torch.float32),
            )
            self.recorder.record(self.record_name, restored)
        return output.to(input_dtype)


def build_keymat_fused_rmsnorm(
    norm_layer: nn.Module,
    keymat_transform: KeyMatTransform,
    kappa: float,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatFusedRMSNorm:
    return KeyMatFusedRMSNorm(
        norm_layer=norm_layer,
        keymat_transform=keymat_transform,
        kappa=kappa,
        recorder=recorder,
        record_name=record_name,
    )
