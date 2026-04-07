from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.hidden_keys import build_identity_hidden_transform
from src.keymat import KeyMatTransform, apply_inverse_keymat_transform
from src.obfuscate_ffn import (
    FFNTransform,
    apply_ffn_permutation,
    apply_ffn_up_transform,
    invert_ffn_product_transform,
    obfuscate_ffn_block,
)


class KeyMatFFNBridgeNormFused(nn.Module):
    def __init__(
        self,
        mlp_module: nn.Module,
        keymat_transform: KeyMatTransform,
        input_norm_weight: torch.Tensor,
        ffn_transform: FFNTransform,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.keymat_transform = keymat_transform
        self.register_buffer("input_norm_weight", input_norm_weight.detach().to(torch.float32), persistent=False)
        self.register_buffer("key_matrix", keymat_transform.key.detach().to(torch.float32), persistent=False)
        self.inner = obfuscate_ffn_block(
            mlp_module=mlp_module,
            hidden_transform=build_identity_hidden_transform(keymat_transform.hidden_size),
            ffn_transform=ffn_transform,
            recorder=recorder,
            record_name=record_name,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_hidden = apply_inverse_keymat_transform(hidden_states, self.keymat_transform)
        base_hidden = base_hidden * self.input_norm_weight.to(device=hidden_states.device, dtype=base_hidden.dtype)
        mlp_base = self.inner(base_hidden)
        return torch.matmul(mlp_base, self.key_matrix.to(device=hidden_states.device, dtype=mlp_base.dtype))


class KeyMatFusedQwen2MLP(nn.Module):
    def __init__(
        self,
        mlp_module: nn.Module,
        keymat_transform: KeyMatTransform,
        input_norm_weight: torch.Tensor,
        ffn_transform: FFNTransform,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.act_fn = mlp_module.act_fn
        self.ffn_transform = ffn_transform
        self.recorder = recorder
        self.record_name = record_name

        q = keymat_transform.inverse.detach().to(torch.float32)
        p = keymat_transform.key.detach().to(torch.float32)
        norm_weight = input_norm_weight.detach().to(torch.float32)
        right_bridge = q * norm_weight.unsqueeze(0)

        gate_weight = mlp_module.gate_proj.weight.detach().to(torch.float32) @ right_bridge.T
        up_weight = mlp_module.up_proj.weight.detach().to(torch.float32) @ right_bridge.T
        down_weight = p.T @ mlp_module.down_proj.weight.detach().to(torch.float32)

        self.register_buffer("gate_weight", gate_weight, persistent=False)
        self.register_buffer("up_weight", up_weight, persistent=False)
        self.register_buffer("down_weight", down_weight, persistent=False)
        self.register_buffer("restore_matrix", q, persistent=False)

        if mlp_module.gate_proj.bias is not None:
            self.register_buffer("gate_bias", mlp_module.gate_proj.bias.detach().to(torch.float32), persistent=False)
        else:
            self.gate_bias = None
        if mlp_module.up_proj.bias is not None:
            self.register_buffer("up_bias", mlp_module.up_proj.bias.detach().to(torch.float32), persistent=False)
        else:
            self.up_bias = None
        if mlp_module.down_proj.bias is not None:
            down_bias = torch.matmul(
                mlp_module.down_proj.bias.detach().to(torch.float32),
                p,
            )
            self.register_buffer("down_bias", down_bias, persistent=False)
        else:
            self.down_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_linear = F.linear(
            hidden_states,
            self.gate_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.gate_bias is None else self.gate_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        up_linear = F.linear(
            hidden_states,
            self.up_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.up_bias is None else self.up_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )

        gate_hidden = apply_ffn_permutation(gate_linear, self.ffn_transform)
        up_hidden = apply_ffn_up_transform(up_linear, self.ffn_transform)
        product = self.act_fn(gate_hidden) * up_hidden
        down_input = invert_ffn_product_transform(product, self.ffn_transform)
        output = F.linear(
            down_input,
            self.down_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.down_bias is None else self.down_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        if self.recorder is not None and self.record_name is not None:
            restored = torch.matmul(
                output.to(torch.float32),
                self.restore_matrix.to(device=hidden_states.device, dtype=torch.float32),
            )
            self.recorder.record(self.record_name, restored)
        return output


def build_keymat_ffn_bridge_norm_fused(
    mlp_module: nn.Module,
    keymat_transform: KeyMatTransform,
    input_norm_weight: torch.Tensor,
    ffn_transform: FFNTransform,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatFFNBridgeNormFused:
    return KeyMatFFNBridgeNormFused(
        mlp_module=mlp_module,
        keymat_transform=keymat_transform,
        input_norm_weight=input_norm_weight,
        ffn_transform=ffn_transform,
        recorder=recorder,
        record_name=record_name,
    )


def build_keymat_fused_ffn(
    mlp_module: nn.Module,
    keymat_transform: KeyMatTransform,
    input_norm_weight: torch.Tensor,
    ffn_transform: FFNTransform,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatFusedQwen2MLP:
    return KeyMatFusedQwen2MLP(
        mlp_module=mlp_module,
        keymat_transform=keymat_transform,
        input_norm_weight=input_norm_weight,
        ffn_transform=ffn_transform,
        recorder=recorder,
        record_name=record_name,
    )
