from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward

from src.attention_keys import AttentionComplexConfig, build_attention_complex_config
from src.gqa_layout import GQALayout
from src.hidden_keys import build_identity_hidden_transform
from src.keymat import KeyMatTransform, apply_inverse_keymat_transform
from src.obfuscate_attention_complex import ComplexAttentionTraceData, ComplexQwen2Attention
from src.stage_b import TraceRecorder, TracingQwen2Attention


def _apply_last_dim_matrix(tensor: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.to(device=tensor.device, dtype=tensor.dtype)
    return torch.matmul(tensor, matrix)


class KeyMatAttentionBridgeNormFused(nn.Module):
    def __init__(
        self,
        attention_module: nn.Module,
        keymat_transform: KeyMatTransform,
        input_norm_weight: torch.Tensor,
        recorder: TraceRecorder,
        layer_idx: int,
        attention_profile: str = "rqk_hqk_block_taukv_taugroup",
        seed: int = 0,
        qk_scale_range: tuple[float, float] = (0.95, 1.05),
        beta: int = 4,
        gamma: float = 1e3,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.keymat_transform = keymat_transform
        self.register_buffer("input_norm_weight", input_norm_weight.detach().to(torch.float32), persistent=False)
        self.register_buffer("key_matrix", keymat_transform.key.detach().to(torch.float32), persistent=False)
        if attention_profile == "simplified":
            self.inner = TracingQwen2Attention(
                attention_module=attention_module,
                recorder=recorder,
                mode="plain",
                hidden_transform=None,
            )
        else:
            self.inner = ComplexQwen2Attention(
                attention_module=attention_module,
                recorder=recorder,
                layer_idx=layer_idx,
                hidden_transform=build_identity_hidden_transform(keymat_transform.hidden_size),
                attention_config=build_attention_complex_config(
                    profile=attention_profile,
                    head_dim=attention_module.head_dim,
                    num_kv_heads=attention_module.config.num_key_value_heads,
                    num_groups=attention_module.config.num_attention_heads // attention_module.config.num_key_value_heads,
                    seed=seed,
                    qk_scale_range=qk_scale_range,
                    beta=beta,
                    gamma=gamma,
                    rope_base=rope_base,
                ),
            )

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        base_hidden = apply_inverse_keymat_transform(hidden_states, self.keymat_transform)
        base_hidden = base_hidden * self.input_norm_weight.to(device=hidden_states.device, dtype=base_hidden.dtype)
        attn_output, attn_weights = self.inner(
            base_hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        return torch.matmul(attn_output, self.key_matrix.to(device=hidden_states.device, dtype=attn_output.dtype)), attn_weights


class KeyMatFusedQwen2Attention(nn.Module):
    def __init__(
        self,
        attention_module: nn.Module,
        keymat_transform: KeyMatTransform,
        input_norm_weight: torch.Tensor,
        recorder: TraceRecorder,
        layer_idx: int,
        attention_profile: str = "rqk_hqk_block_taukv_taugroup",
        seed: int = 0,
        qk_scale_range: tuple[float, float] = (0.95, 1.05),
        beta: int = 4,
        gamma: float = 1e3,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.recorder = recorder
        self.layer_idx = layer_idx
        self.prefix = f"layer_{layer_idx}"
        self.keymat_transform = keymat_transform

        self.config = attention_module.config
        self.layer_idx_attr = attention_module.layer_idx
        self.head_dim = attention_module.head_dim
        self.scaling = attention_module.scaling
        self.sliding_window = attention_module.sliding_window
        self.attention_dropout = attention_module.attention_dropout
        self.is_causal = getattr(attention_module, "is_causal", True)
        if hasattr(attention_module, "num_key_value_groups"):
            self.num_key_value_groups = attention_module.num_key_value_groups

        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.layout = GQALayout(self.num_heads, self.num_kv_heads)
        self.attention_config: AttentionComplexConfig = build_attention_complex_config(
            profile=attention_profile,
            head_dim=attention_module.head_dim,
            num_kv_heads=self.num_kv_heads,
            num_groups=self.config.num_attention_heads // self.num_kv_heads,
            seed=seed,
            qk_scale_range=qk_scale_range,
            beta=beta,
            gamma=gamma,
            rope_base=rope_base,
        )

        q = keymat_transform.inverse.detach().to(torch.float32)
        p = keymat_transform.key.detach().to(torch.float32)
        norm_weight = input_norm_weight.detach().to(torch.float32)
        right_bridge = q * norm_weight.unsqueeze(0)

        q_weight = attention_module.q_proj.weight.detach().to(torch.float32) @ right_bridge.T
        k_weight = attention_module.k_proj.weight.detach().to(torch.float32) @ right_bridge.T
        v_weight = attention_module.v_proj.weight.detach().to(torch.float32) @ right_bridge.T
        o_weight = p.T @ attention_module.o_proj.weight.detach().to(torch.float32)

        self.register_buffer("q_weight", q_weight, persistent=False)
        self.register_buffer("k_weight", k_weight, persistent=False)
        self.register_buffer("v_weight", v_weight, persistent=False)
        self.register_buffer("o_weight", o_weight, persistent=False)
        self.register_buffer("restore_matrix", q, persistent=False)

        if attention_module.q_proj.bias is not None:
            self.register_buffer("q_bias", attention_module.q_proj.bias.detach().to(torch.float32), persistent=False)
        else:
            self.q_bias = None
        if attention_module.k_proj.bias is not None:
            self.register_buffer("k_bias", attention_module.k_proj.bias.detach().to(torch.float32), persistent=False)
        else:
            self.k_bias = None
        if attention_module.v_proj.bias is not None:
            self.register_buffer("v_bias", attention_module.v_proj.bias.detach().to(torch.float32), persistent=False)
        else:
            self.v_bias = None
        if attention_module.o_proj.bias is not None:
            o_bias = torch.matmul(attention_module.o_proj.bias.detach().to(torch.float32), p)
            self.register_buffer("o_bias", o_bias, persistent=False)
        else:
            self.o_bias = None

    def _apply_intra_head(self, q_states: torch.Tensor, k_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_states = _apply_last_dim_matrix(q_states, self.attention_config.intra_head.q_matrix)
        k_states = _apply_last_dim_matrix(k_states, self.attention_config.intra_head.k_matrix)
        return q_states, k_states

    def _apply_inter_head(
        self,
        q_states: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inter = self.attention_config.inter_head
        self.recorder.record(f"{self.prefix}_q_heads_pre_inter_raw", q_states)
        self.recorder.record(f"{self.prefix}_k_heads_pre_inter_raw", k_states)
        self.recorder.record(f"{self.prefix}_v_heads_pre_inter_raw", v_states)
        q_grouped = self.layout.reshape_query_groups(q_states)
        q_grouped = self.layout.permute_query_groups(q_grouped, inter.tau_kv, inter.tau_group)
        q_states = self.layout.merge_query_groups(q_grouped)
        if inter.tau_kv is not None:
            k_states = self.layout.permute_kv_heads(k_states, inter.tau_kv)
            v_states = self.layout.permute_kv_heads(v_states, inter.tau_kv)
        self.recorder.record(f"{self.prefix}_q_heads_post_inter_raw", q_states)
        self.recorder.record(f"{self.prefix}_k_heads_post_inter_raw", k_states)
        self.recorder.record(f"{self.prefix}_v_heads_post_inter_raw", v_states)
        return q_states, k_states, v_states

    def _invert_inter_head(self, attn_output: torch.Tensor) -> torch.Tensor:
        inter = self.attention_config.inter_head
        transposed = False
        if attn_output.shape[1] != self.num_heads and attn_output.shape[2] == self.num_heads:
            attn_output = attn_output.transpose(1, 2)
            transposed = True
        attn_grouped = self.layout.reshape_query_groups(attn_output)
        attn_grouped = self.layout.invert_query_groups(attn_grouped, inter.inv_tau_kv, inter.inv_tau_group)
        attn_output = self.layout.merge_query_groups(attn_grouped)
        if transposed:
            attn_output = attn_output.transpose(1, 2)
        return attn_output

    def _restore_qkv_for_compare(
        self,
        q_states: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
    ) -> ComplexAttentionTraceData:
        inter = self.attention_config.inter_head
        q_grouped = self.layout.reshape_query_groups(q_states)
        q_grouped = self.layout.invert_query_groups(q_grouped, inter.inv_tau_kv, inter.inv_tau_group)
        q_states = self.layout.merge_query_groups(q_grouped)
        if inter.inv_tau_kv is not None:
            k_states = self.layout.permute_kv_heads(k_states, inter.inv_tau_kv)
            v_states = self.layout.permute_kv_heads(v_states, inter.inv_tau_kv)
        q_states = _apply_last_dim_matrix(q_states, self.attention_config.intra_head.q_inverse)
        k_states = _apply_last_dim_matrix(k_states, self.attention_config.intra_head.k_inverse)
        return ComplexAttentionTraceData(
            q_complex=q_states,
            k_complex=k_states,
            v_complex=v_states,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_linear = F.linear(
            hidden_states,
            self.q_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.q_bias is None else self.q_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        key_linear = F.linear(
            hidden_states,
            self.k_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.k_bias is None else self.k_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        value_linear = F.linear(
            hidden_states,
            self.v_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.v_bias is None else self.v_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )

        q_states = query_linear.view(hidden_shape).transpose(1, 2)
        k_states = key_linear.view(hidden_shape).transpose(1, 2)
        v_states = value_linear.view(hidden_shape).transpose(1, 2)

        q_states, k_states = self._apply_intra_head(q_states, k_states)
        q_states, k_states, v_states = self._apply_inter_head(q_states, k_states, v_states)

        restored = self._restore_qkv_for_compare(q_states, k_states, v_states)
        self.recorder.record(f"{self.prefix}_q_proj_out", restored.q_complex.transpose(1, 2).reshape(*input_shape, -1))
        self.recorder.record(f"{self.prefix}_k_proj_out", restored.k_complex.transpose(1, 2).reshape(*input_shape, -1))
        self.recorder.record(f"{self.prefix}_v_proj_out", restored.v_complex.transpose(1, 2).reshape(*input_shape, -1))

        cos, sin = position_embeddings
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k_states, v_states = past_key_values.update(k_states, v_states, self.layer_idx_attr, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            q_states,
            k_states,
            v_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        self.recorder.record(f"{self.prefix}_attn_heads_pre_inverse_raw", attn_output)
        attn_output = self._invert_inter_head(attn_output)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = F.linear(
            attn_output,
            self.o_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.o_bias is None else self.o_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        restored_attn = torch.matmul(
            attn_output.to(torch.float32),
            self.restore_matrix.to(device=hidden_states.device, dtype=torch.float32),
        )
        self.recorder.record(f"{self.prefix}_attn_out", restored_attn)
        return attn_output, attn_weights


def build_keymat_attention_bridge_norm_fused(
    attention_module: nn.Module,
    keymat_transform: KeyMatTransform,
    input_norm_weight: torch.Tensor,
    recorder: TraceRecorder,
    layer_idx: int,
    attention_profile: str,
    seed: int,
    qk_scale_range: tuple[float, float] = (0.95, 1.05),
    beta: int = 4,
    gamma: float = 1e3,
    rope_base: float = 10000.0,
) -> KeyMatAttentionBridgeNormFused:
    return KeyMatAttentionBridgeNormFused(
        attention_module=attention_module,
        keymat_transform=keymat_transform,
        input_norm_weight=input_norm_weight,
        recorder=recorder,
        layer_idx=layer_idx,
        attention_profile=attention_profile,
        seed=seed,
        qk_scale_range=qk_scale_range,
        beta=beta,
        gamma=gamma,
        rope_base=rope_base,
    )


def build_keymat_fused_attention(
    attention_module: nn.Module,
    keymat_transform: KeyMatTransform,
    input_norm_weight: torch.Tensor,
    recorder: TraceRecorder,
    layer_idx: int,
    attention_profile: str,
    seed: int,
    qk_scale_range: tuple[float, float] = (0.95, 1.05),
    beta: int = 4,
    gamma: float = 1e3,
    rope_base: float = 10000.0,
) -> KeyMatFusedQwen2Attention:
    return KeyMatFusedQwen2Attention(
        attention_module=attention_module,
        keymat_transform=keymat_transform,
        input_norm_weight=input_norm_weight,
        recorder=recorder,
        layer_idx=layer_idx,
        attention_profile=attention_profile,
        seed=seed,
        qk_scale_range=qk_scale_range,
        beta=beta,
        gamma=gamma,
        rope_base=rope_base,
    )
