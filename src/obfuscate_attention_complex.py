from __future__ import annotations

from dataclasses import dataclass

from copy import deepcopy
import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward

from src.attention_keys import AttentionComplexConfig
from src.gqa_layout import GQALayout
from src.hidden_keys import HiddenTransform
from src.stage_b import TraceRecorder
from src.transforms import apply_hidden_transform, apply_inverse_hidden_transform


def _apply_last_dim_matrix(tensor: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.to(device=tensor.device, dtype=tensor.dtype)
    return torch.matmul(tensor, matrix)


@dataclass
class ComplexAttentionTraceData:
    q_complex: torch.Tensor
    k_complex: torch.Tensor
    v_complex: torch.Tensor


class ComplexQwen2Attention(nn.Module):
    def __init__(
        self,
        attention_module: nn.Module,
        recorder: TraceRecorder,
        layer_idx: int,
        hidden_transform: HiddenTransform,
        attention_config: AttentionComplexConfig,
    ) -> None:
        super().__init__()
        self.recorder = recorder
        self.layer_idx = layer_idx
        self.prefix = f"layer_{layer_idx}"
        self.hidden_transform = hidden_transform
        self.attention_config = attention_config

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

        self.q_proj = attention_module.q_proj
        self.k_proj = attention_module.k_proj
        self.v_proj = attention_module.v_proj
        self.o_proj = attention_module.o_proj

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

    def _invert_inter_head(self, attn_output: torch.Tensor, v_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        if inter.inv_tau_kv is not None:
            v_states = self.layout.permute_kv_heads(v_states, inter.inv_tau_kv)
        return attn_output, v_states

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

        base_hidden = apply_inverse_hidden_transform(hidden_states, self.hidden_transform)
        q_linear_base = self.q_proj(base_hidden)
        k_linear_base = self.k_proj(base_hidden)
        v_linear_base = self.v_proj(base_hidden)

        q_states = q_linear_base.view(hidden_shape).transpose(1, 2)
        k_states = k_linear_base.view(hidden_shape).transpose(1, 2)
        v_states = v_linear_base.view(hidden_shape).transpose(1, 2)

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
        attn_output, _ = self._invert_inter_head(attn_output, v_states)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = apply_hidden_transform(attn_output, self.hidden_transform)
        self.recorder.record(f"{self.prefix}_attn_out", attn_output)
        return attn_output, attn_weights


def _repeat_block_matrix(matrix: torch.Tensor, repeats: int) -> torch.Tensor:
    return torch.block_diag(*[matrix for _ in range(repeats)])


def fuse_intra_head_qk_transforms(attention_module: nn.Module, attention_config: AttentionComplexConfig) -> nn.Module:
    fused_attention = deepcopy(attention_module)
    q_matrix = attention_config.intra_head.q_matrix.to(
        device=fused_attention.q_proj.weight.device,
        dtype=fused_attention.q_proj.weight.dtype,
    )
    k_matrix = attention_config.intra_head.k_matrix.to(
        device=fused_attention.k_proj.weight.device,
        dtype=fused_attention.k_proj.weight.dtype,
    )
    q_block = _repeat_block_matrix(q_matrix, fused_attention.config.num_attention_heads)
    k_block = _repeat_block_matrix(k_matrix, fused_attention.config.num_key_value_heads)
    with torch.no_grad():
        fused_attention.q_proj.weight.copy_(q_block.T @ fused_attention.q_proj.weight)
        fused_attention.k_proj.weight.copy_(k_block.T @ fused_attention.k_proj.weight)
        if fused_attention.q_proj.bias is not None:
            fused_attention.q_proj.bias.copy_(q_block.T @ fused_attention.q_proj.bias)
        if fused_attention.k_proj.bias is not None:
            fused_attention.k_proj.bias.copy_(k_block.T @ fused_attention.k_proj.bias)
    return fused_attention
