from __future__ import annotations

from torch import nn

from src.attention_keys import build_attention_complex_config
from src.hidden_keys import build_identity_hidden_transform
from src.keymat import KeyMatTransform, apply_inverse_keymat_transform, apply_keymat_transform
from src.obfuscate_attention_complex import ComplexQwen2Attention
from src.stage_b import TraceRecorder, TracingQwen2Attention


class KeyMatAttentionBridge(nn.Module):
    def __init__(
        self,
        attention_module: nn.Module,
        keymat_transform: KeyMatTransform,
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
        self.layer_idx = layer_idx
        self.attention_profile = attention_profile
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
        attn_output, attn_weights = self.inner(
            base_hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        return apply_keymat_transform(attn_output, self.keymat_transform), attn_weights
