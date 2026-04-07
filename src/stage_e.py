from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.attention_keys import AttentionComplexConfig, build_attention_complex_config
from src.hidden_keys import HiddenTransform
from src.obfuscate_attention_complex import ComplexQwen2Attention
from src.obfuscate_ffn import FFNTransform, obfuscate_ffn_block
from src.obfuscate_rmsnorm import apply_rmsnorm_obfuscation
from src.stage_b import TraceRecorder
from src.stage_d import LayerStageDConfig


@dataclass(frozen=True)
class LayerStageEConfig:
    hidden_transform: HiddenTransform
    input_kappa: float
    post_attn_kappa: float
    ffn_transform: FFNTransform
    attention_config: AttentionComplexConfig


def attach_stage_e_hooks(
    model,
    recorder: TraceRecorder,
    trace_layers: Iterable[int],
    layer_configs: dict[int, LayerStageEConfig] | None = None,
    adapted_attention_layers: Iterable[int] | None = None,
    adapted_norm_layers: Iterable[int] | None = None,
    adapted_ffn_layers: Iterable[int] | None = None,
    capture_embed_output: bool = False,
):
    trace_layers = list(sorted(set(trace_layers)))
    adapted_attention_layers = set(adapted_attention_layers or [])
    adapted_norm_layers = set(adapted_norm_layers or [])
    adapted_ffn_layers = set(adapted_ffn_layers or [])
    layer_configs = layer_configs or {}

    handles = []
    restorations: list[tuple[int, str, object]] = []

    if capture_embed_output:
        handles.append(
            model.model.embed_tokens.register_forward_hook(
                lambda _, __, output: recorder.record("embed_out", output)
            )
        )

    for layer_idx in trace_layers:
        layer = model.model.layers[layer_idx]
        prefix = f"layer_{layer_idx}"

        original_attention = layer.self_attn
        if layer_idx in adapted_attention_layers:
            if layer_idx not in layer_configs:
                raise ValueError(f"Missing layer config for adapted attention layer {layer_idx}")
            layer.self_attn = ComplexQwen2Attention(
                attention_module=original_attention,
                recorder=recorder,
                layer_idx=layer_idx,
                hidden_transform=layer_configs[layer_idx].hidden_transform,
                attention_config=layer_configs[layer_idx].attention_config,
            )
            restorations.append((layer_idx, "self_attn", original_attention))
        else:
            from src.stage_d import LayerTracingQwen2Attention

            layer.self_attn = LayerTracingQwen2Attention(
                attention_module=original_attention,
                recorder=recorder,
                layer_idx=layer_idx,
                mode="plain",
                hidden_transform=None,
            )
            restorations.append((layer_idx, "self_attn", original_attention))

        original_input_norm = layer.input_layernorm
        if layer_idx in adapted_norm_layers:
            layer.input_layernorm = apply_rmsnorm_obfuscation(
                norm_layer=original_input_norm,
                hidden_transform=layer_configs[layer_idx].hidden_transform,
                kappa=layer_configs[layer_idx].input_kappa,
                recorder=recorder,
                record_name=f"{prefix}_input_norm_out",
            )
            restorations.append((layer_idx, "input_layernorm", original_input_norm))
        else:
            handles.append(
                original_input_norm.register_forward_hook(
                    lambda _, __, output, name=f"{prefix}_input_norm_out": recorder.record(name, output)
                )
            )

        original_post_norm = layer.post_attention_layernorm
        if layer_idx in adapted_norm_layers:
            layer.post_attention_layernorm = apply_rmsnorm_obfuscation(
                norm_layer=original_post_norm,
                hidden_transform=layer_configs[layer_idx].hidden_transform,
                kappa=layer_configs[layer_idx].post_attn_kappa,
                recorder=recorder,
                record_name=f"{prefix}_post_attn_norm_out",
            )
            restorations.append((layer_idx, "post_attention_layernorm", original_post_norm))
        else:
            handles.append(
                original_post_norm.register_forward_hook(
                    lambda _, __, output, name=f"{prefix}_post_attn_norm_out": recorder.record(name, output)
                )
            )

        original_mlp = layer.mlp
        if layer_idx in adapted_ffn_layers:
            layer.mlp = obfuscate_ffn_block(
                mlp_module=original_mlp,
                hidden_transform=layer_configs[layer_idx].hidden_transform,
                ffn_transform=layer_configs[layer_idx].ffn_transform,
                recorder=recorder,
                record_name=f"{prefix}_mlp_out",
            )
            restorations.append((layer_idx, "mlp", original_mlp))
        else:
            handles.append(
                original_mlp.register_forward_hook(
                    lambda _, __, output, name=f"{prefix}_mlp_out": recorder.record(name, output)
                )
            )

        handles.append(
            layer.register_forward_pre_hook(
                lambda _, inputs, name=f"{prefix}_input": recorder.record(name, inputs[0])
            )
        )
        handles.append(
            layer.register_forward_hook(
                lambda _, __, output, name=f"{prefix}_block_out": recorder.record(name, output)
            )
        )

    def cleanup() -> None:
        for handle in handles:
            handle.remove()
        for layer_idx, attr_name, module in restorations:
            setattr(model.model.layers[layer_idx], attr_name, module)

    return cleanup


def build_layer_stage_e_configs(
    *,
    baseline_model,
    hidden_transform: HiddenTransform,
    kappa_by_layer: dict[int, dict[str, float]],
    layer_indices: Iterable[int],
    ffn_configs: dict[int, FFNTransform | LayerStageDConfig],
    attention_profile: str,
    seed: int,
    qk_scale_range: tuple[float, float] = (0.95, 1.05),
    beta: int = 4,
    gamma: float = 1e3,
) -> dict[int, LayerStageEConfig]:
    layer_indices = list(sorted(set(layer_indices)))
    configs: dict[int, LayerStageEConfig] = {}
    rope_base = float(getattr(baseline_model.config, "rope_theta", 10000.0))
    for layer_idx in layer_indices:
        attention_cfg = build_attention_complex_config(
            profile=attention_profile,
            head_dim=baseline_model.model.layers[layer_idx].self_attn.head_dim,
            num_kv_heads=baseline_model.config.num_key_value_heads,
            num_groups=baseline_model.config.num_attention_heads // baseline_model.config.num_key_value_heads,
            seed=seed + 5000 + layer_idx,
            qk_scale_range=qk_scale_range,
            beta=beta,
            gamma=gamma,
            rope_base=rope_base,
        )
        configs[layer_idx] = LayerStageEConfig(
            hidden_transform=hidden_transform,
            input_kappa=kappa_by_layer[layer_idx]["input"],
            post_attn_kappa=kappa_by_layer[layer_idx]["post_attn"],
            ffn_transform=ffn_configs[layer_idx].ffn_transform if isinstance(ffn_configs[layer_idx], LayerStageDConfig) else ffn_configs[layer_idx],
            attention_config=attention_cfg,
        )
    return configs
