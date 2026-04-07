from __future__ import annotations

from dataclasses import dataclass

from src.hidden_keys import HiddenTransform
from src.obfuscate_ffn import FFNTransform, obfuscate_ffn_block
from src.obfuscate_rmsnorm import apply_rmsnorm_obfuscation
from src.stage_b import TraceRecorder, TracingQwen2Attention


@dataclass(frozen=True)
class StageCConfig:
    hidden_transform: HiddenTransform
    kappa_input: float
    kappa_post_attn: float
    ffn_transform: FFNTransform


def attach_stage_c_hooks(
    model,
    recorder: TraceRecorder,
    attention_mode: str,
    stage_c_config: StageCConfig | None = None,
    input_norm_mode: str = "plain",
    post_attn_norm_mode: str = "plain",
    ffn_mode: str = "plain",
    capture_embed_output: bool = False,
):
    if input_norm_mode not in {"plain", "wrapper"}:
        raise ValueError(f"Unsupported input norm mode: {input_norm_mode}")
    if post_attn_norm_mode not in {"plain", "wrapper"}:
        raise ValueError(f"Unsupported post attention norm mode: {post_attn_norm_mode}")
    if ffn_mode not in {"plain", "wrapper"}:
        raise ValueError(f"Unsupported FFN mode: {ffn_mode}")
    if attention_mode not in {"plain", "wrapper"}:
        raise ValueError(f"Unsupported attention mode: {attention_mode}")

    handles = []
    if capture_embed_output:
        handles.append(
            model.model.embed_tokens.register_forward_hook(
                lambda _, __, output: recorder.record("embed_out", output)
            )
        )

    layer0 = model.model.layers[0]
    original_attention = layer0.self_attn
    original_input_norm = layer0.input_layernorm
    original_post_norm = layer0.post_attention_layernorm
    original_mlp = layer0.mlp

    layer0.self_attn = TracingQwen2Attention(
        attention_module=original_attention,
        recorder=recorder,
        mode=attention_mode,
        hidden_transform=stage_c_config.hidden_transform if stage_c_config else None,
    )

    if input_norm_mode == "wrapper":
        if stage_c_config is None:
            raise ValueError("stage_c_config is required for RMSNorm wrapper.")
        layer0.input_layernorm = apply_rmsnorm_obfuscation(
            norm_layer=original_input_norm,
            hidden_transform=stage_c_config.hidden_transform,
            kappa=stage_c_config.kappa_input,
            recorder=recorder,
            record_name="layer_0_input_norm_out",
        )
    else:
        handles.append(
            original_input_norm.register_forward_hook(
                lambda _, __, output: recorder.record("layer_0_input_norm_out", output)
            )
        )

    if post_attn_norm_mode == "wrapper":
        if stage_c_config is None:
            raise ValueError("stage_c_config is required for RMSNorm wrapper.")
        layer0.post_attention_layernorm = apply_rmsnorm_obfuscation(
            norm_layer=original_post_norm,
            hidden_transform=stage_c_config.hidden_transform,
            kappa=stage_c_config.kappa_post_attn,
            recorder=recorder,
            record_name="layer_0_post_attn_norm_out",
        )
    else:
        handles.append(
            original_post_norm.register_forward_hook(
                lambda _, __, output: recorder.record("layer_0_post_attn_norm_out", output)
            )
        )

    if ffn_mode == "wrapper":
        if stage_c_config is None:
            raise ValueError("stage_c_config is required for FFN wrapper.")
        layer0.mlp = obfuscate_ffn_block(
            mlp_module=original_mlp,
            hidden_transform=stage_c_config.hidden_transform,
            ffn_transform=stage_c_config.ffn_transform,
            recorder=recorder,
            record_name="layer_0_mlp_out",
        )
    else:
        handles.append(
            original_mlp.register_forward_hook(
                lambda _, __, output: recorder.record("layer_0_mlp_out", output)
            )
        )

    def layer0_pre_hook(_, inputs):
        recorder.record("layer_0_input", inputs[0])

    def layer0_post_hook(_, __, output):
        recorder.record("layer_0_block_out", output)

    handles.append(layer0.register_forward_pre_hook(layer0_pre_hook))
    handles.append(layer0.register_forward_hook(layer0_post_hook))

    def cleanup() -> None:
        for handle in handles:
            handle.remove()
        layer0.self_attn = original_attention
        layer0.input_layernorm = original_input_norm
        layer0.post_attention_layernorm = original_post_norm
        layer0.mlp = original_mlp

    return cleanup
