from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from src.keymat import KeyMatTransform
from src.keymat_embed_head import (
    KeyMatEmbeddingWrapper,
    KeyMatHeadWrapper,
    add_head_noise,
    build_keymat_embed_head_artifacts,
)
from src.stage_f import (
    LayerStageFConfig,
    KeyMatDecoderLayerHandoff,
    _clear_copied_hooks,
    _ensure_list,
)
from src.stage_b import TraceRecorder, prepare_stage_a_model
from src.stage_g_attention import build_keymat_attention_bridge_norm_fused, build_keymat_fused_attention
from src.stage_g_ffn import build_keymat_ffn_bridge_norm_fused, build_keymat_fused_ffn
from src.stage_g_norm import build_keymat_fused_rmsnorm
from src.keymat_norm import estimate_kappa_for_keymat


@dataclass(frozen=True)
class LayerStageGConfig:
    keymat_transform: KeyMatTransform
    input_kappa: float
    post_attn_kappa: float
    ffn_transform: object
    attention_profile: str
    alpha_e: float = 0.0
    alpha_h: float = 0.0


def build_layer_stage_g_configs(
    layer_configs_f: dict[int, LayerStageFConfig],
) -> dict[int, LayerStageGConfig]:
    return {
        idx: LayerStageGConfig(
            keymat_transform=cfg.keymat_transform,
            input_kappa=cfg.input_kappa,
            post_attn_kappa=cfg.post_attn_kappa,
            ffn_transform=cfg.ffn_transform,
            attention_profile=cfg.attention_profile,
            alpha_e=cfg.alpha_e,
            alpha_h=cfg.alpha_h,
        )
        for idx, cfg in layer_configs_f.items()
    }


class StageGModel(nn.Module):
    def __init__(
        self,
        stage_a_model,
        keymat_transform: KeyMatTransform,
        recorder: TraceRecorder | None,
        layer_configs: dict[int, LayerStageGConfig],
        adapted_layers: Iterable[int],
        movable_ids: torch.Tensor,
        seed: int,
        mode: str,
        alpha_e: float = 0.0,
        alpha_h: float = 0.0,
        handoff_layer: int | None = None,
        use_keymat_head: bool = True,
        qk_scale_range: tuple[float, float] = (0.95, 1.05),
        beta: int = 4,
        gamma: float = 1e3,
    ) -> None:
        super().__init__()
        if mode not in {"norm_fused", "ffn_fused", "attention_fused"}:
            raise ValueError(f"Unsupported stage-G mode: {mode}")
        self.stage_a_model = stage_a_model
        self.mode = mode
        self.keymat_transform = keymat_transform
        self.recorder = recorder
        self.adapted_layers = _ensure_list(adapted_layers)

        artifacts = build_keymat_embed_head_artifacts(
            stage_a_model=stage_a_model,
            keymat_transform=keymat_transform,
            alpha_e=alpha_e,
            alpha_h=alpha_h,
            seed=seed,
            movable_ids=movable_ids,
        )
        original_embed = self.stage_a_model.model.embed_tokens
        self.stage_a_model.model.embed_tokens = KeyMatEmbeddingWrapper(
            obfuscated_weight=artifacts.embed_weight_obf,
            base_weight_for_recording=original_embed.weight.detach().cpu().to(torch.float32),
            recorder=recorder,
        )

        rope_base = float(getattr(self.stage_a_model.config, "rope_theta", 10000.0))

        for layer_idx in self.adapted_layers:
            layer = self.stage_a_model.model.layers[layer_idx]
            config = layer_configs[layer_idx]
            if recorder is not None:
                layer.register_forward_pre_hook(
                    lambda _, inputs, idx=layer_idx: recorder.record(
                        f"layer_{idx}_input",
                        torch.matmul(
                            inputs[0].to(torch.float32),
                            keymat_transform.inverse.to(device=inputs[0].device, dtype=torch.float32),
                        ),
                    )
                )
                layer.register_forward_hook(
                    lambda _, __, output, idx=layer_idx: recorder.record(
                        f"layer_{idx}_block_out",
                        torch.matmul(
                            output.to(torch.float32),
                            keymat_transform.inverse.to(device=output.device, dtype=torch.float32),
                        ),
                    )
                )

            input_norm_weight = layer.input_layernorm.weight.detach().to(torch.float32)
            post_norm_weight = layer.post_attention_layernorm.weight.detach().to(torch.float32)

            layer.input_layernorm = build_keymat_fused_rmsnorm(
                layer.input_layernorm,
                keymat_transform=keymat_transform,
                kappa=config.input_kappa,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_input_norm_out",
            )
            if mode == "attention_fused":
                layer.self_attn = build_keymat_fused_attention(
                    attention_module=layer.self_attn,
                    keymat_transform=keymat_transform,
                    input_norm_weight=input_norm_weight,
                    recorder=recorder or TraceRecorder(),
                    layer_idx=layer_idx,
                    attention_profile=config.attention_profile,
                    seed=seed + 10000 + layer_idx,
                    qk_scale_range=qk_scale_range,
                    beta=beta,
                    gamma=gamma,
                    rope_base=rope_base,
                )
            else:
                layer.self_attn = build_keymat_attention_bridge_norm_fused(
                    attention_module=layer.self_attn,
                    keymat_transform=keymat_transform,
                    input_norm_weight=input_norm_weight,
                    recorder=recorder or TraceRecorder(),
                    layer_idx=layer_idx,
                    attention_profile=config.attention_profile,
                    seed=seed + 10000 + layer_idx,
                    qk_scale_range=qk_scale_range,
                    beta=beta,
                    gamma=gamma,
                    rope_base=rope_base,
                )
            layer.post_attention_layernorm = build_keymat_fused_rmsnorm(
                layer.post_attention_layernorm,
                keymat_transform=keymat_transform,
                kappa=config.post_attn_kappa,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_post_attn_norm_out",
            )
            if mode in {"ffn_fused", "attention_fused"}:
                layer.mlp = build_keymat_fused_ffn(
                    mlp_module=layer.mlp,
                    keymat_transform=keymat_transform,
                    input_norm_weight=post_norm_weight,
                    ffn_transform=config.ffn_transform,
                    recorder=recorder,
                    record_name=f"layer_{layer_idx}_mlp_out",
                )
            else:
                layer.mlp = build_keymat_ffn_bridge_norm_fused(
                    mlp_module=layer.mlp,
                    keymat_transform=keymat_transform,
                    input_norm_weight=post_norm_weight,
                    ffn_transform=config.ffn_transform,
                    recorder=recorder,
                    record_name=f"layer_{layer_idx}_mlp_out",
                )

        if handoff_layer is not None:
            original_handoff = self.stage_a_model.model.layers[handoff_layer]
            self.stage_a_model.model.layers[handoff_layer] = KeyMatDecoderLayerHandoff(
                layer_module=original_handoff,
                keymat_transform=keymat_transform,
                layer_idx=handoff_layer,
                recorder=recorder,
            )

        if use_keymat_head:
            if handoff_layer is None:
                final_norm = self.stage_a_model.model.norm
                final_norm_weight = final_norm.weight.detach().to(torch.float32)
                self.stage_a_model.model.norm = build_keymat_fused_rmsnorm(
                    final_norm,
                    keymat_transform=keymat_transform,
                    kappa=estimate_kappa_for_keymat(
                        keymat_transform,
                        hidden_size=keymat_transform.hidden_size,
                        num_samples=1024,
                        seed=seed + 12345,
                    ),
                    recorder=recorder,
                    record_name="final_norm_out",
                )
                head_weight = self.stage_a_model.get_output_embeddings().weight.detach().cpu().to(torch.float32)
                noisy_head = add_head_noise(head_weight, alpha_h=alpha_h, seed=seed + 202, movable_ids=movable_ids)
                fused_head = (noisy_head * final_norm_weight.unsqueeze(0)) @ keymat_transform.inverse.to(torch.float32).T
                self.stage_a_model.lm_head = KeyMatHeadWrapper(
                    obfuscated_weight=fused_head,
                    keymat_transform=keymat_transform,
                    expects_obfuscated_input=True,
                )
            else:
                self.stage_a_model.lm_head = KeyMatHeadWrapper(
                    obfuscated_weight=artifacts.head_weight_obf,
                    keymat_transform=keymat_transform,
                    expects_obfuscated_input=False,
                )

    def forward(self, *args, **kwargs):
        return self.stage_a_model(*args, **kwargs)


def build_stage_g_model(
    baseline_model,
    tokenizer,
    keymat_transform: KeyMatTransform,
    seed: int,
    recorder: TraceRecorder | None,
    layer_configs: dict[int, LayerStageGConfig],
    adapted_layers: Iterable[int],
    mode: str,
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
    use_keymat_head: bool = True,
    qk_scale_range: tuple[float, float] = (0.95, 1.05),
    beta: int = 4,
    gamma: float = 1e3,
):
    adapted_layers = _ensure_list(adapted_layers)
    total_layers = baseline_model.config.num_hidden_layers
    if adapted_layers and adapted_layers != list(range(max(adapted_layers) + 1)):
        raise ValueError("Stage-G currently supports only contiguous prefix adapted layers.")
    handoff_layer = None if len(adapted_layers) == total_layers else len(adapted_layers)
    stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, seed)
    _clear_copied_hooks(stage_a_model)
    movable_ids = torch.arange(tokenizer.vocab_size, dtype=torch.long)
    model = StageGModel(
        stage_a_model=stage_a_model,
        keymat_transform=keymat_transform,
        recorder=recorder,
        layer_configs=layer_configs,
        adapted_layers=adapted_layers,
        movable_ids=movable_ids,
        seed=seed,
        mode=mode,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
        handoff_layer=handoff_layer,
        use_keymat_head=use_keymat_head,
        qk_scale_range=qk_scale_range,
        beta=beta,
        gamma=gamma,
    )
    return model, perm_vocab, inv_perm_vocab
