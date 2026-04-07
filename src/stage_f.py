from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from src.evaluator import max_abs_error, mean_abs_error
from src.keymat import KeyMatTransform, apply_inverse_keymat_transform, apply_keymat_transform, build_keymat_transform
from src.keymat_attention_bridge import KeyMatAttentionBridge
from src.keymat_embed_head import KeyMatEmbeddingWrapper, KeyMatHeadWrapper, build_keymat_embed_head_artifacts
from src.keymat_ffn import build_keymat_ffn_wrapper
from src.keymat_norm import build_keymat_rmsnorm_wrapper, estimate_kappa_for_keymat
from src.model_loader import format_chat_prompt
from src.obfuscate_ffn import FFNTransform, build_ffn_transform, generate_ffn_permutation, generate_ffn_scaling
from src.stage_b import TraceRecorder, prepare_stage_a_model
from src.stage_d import attach_stage_d_hooks, manual_greedy_generate_baseline, manual_greedy_generate_stage_model
from src.transforms import map_input_ids, restore_logits


@dataclass(frozen=True)
class LayerStageFConfig:
    keymat_transform: KeyMatTransform
    input_kappa: float
    post_attn_kappa: float
    ffn_transform: FFNTransform
    attention_profile: str
    alpha_e: float = 0.0
    alpha_h: float = 0.0


@dataclass
class StageFRunResult:
    prompt: str
    mapped_input_ids: list[int]
    metrics: dict[str, float | bool | str]


def estimate_kappa_from_hidden_keymat(
    hidden_states: torch.Tensor,
    keymat_transform: KeyMatTransform,
) -> float:
    hidden = hidden_states.detach().cpu().to(torch.float32).reshape(-1, hidden_states.shape[-1])
    obf = apply_keymat_transform(hidden, keymat_transform)
    ratio = torch.linalg.vector_norm(obf, dim=-1) / torch.linalg.vector_norm(hidden, dim=-1)
    return float(ratio.mean().item())


def _ensure_list(indices: Iterable[int]) -> list[int]:
    return list(sorted(set(int(index) for index in indices)))


def _clear_copied_hooks(module: nn.Module) -> None:
    for submodule in module.modules():
        submodule._forward_hooks.clear()
        submodule._forward_pre_hooks.clear()
        submodule._backward_hooks.clear()


def calibrate_keymat_kappas(
    baseline_model,
    tokenizer,
    prompts: list[str],
    keymat_transform: KeyMatTransform,
    trace_layers: Iterable[int],
) -> dict[int, dict[str, float]]:
    trace_layers = _ensure_list(trace_layers)
    recorder = TraceRecorder()
    cleanup = attach_stage_d_hooks(
        baseline_model,
        recorder,
        trace_layers=trace_layers,
        layer_configs={},
        attention_mode="plain",
        adapted_attention_layers=[],
        adapted_norm_layers=[],
        adapted_ffn_layers=[],
        capture_embed_output=False,
        record_norm_inputs=True,
    )
    ratios: dict[int, dict[str, list[float]]] = {
        layer_idx: {"input": [], "post_attn": []}
        for layer_idx in trace_layers
    }
    try:
        for prompt in prompts:
            recorder.clear()
            encoded = tokenizer(format_chat_prompt(tokenizer, prompt), return_tensors="pt")
            baseline_model(**encoded)
            for layer_idx in trace_layers:
                ratios[layer_idx]["input"].append(
                    estimate_kappa_from_hidden_keymat(
                        recorder.tensors[f"layer_{layer_idx}_input_norm_in"],
                        keymat_transform,
                    )
                )
                ratios[layer_idx]["post_attn"].append(
                    estimate_kappa_from_hidden_keymat(
                        recorder.tensors[f"layer_{layer_idx}_post_attn_norm_in"],
                        keymat_transform,
                    )
                )
    finally:
        cleanup()
    return {
        layer_idx: {
            "input": float(sum(values["input"]) / len(values["input"])),
            "post_attn": float(sum(values["post_attn"]) / len(values["post_attn"])),
        }
        for layer_idx, values in ratios.items()
    }


def build_layer_stage_f_configs(
    baseline_model,
    keymat_transform: KeyMatTransform,
    trace_layers: Iterable[int],
    kappa_by_layer: dict[int, dict[str, float]],
    attention_profile: str,
    seed: int,
    ffn_scale_range: tuple[float, float] = (0.95, 1.05),
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
) -> dict[int, LayerStageFConfig]:
    configs: dict[int, LayerStageFConfig] = {}
    for layer_idx in _ensure_list(trace_layers):
        layer = baseline_model.model.layers[layer_idx]
        intermediate_size = layer.mlp.gate_proj.out_features
        ffn_transform = build_ffn_transform(
            generate_ffn_permutation(intermediate_size, seed=seed + 7000 + layer_idx),
            generate_ffn_scaling(intermediate_size, ffn_scale_range, seed=seed + 8000 + layer_idx),
        )
        kappas = kappa_by_layer[layer_idx]
        configs[layer_idx] = LayerStageFConfig(
            keymat_transform=keymat_transform,
            input_kappa=kappas["input"],
            post_attn_kappa=kappas["post_attn"],
            ffn_transform=ffn_transform,
            attention_profile=attention_profile,
            alpha_e=float(alpha_e),
            alpha_h=float(alpha_h),
        )
    return configs


class KeyMatDecoderLayerHandoff(nn.Module):
    def __init__(
        self,
        layer_module: nn.Module,
        keymat_transform: KeyMatTransform,
        layer_idx: int,
        recorder: TraceRecorder | None = None,
    ) -> None:
        super().__init__()
        self.layer_module = layer_module
        self.keymat_transform = keymat_transform
        self.layer_idx = layer_idx
        self.recorder = recorder
        self.attention_type = getattr(layer_module, "attention_type", "full_attention")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        base_hidden = apply_inverse_keymat_transform(hidden_states, self.keymat_transform)
        if self.recorder is not None:
            self.recorder.record(f"layer_{self.layer_idx}_input", base_hidden)
        output = self.layer_module(
            base_hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        if self.recorder is not None:
            self.recorder.record(f"layer_{self.layer_idx}_block_out", output)
        return output


class StageFModel(nn.Module):
    def __init__(
        self,
        stage_a_model,
        keymat_transform: KeyMatTransform,
        recorder: TraceRecorder | None,
        layer_configs: dict[int, LayerStageFConfig],
        adapted_layers: Iterable[int],
        movable_ids: torch.Tensor,
        seed: int,
        alpha_e: float = 0.0,
        alpha_h: float = 0.0,
        handoff_layer: int | None = None,
        use_keymat_head: bool = False,
        qk_scale_range: tuple[float, float] = (0.95, 1.05),
        beta: int = 4,
        gamma: float = 1e3,
    ) -> None:
        super().__init__()
        self.stage_a_model = stage_a_model
        self.keymat_transform = keymat_transform
        self.recorder = recorder
        self.layer_configs = layer_configs
        self.adapted_layers = _ensure_list(adapted_layers)
        self.handoff_layer = handoff_layer
        self.use_keymat_head = use_keymat_head
        self._handles = []

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
        trace_set = set(self.adapted_layers)
        for layer_idx in self.adapted_layers:
            layer = self.stage_a_model.model.layers[layer_idx]
            config = layer_configs[layer_idx]
            if recorder is not None:
                self._handles.append(
                    layer.register_forward_pre_hook(
                        lambda _, inputs, idx=layer_idx: recorder.record(
                            f"layer_{idx}_input",
                            apply_inverse_keymat_transform(inputs[0], keymat_transform),
                        )
                    )
                )
                self._handles.append(
                    layer.register_forward_hook(
                        lambda _, __, output, idx=layer_idx: recorder.record(
                            f"layer_{idx}_block_out",
                            apply_inverse_keymat_transform(output, keymat_transform),
                        )
                    )
                )

            layer.input_layernorm = build_keymat_rmsnorm_wrapper(
                layer.input_layernorm,
                keymat_transform=keymat_transform,
                kappa=config.input_kappa,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_input_norm_out",
            )
            layer.self_attn = KeyMatAttentionBridge(
                attention_module=layer.self_attn,
                keymat_transform=keymat_transform,
                recorder=recorder or TraceRecorder(),
                layer_idx=layer_idx,
                attention_profile=config.attention_profile,
                seed=seed + 9000 + layer_idx,
                qk_scale_range=qk_scale_range,
                beta=beta,
                gamma=gamma,
                rope_base=rope_base,
            )
            layer.post_attention_layernorm = build_keymat_rmsnorm_wrapper(
                layer.post_attention_layernorm,
                keymat_transform=keymat_transform,
                kappa=config.post_attn_kappa,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_post_attn_norm_out",
            )
            layer.mlp = build_keymat_ffn_wrapper(
                layer.mlp,
                keymat_transform=keymat_transform,
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
            original_norm = self.stage_a_model.model.norm
            if handoff_layer is None:
                self.stage_a_model.model.norm = build_keymat_rmsnorm_wrapper(
                    original_norm,
                    keymat_transform=keymat_transform,
                    kappa=estimate_kappa_for_keymat(
                        keymat_transform,
                        hidden_size=keymat_transform.hidden_size,
                        num_samples=1024,
                        seed=seed + 11111,
                    ),
                    recorder=recorder,
                    record_name="final_norm_out",
                )
            self.stage_a_model.lm_head = KeyMatHeadWrapper(
                obfuscated_weight=artifacts.head_weight_obf,
                keymat_transform=keymat_transform,
                expects_obfuscated_input=handoff_layer is None,
            )

    def forward(self, *args, **kwargs):
        return self.stage_a_model(*args, **kwargs)


def build_stage_f_model(
    baseline_model,
    tokenizer,
    keymat_transform: KeyMatTransform,
    seed: int,
    recorder: TraceRecorder | None,
    layer_configs: dict[int, LayerStageFConfig],
    adapted_layers: Iterable[int],
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
        raise ValueError("Stage-F currently supports only contiguous prefix adapted layers.")
    handoff_layer = None if len(adapted_layers) == total_layers else len(adapted_layers)
    stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, seed)
    _clear_copied_hooks(stage_a_model)
    movable_ids = torch.arange(tokenizer.vocab_size, dtype=torch.long)
    stage_model = StageFModel(
        stage_a_model=stage_a_model,
        keymat_transform=keymat_transform,
        recorder=recorder,
        layer_configs=layer_configs,
        adapted_layers=adapted_layers,
        movable_ids=movable_ids,
        seed=seed,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
        handoff_layer=handoff_layer,
        use_keymat_head=use_keymat_head,
        qk_scale_range=qk_scale_range,
        beta=beta,
        gamma=gamma,
    )
    return stage_model, perm_vocab, inv_perm_vocab


def summarize_stage_f_metrics(
    baseline_recorder: TraceRecorder,
    observed_recorder: TraceRecorder,
    baseline_logits: torch.Tensor,
    observed_logits_perm: torch.Tensor,
    perm_vocab: torch.Tensor,
    trace_layers: Iterable[int],
) -> dict[str, float | bool]:
    trace_layers = _ensure_list(trace_layers)
    restored_logits = restore_logits(observed_logits_perm, perm_vocab)
    metrics: dict[str, float | bool] = {
        "final_logits_restored_max_abs_error": max_abs_error(baseline_logits, restored_logits),
        "final_logits_restored_mean_abs_error": mean_abs_error(baseline_logits, restored_logits),
    }
    global_direct_keys = ["embed_out"]
    for key in global_direct_keys:
        if key in baseline_recorder.tensors and key in observed_recorder.tensors:
            metrics[f"{key}_max_abs_error"] = max_abs_error(
                baseline_recorder.tensors[key],
                observed_recorder.tensors[key],
            )
            metrics[f"{key}_mean_abs_error"] = mean_abs_error(
                baseline_recorder.tensors[key],
                observed_recorder.tensors[key],
            )

    for layer_idx in trace_layers:
        prefix = f"layer_{layer_idx}"
        direct_keys = [
            "input",
            "input_norm_out",
            "q_proj_out",
            "k_proj_out",
            "v_proj_out",
            "attn_out",
            "post_attn_norm_out",
            "mlp_out",
            "block_out",
        ]
        for key in direct_keys:
            name = f"{prefix}_{key}"
            if name in baseline_recorder.tensors and name in observed_recorder.tensors:
                metrics[f"{name}_max_abs_error"] = max_abs_error(
                    baseline_recorder.tensors[name],
                    observed_recorder.tensors[name],
                )
                metrics[f"{name}_mean_abs_error"] = mean_abs_error(
                    baseline_recorder.tensors[name],
                    observed_recorder.tensors[name],
                )
    return metrics


@torch.inference_mode()
def run_stage_f_single_prompt(
    baseline_model,
    tokenizer,
    prompt: str,
    stage_model: StageFModel,
    perm_vocab: torch.Tensor,
    inv_perm_vocab: torch.Tensor,
    baseline_recorder: TraceRecorder,
    observed_recorder: TraceRecorder,
    trace_layers: Iterable[int],
    max_new_tokens: int = 8,
) -> StageFRunResult:
    del inv_perm_vocab
    baseline_recorder.clear()
    observed_recorder.clear()
    encoded = tokenizer(format_chat_prompt(tokenizer, prompt), return_tensors="pt")
    baseline_logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)
    mapped_input_ids = map_input_ids(encoded["input_ids"], perm_vocab)
    observed_logits_perm = stage_model(
        input_ids=mapped_input_ids,
        attention_mask=encoded.get("attention_mask"),
    ).logits.detach().cpu().to(torch.float32)
    metrics = summarize_stage_f_metrics(
        baseline_recorder=baseline_recorder,
        observed_recorder=observed_recorder,
        baseline_logits=baseline_logits,
        observed_logits_perm=observed_logits_perm,
        perm_vocab=perm_vocab,
        trace_layers=trace_layers,
    )
    baseline_generated_ids, baseline_generated_text = manual_greedy_generate_baseline(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    observed_generated_ids, observed_generated_text = manual_greedy_generate_stage_model(
        stage_model=stage_model,
        tokenizer=tokenizer,
        prompt=prompt,
        perm_vocab=perm_vocab,
        inv_perm_vocab=torch.empty(0),
        max_new_tokens=max_new_tokens,
    )
    metrics["greedy_first_token_match"] = bool(baseline_generated_ids[0] == observed_generated_ids[0])
    metrics["generated_ids_exact_match"] = bool(baseline_generated_ids == observed_generated_ids)
    metrics["generated_text_exact_match"] = bool(baseline_generated_text == observed_generated_text)
    metrics["baseline_generated_text"] = baseline_generated_text
    metrics["restored_generated_text"] = observed_generated_text
    return StageFRunResult(
        prompt=prompt,
        mapped_input_ids=mapped_input_ids[0].tolist(),
        metrics=metrics,
    )


def aggregate_stage_f_results(results: list[StageFRunResult]) -> dict[str, float]:
    numeric_keys = [
        key
        for key, value in results[0].metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    summary: dict[str, float] = {}
    for key in numeric_keys:
        summary[f"avg_{key}"] = float(sum(float(result.metrics[key]) for result in results) / len(results))
    for bool_key in ["greedy_first_token_match", "generated_ids_exact_match", "generated_text_exact_match"]:
        if bool_key in results[0].metrics:
            summary[f"{bool_key}_rate"] = float(sum(1.0 for result in results if result.metrics[bool_key]) / len(results))
    return summary


def build_default_stage_f_keymat(
    baseline_model,
    lam: float,
    h: int,
    seed: int,
) -> KeyMatTransform:
    return build_keymat_transform(
        d=baseline_model.config.hidden_size,
        h=h,
        lam=lam,
        init_seed=seed + 12000,
        key_seed=seed + 12100,
        inv_seed=seed + 12200,
        dtype=torch.float32,
    )
