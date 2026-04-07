from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward

from src.evaluator import max_abs_error, mean_abs_error
from src.hidden_keys import HiddenTransform
from src.model_loader import format_chat_prompt
from src.obfuscate_ffn import (
    FFNTransform,
    build_ffn_transform,
    generate_ffn_permutation,
    generate_ffn_scaling,
    obfuscate_ffn_block,
)
from src.obfuscate_rmsnorm import apply_rmsnorm_obfuscation
from src.stage_b import StageBHiddenPermutationModel, TraceRecorder
from src.transforms import apply_hidden_transform, apply_inverse_hidden_transform, map_input_ids, restore_logits, unmap_output_ids


@dataclass(frozen=True)
class LayerStageDConfig:
    hidden_transform: HiddenTransform
    input_kappa: float
    post_attn_kappa: float
    ffn_transform: FFNTransform


@dataclass
class StageDRunResult:
    prompt: str
    mapped_input_ids: list[int]
    metrics: dict[str, float | bool]


class LayerTracingQwen2Attention(nn.Module):
    def __init__(
        self,
        attention_module: nn.Module,
        recorder: TraceRecorder,
        layer_idx: int,
        mode: str = "plain",
        hidden_transform: HiddenTransform | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"plain", "wrapper"}:
            raise ValueError(f"Unsupported attention tracing mode: {mode}")
        self.mode = mode
        self.recorder = recorder
        self.layer_idx = layer_idx
        self.hidden_transform = hidden_transform
        self.record_prefix = f"layer_{layer_idx}"

        self.config = attention_module.config
        self.layer_idx_attr = attention_module.layer_idx
        self.head_dim = attention_module.head_dim
        self.scaling = attention_module.scaling
        self.sliding_window = attention_module.sliding_window
        self.attention_dropout = attention_module.attention_dropout
        self.is_causal = getattr(attention_module, "is_causal", True)
        if hasattr(attention_module, "num_key_value_groups"):
            self.num_key_value_groups = attention_module.num_key_value_groups

        self.q_proj = attention_module.q_proj
        self.k_proj = attention_module.k_proj
        self.v_proj = attention_module.v_proj
        self.o_proj = attention_module.o_proj

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

        project_hidden = hidden_states
        if self.mode == "wrapper":
            if self.hidden_transform is None:
                raise ValueError("hidden_transform is required in wrapper mode")
            project_hidden = apply_inverse_hidden_transform(hidden_states, self.hidden_transform)

        query_linear = self.q_proj(project_hidden)
        key_linear = self.k_proj(project_hidden)
        value_linear = self.v_proj(project_hidden)
        self.recorder.record(f"{self.record_prefix}_q_proj_out", query_linear)
        self.recorder.record(f"{self.record_prefix}_k_proj_out", key_linear)
        self.recorder.record(f"{self.record_prefix}_v_proj_out", value_linear)

        query_states = query_linear.view(hidden_shape).transpose(1, 2)
        key_states = key_linear.view(hidden_shape).transpose(1, 2)
        value_states = value_linear.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx_attr, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if self.mode == "wrapper":
            attn_output = apply_hidden_transform(attn_output, self.hidden_transform)
        self.recorder.record(f"{self.record_prefix}_attn_out", attn_output)
        return attn_output, attn_weights


def _ensure_list(indices: Iterable[int]) -> list[int]:
    return list(sorted(set(indices)))


def estimate_kappa_from_hidden(
    hidden_states: torch.Tensor,
    hidden_transform: HiddenTransform,
) -> float:
    hidden = hidden_states.detach().cpu().to(torch.float32).reshape(-1, hidden_states.shape[-1])
    obf = apply_hidden_transform(hidden, hidden_transform)
    ratio = torch.linalg.vector_norm(obf, dim=-1) / torch.linalg.vector_norm(hidden, dim=-1)
    return float(ratio.mean().item())


def attach_stage_d_hooks(
    model,
    recorder: TraceRecorder,
    trace_layers: Iterable[int],
    layer_configs: dict[int, LayerStageDConfig] | None = None,
    attention_mode: str = "plain",
    adapted_attention_layers: Iterable[int] | None = None,
    adapted_norm_layers: Iterable[int] | None = None,
    adapted_ffn_layers: Iterable[int] | None = None,
    capture_embed_output: bool = False,
    record_norm_inputs: bool = False,
):
    trace_layers = _ensure_list(trace_layers)
    adapted_attention_layers = set(adapted_attention_layers or [])
    adapted_norm_layers = set(adapted_norm_layers or [])
    adapted_ffn_layers = set(adapted_ffn_layers or [])
    layer_configs = layer_configs or {}

    handles = []
    restorations: list[tuple[int, str, nn.Module]] = []

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
        attention_hidden_transform = None
        if layer_idx in adapted_attention_layers:
            if layer_idx not in layer_configs:
                raise ValueError(f"Missing layer config for adapted attention layer {layer_idx}")
            attention_hidden_transform = layer_configs[layer_idx].hidden_transform
        layer.self_attn = LayerTracingQwen2Attention(
            attention_module=original_attention,
            recorder=recorder,
            layer_idx=layer_idx,
            mode="wrapper" if layer_idx in adapted_attention_layers else "plain",
            hidden_transform=attention_hidden_transform,
        )
        restorations.append((layer_idx, "self_attn", original_attention))

        original_input_norm = layer.input_layernorm
        if record_norm_inputs:
            handles.append(
                layer.input_layernorm.register_forward_pre_hook(
                    lambda _, inputs, name=f"{prefix}_input_norm_in": recorder.record(name, inputs[0])
                )
            )
        if layer_idx in adapted_norm_layers:
            if layer_idx not in layer_configs:
                raise ValueError(f"Missing layer config for adapted norm layer {layer_idx}")
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
        if record_norm_inputs:
            handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    lambda _, inputs, name=f"{prefix}_post_attn_norm_in": recorder.record(name, inputs[0])
                )
            )
        if layer_idx in adapted_norm_layers:
            if layer_idx not in layer_configs:
                raise ValueError(f"Missing layer config for adapted norm layer {layer_idx}")
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
            if layer_idx not in layer_configs:
                raise ValueError(f"Missing layer config for adapted FFN layer {layer_idx}")
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


def calibrate_layer_kappas(
    baseline_model,
    tokenizer,
    prompts: list[str],
    hidden_transform: HiddenTransform,
    trace_layers: Iterable[int],
) -> dict[int, dict[str, float]]:
    trace_layers = _ensure_list(trace_layers)
    recorder = TraceRecorder()
    cleanup = attach_stage_d_hooks(
        baseline_model,
        recorder,
        trace_layers=trace_layers,
        attention_mode="plain",
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
            encoded = tokenizer(
                format_chat_prompt(tokenizer, prompt),
                return_tensors="pt",
            )
            baseline_model(**encoded)
            for layer_idx in trace_layers:
                ratios[layer_idx]["input"].append(
                    estimate_kappa_from_hidden(
                        recorder.tensors[f"layer_{layer_idx}_input_norm_in"],
                        hidden_transform,
                    )
                )
                ratios[layer_idx]["post_attn"].append(
                    estimate_kappa_from_hidden(
                        recorder.tensors[f"layer_{layer_idx}_post_attn_norm_in"],
                        hidden_transform,
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


def build_layer_configs(
    baseline_model,
    hidden_transform: HiddenTransform,
    kappa_by_layer: dict[int, dict[str, float]],
    trace_layers: Iterable[int],
    seed: int,
    ffn_scale_range: tuple[float, float],
) -> dict[int, LayerStageDConfig]:
    configs: dict[int, LayerStageDConfig] = {}
    for layer_idx in _ensure_list(trace_layers):
        layer = baseline_model.model.layers[layer_idx]
        intermediate_size = layer.mlp.gate_proj.out_features
        ffn_transform = build_ffn_transform(
            generate_ffn_permutation(intermediate_size, seed=seed + 1000 + layer_idx),
            generate_ffn_scaling(intermediate_size, scale_range=ffn_scale_range, seed=seed + 2000 + layer_idx),
        )
        kappas = kappa_by_layer[layer_idx]
        configs[layer_idx] = LayerStageDConfig(
            hidden_transform=hidden_transform,
            input_kappa=kappas["input"],
            post_attn_kappa=kappas["post_attn"],
            ffn_transform=ffn_transform,
        )
    return configs


def summarize_stage_d_metrics(
    baseline_recorder: TraceRecorder,
    observed_recorder: TraceRecorder,
    baseline_logits: torch.Tensor,
    observed_logits_perm: torch.Tensor,
    perm_vocab: torch.Tensor,
    hidden_transform: HiddenTransform,
    trace_layers: Iterable[int],
) -> dict[str, float | bool]:
    trace_layers = _ensure_list(trace_layers)
    restored_logits = restore_logits(observed_logits_perm, perm_vocab)
    metrics: dict[str, float | bool] = {
        "final_logits_restored_max_abs_error": max_abs_error(baseline_logits, restored_logits),
        "final_logits_restored_mean_abs_error": mean_abs_error(baseline_logits, restored_logits),
    }

    for layer_idx in trace_layers:
        prefix = f"layer_{layer_idx}"
        restored_hidden_keys = [
            "input",
            "input_norm_out",
            "attn_out",
            "post_attn_norm_out",
            "mlp_out",
            "block_out",
        ]
        direct_compare_keys = [
            "q_proj_out",
            "k_proj_out",
            "v_proj_out",
        ]
        for key in restored_hidden_keys:
            baseline_name = f"{prefix}_{key}"
            observed_name = baseline_name
            if baseline_name in baseline_recorder.tensors and observed_name in observed_recorder.tensors:
                restored = apply_inverse_hidden_transform(observed_recorder.tensors[observed_name], hidden_transform)
                metrics[f"{baseline_name}_restored_max_abs_error"] = max_abs_error(
                    baseline_recorder.tensors[baseline_name],
                    restored,
                )
                metrics[f"{baseline_name}_restored_mean_abs_error"] = mean_abs_error(
                    baseline_recorder.tensors[baseline_name],
                    restored,
                )
        for key in direct_compare_keys:
            baseline_name = f"{prefix}_{key}"
            observed_name = baseline_name
            if baseline_name in baseline_recorder.tensors and observed_name in observed_recorder.tensors:
                metrics[f"{baseline_name}_max_abs_error"] = max_abs_error(
                    baseline_recorder.tensors[baseline_name],
                    observed_recorder.tensors[observed_name],
                )
                metrics[f"{baseline_name}_mean_abs_error"] = mean_abs_error(
                    baseline_recorder.tensors[baseline_name],
                    observed_recorder.tensors[observed_name],
                )

    return metrics


@torch.inference_mode()
def manual_greedy_generate_baseline(
    baseline_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    text = format_chat_prompt(tokenizer, prompt)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    generated_ids: list[int] = []
    current_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = baseline_model(input_ids=current_ids).logits.detach().cpu().to(torch.float32)
        next_token = int(torch.argmax(logits[0, -1]).item())
        generated_ids.append(next_token)
        next_tensor = torch.tensor([[next_token]], dtype=current_ids.dtype)
        current_ids = torch.cat([current_ids, next_tensor], dim=1)
    return generated_ids, tokenizer.decode(generated_ids, skip_special_tokens=True)


@torch.inference_mode()
def manual_greedy_generate_stage_model(
    stage_model: StageBHiddenPermutationModel,
    tokenizer,
    prompt: str,
    perm_vocab: torch.Tensor,
    inv_perm_vocab: torch.Tensor,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    text = format_chat_prompt(tokenizer, prompt)
    original_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    current_original_ids = original_ids.clone()
    generated_original_ids: list[int] = []

    for _ in range(max_new_tokens):
        mapped_ids = map_input_ids(current_original_ids, perm_vocab)
        logits_perm = stage_model(input_ids=mapped_ids).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(logits_perm[0, -1], perm_vocab)
        next_original_token = int(torch.argmax(restored_logits).item())
        generated_original_ids.append(next_original_token)
        next_tensor = torch.tensor([[next_original_token]], dtype=current_original_ids.dtype)
        current_original_ids = torch.cat([current_original_ids, next_tensor], dim=1)

    restored_ids = torch.tensor(generated_original_ids, dtype=torch.long)
    return generated_original_ids, tokenizer.decode(restored_ids, skip_special_tokens=True)


@torch.inference_mode()
def run_stage_d_single_prompt(
    baseline_model,
    tokenizer,
    prompt: str,
    stage_model: StageBHiddenPermutationModel,
    perm_vocab: torch.Tensor,
    inv_perm_vocab: torch.Tensor,
    baseline_recorder: TraceRecorder,
    observed_recorder: TraceRecorder,
    hidden_transform: HiddenTransform,
    trace_layers: Iterable[int],
    max_new_tokens: int = 8,
) -> StageDRunResult:
    del inv_perm_vocab
    baseline_recorder.clear()
    observed_recorder.clear()

    encoded = tokenizer(
        format_chat_prompt(tokenizer, prompt),
        return_tensors="pt",
    )
    baseline_logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)
    mapped_input_ids = map_input_ids(encoded["input_ids"], perm_vocab)
    observed_logits_perm = stage_model(
        input_ids=mapped_input_ids,
        attention_mask=encoded.get("attention_mask"),
    ).logits.detach().cpu().to(torch.float32)

    metrics = summarize_stage_d_metrics(
        baseline_recorder=baseline_recorder,
        observed_recorder=observed_recorder,
        baseline_logits=baseline_logits,
        observed_logits_perm=observed_logits_perm,
        perm_vocab=perm_vocab,
        hidden_transform=hidden_transform,
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

    return StageDRunResult(
        prompt=prompt,
        mapped_input_ids=mapped_input_ids[0].tolist(),
        metrics=metrics | {
            "baseline_generated_text": baseline_generated_text,
            "restored_generated_text": observed_generated_text,
        },
    )


def aggregate_stage_d_results(results: list[StageDRunResult]) -> dict[str, float]:
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
