from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward

from src.evaluator import max_abs_error, mean_abs_error
from src.hidden_keys import HiddenTransform, invert_hidden_transform
from src.key_manager import generate_vocab_permutation, invert_permutation, ordinary_token_ids
from src.obfuscate_embed_head import build_vocab_permuted_model
from src.transforms import apply_hidden_transform, apply_inverse_hidden_transform, map_input_ids, restore_logits


class TraceRecorder:
    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}

    def clear(self) -> None:
        self.tensors.clear()

    def record(self, name: str, tensor: torch.Tensor | tuple[torch.Tensor, ...] | None) -> None:
        if tensor is None:
            return
        if isinstance(tensor, tuple):
            if not tensor:
                return
            tensor = tensor[0]
        self.tensors[name] = tensor.detach().cpu().to(torch.float32)


class TracingQwen2Attention(nn.Module):
    def __init__(
        self,
        attention_module: nn.Module,
        recorder: TraceRecorder,
        mode: str = "plain",
        hidden_transform: HiddenTransform | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"plain", "wrapper"}:
            raise ValueError(f"Unsupported attention tracing mode: {mode}")
        self.mode = mode
        self.recorder = recorder
        self.hidden_transform = hidden_transform

        self.config = attention_module.config
        self.layer_idx = attention_module.layer_idx
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
        self.recorder.record("layer_0_q_proj_out", query_linear)
        self.recorder.record("layer_0_k_proj_out", key_linear)
        self.recorder.record("layer_0_v_proj_out", value_linear)

        query_states = query_linear.view(hidden_shape).transpose(1, 2)
        key_states = key_linear.view(hidden_shape).transpose(1, 2)
        value_states = value_linear.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
        self.recorder.record("layer_0_attn_out", attn_output)
        return attn_output, attn_weights


def attach_stage_b_hooks(
    model,
    recorder: TraceRecorder,
    attention_mode: str,
    hidden_transform: HiddenTransform | None = None,
    capture_embed_output: bool = False,
):
    handles = []
    if capture_embed_output:
        handles.append(
            model.model.embed_tokens.register_forward_hook(
                lambda _, __, output: recorder.record("embed_out", output)
            )
        )

    layer0 = model.model.layers[0]
    original_attention = layer0.self_attn
    layer0.self_attn = TracingQwen2Attention(
        attention_module=original_attention,
        recorder=recorder,
        mode=attention_mode,
        hidden_transform=hidden_transform,
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

    return cleanup


class StageBHiddenPermutationModel(nn.Module):
    def __init__(
        self,
        stage_a_model,
        hidden_transform: HiddenTransform,
        recorder: TraceRecorder | None = None,
    ) -> None:
        super().__init__()
        self.stage_a_model = stage_a_model
        self.hidden_transform = hidden_transform
        self.recorder = recorder

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.stage_a_model.get_input_embeddings()(input_ids)

        if self.recorder is not None:
            self.recorder.record("embed_out", inputs_embeds)

        obfuscated_embeds = apply_hidden_transform(inputs_embeds, self.hidden_transform)
        if self.recorder is not None:
            self.recorder.record("embed_out_obf", obfuscated_embeds)

        outputs = self.stage_a_model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=obfuscated_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states_obf = outputs.last_hidden_state
        if self.recorder is not None:
            self.recorder.record("final_hidden_obf", hidden_states_obf)
        hidden_states_for_head = apply_inverse_hidden_transform(hidden_states_obf, self.hidden_transform)
        if self.recorder is not None:
            self.recorder.record("final_hidden_restored", hidden_states_for_head)
        logits = self.stage_a_model.get_output_embeddings()(hidden_states_for_head)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


def prepare_stage_a_model(model, tokenizer, seed: int):
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    movable_ids = ordinary_token_ids(tokenizer)
    perm_vocab = generate_vocab_permutation(
        vocab_size=model_vocab_size,
        seed=seed,
        movable_ids=movable_ids,
    )
    inv_perm_vocab = invert_permutation(perm_vocab)
    stage_a_model = build_vocab_permuted_model(model, perm_vocab, inv_perm_vocab)
    return stage_a_model, perm_vocab, inv_perm_vocab


def fuse_block0_attention_hidden_transform(
    stage_a_model,
    hidden_transform: HiddenTransform,
):
    fused_model = deepcopy(stage_a_model)
    layer0_attn = fused_model.model.layers[0].self_attn
    transform_matrix = hidden_transform.matrix(
        device=layer0_attn.q_proj.weight.device,
        dtype=layer0_attn.q_proj.weight.dtype,
    )
    inverse_matrix = invert_hidden_transform(hidden_transform).matrix(
        device=layer0_attn.q_proj.weight.device,
        dtype=layer0_attn.q_proj.weight.dtype,
    )

    with torch.no_grad():
        layer0_attn.q_proj.weight.copy_(layer0_attn.q_proj.weight @ inverse_matrix.T)
        layer0_attn.k_proj.weight.copy_(layer0_attn.k_proj.weight @ inverse_matrix.T)
        layer0_attn.v_proj.weight.copy_(layer0_attn.v_proj.weight @ inverse_matrix.T)
        layer0_attn.o_proj.weight.copy_(transform_matrix.T @ layer0_attn.o_proj.weight)
    return fused_model


@dataclass
class StageBRunResult:
    prompt: str
    mapped_input_ids: list[int]
    metrics: dict[str, float | bool]


def summarize_stage_b_metrics(
    baseline_recorder: TraceRecorder,
    observed_recorder: TraceRecorder,
    baseline_logits: torch.Tensor,
    observed_logits_perm: torch.Tensor,
    perm_vocab: torch.Tensor,
    hidden_transform: HiddenTransform,
) -> dict[str, float | bool]:
    restored_logits = restore_logits(observed_logits_perm, perm_vocab)

    metrics: dict[str, float | bool] = {
        "embed_out_restored_max_abs_error": max_abs_error(
            baseline_recorder.tensors["embed_out"],
            apply_inverse_hidden_transform(observed_recorder.tensors["embed_out_obf"], hidden_transform),
        ),
        "embed_out_restored_mean_abs_error": mean_abs_error(
            baseline_recorder.tensors["embed_out"],
            apply_inverse_hidden_transform(observed_recorder.tensors["embed_out_obf"], hidden_transform),
        ),
        "layer_0_input_restored_max_abs_error": max_abs_error(
            baseline_recorder.tensors["layer_0_input"],
            apply_inverse_hidden_transform(observed_recorder.tensors["layer_0_input"], hidden_transform),
        ),
        "layer_0_input_restored_mean_abs_error": mean_abs_error(
            baseline_recorder.tensors["layer_0_input"],
            apply_inverse_hidden_transform(observed_recorder.tensors["layer_0_input"], hidden_transform),
        ),
        "layer_0_q_proj_out_max_abs_error": max_abs_error(
            baseline_recorder.tensors["layer_0_q_proj_out"],
            observed_recorder.tensors["layer_0_q_proj_out"],
        ),
        "layer_0_q_proj_out_mean_abs_error": mean_abs_error(
            baseline_recorder.tensors["layer_0_q_proj_out"],
            observed_recorder.tensors["layer_0_q_proj_out"],
        ),
        "layer_0_k_proj_out_max_abs_error": max_abs_error(
            baseline_recorder.tensors["layer_0_k_proj_out"],
            observed_recorder.tensors["layer_0_k_proj_out"],
        ),
        "layer_0_k_proj_out_mean_abs_error": mean_abs_error(
            baseline_recorder.tensors["layer_0_k_proj_out"],
            observed_recorder.tensors["layer_0_k_proj_out"],
        ),
        "layer_0_v_proj_out_max_abs_error": max_abs_error(
            baseline_recorder.tensors["layer_0_v_proj_out"],
            observed_recorder.tensors["layer_0_v_proj_out"],
        ),
        "layer_0_v_proj_out_mean_abs_error": mean_abs_error(
            baseline_recorder.tensors["layer_0_v_proj_out"],
            observed_recorder.tensors["layer_0_v_proj_out"],
        ),
        "layer_0_attn_out_restored_max_abs_error": max_abs_error(
            baseline_recorder.tensors["layer_0_attn_out"],
            apply_inverse_hidden_transform(observed_recorder.tensors["layer_0_attn_out"], hidden_transform),
        ),
        "layer_0_attn_out_restored_mean_abs_error": mean_abs_error(
            baseline_recorder.tensors["layer_0_attn_out"],
            apply_inverse_hidden_transform(observed_recorder.tensors["layer_0_attn_out"], hidden_transform),
        ),
        "layer_0_block_out_restored_max_abs_error": max_abs_error(
            baseline_recorder.tensors["layer_0_block_out"],
            apply_inverse_hidden_transform(observed_recorder.tensors["layer_0_block_out"], hidden_transform),
        ),
        "layer_0_block_out_restored_mean_abs_error": mean_abs_error(
            baseline_recorder.tensors["layer_0_block_out"],
            apply_inverse_hidden_transform(observed_recorder.tensors["layer_0_block_out"], hidden_transform),
        ),
        "final_logits_restored_max_abs_error": max_abs_error(
            baseline_logits,
            restored_logits,
        ),
        "final_logits_restored_mean_abs_error": mean_abs_error(
            baseline_logits,
            restored_logits,
        ),
    }

    optional_hidden_names = [
        "layer_0_input_norm_out",
        "layer_0_post_attn_norm_out",
        "layer_0_mlp_out",
    ]
    for name in optional_hidden_names:
        if name in baseline_recorder.tensors and name in observed_recorder.tensors:
            restored = apply_inverse_hidden_transform(observed_recorder.tensors[name], hidden_transform)
            metrics[f"{name}_restored_max_abs_error"] = max_abs_error(
                baseline_recorder.tensors[name],
                restored,
            )
            metrics[f"{name}_restored_mean_abs_error"] = mean_abs_error(
                baseline_recorder.tensors[name],
                restored,
            )
    return metrics


@torch.inference_mode()
def run_stage_b_single_prompt(
    baseline_model,
    tokenizer,
    prompt: str,
    stage_b_model: StageBHiddenPermutationModel,
    perm_vocab: torch.Tensor,
    baseline_recorder: TraceRecorder,
    observed_recorder: TraceRecorder,
    hidden_transform: HiddenTransform,
) -> StageBRunResult:
    baseline_recorder.clear()
    observed_recorder.clear()

    encoded = tokenizer(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        ),
        return_tensors="pt",
    )
    baseline_logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)
    mapped_input_ids = map_input_ids(encoded["input_ids"], perm_vocab)
    stage_b_logits = stage_b_model(
        input_ids=mapped_input_ids,
        attention_mask=encoded.get("attention_mask"),
    ).logits.detach().cpu().to(torch.float32)

    metrics = summarize_stage_b_metrics(
        baseline_recorder=baseline_recorder,
        observed_recorder=observed_recorder,
        baseline_logits=baseline_logits,
        observed_logits_perm=stage_b_logits,
        perm_vocab=perm_vocab,
        hidden_transform=hidden_transform,
    )
    return StageBRunResult(
        prompt=prompt,
        mapped_input_ids=mapped_input_ids[0].tolist(),
        metrics=metrics,
    )


def aggregate_stage_b_results(results: list[StageBRunResult]) -> dict[str, float]:
    numeric_keys = [
        key
        for key, value in results[0].metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    summary: dict[str, float] = {}
    for key in numeric_keys:
        summary[f"avg_{key}"] = float(sum(float(result.metrics[key]) for result in results) / len(results))
    return summary
