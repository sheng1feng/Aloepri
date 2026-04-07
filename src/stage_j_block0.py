from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from src.key_manager import ordinary_token_ids
from src.keymat_embed_head import add_embed_noise, add_head_noise
from src.stage_b import TraceRecorder, prepare_stage_a_model
from src.stage_i_square import SquareMonomialTransform, build_square_monomial_transform


def _apply_square_matrix_to_last_dim(hidden: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return torch.matmul(hidden, matrix.to(device=hidden.device, dtype=hidden.dtype))


def permute_rmsnorm_weight_for_square(
    weight: torch.Tensor,
    transform: SquareMonomialTransform,
) -> torch.Tensor:
    return weight.index_select(0, transform.inv_perm.to(weight.device))


def adapt_input_linear_weight_for_square(
    weight: torch.Tensor,
    transform: SquareMonomialTransform,
) -> torch.Tensor:
    return weight @ transform.key(dtype=weight.dtype).to(weight.device)


def adapt_output_linear_weight_for_square(
    weight: torch.Tensor,
    transform: SquareMonomialTransform,
) -> torch.Tensor:
    return transform.key(dtype=weight.dtype).T.to(weight.device) @ weight


def adapt_output_bias_for_square(
    bias: torch.Tensor | None,
    transform: SquareMonomialTransform,
) -> torch.Tensor | None:
    if bias is None:
        return None
    return torch.matmul(bias, transform.key(dtype=bias.dtype).to(bias.device))


def obfuscate_embedding_with_square_transform_stage_a(
    embed_weight: torch.Tensor,
    transform: SquareMonomialTransform,
    *,
    alpha_e: float,
    seed: int,
    movable_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    noisy = add_embed_noise(embed_weight, alpha_e=alpha_e, seed=seed, movable_ids=movable_ids)
    return noisy @ transform.key(dtype=noisy.dtype)


def obfuscate_head_with_square_transform_stage_a(
    head_weight: torch.Tensor,
    transform: SquareMonomialTransform,
    *,
    alpha_h: float = 0.0,
    seed: int = 0,
    movable_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    noisy = add_head_noise(head_weight, alpha_h=alpha_h, seed=seed, movable_ids=movable_ids)
    return noisy @ transform.inverse(dtype=noisy.dtype).T


class SquareDecoderLayerHandoff(nn.Module):
    def __init__(
        self,
        layer_module: nn.Module,
        transform: SquareMonomialTransform,
        layer_idx: int,
        recorder: TraceRecorder | None = None,
    ) -> None:
        super().__init__()
        self.layer_module = layer_module
        self.transform = transform
        self.layer_idx = layer_idx
        self.recorder = recorder
        self.inverse_matrix = transform.inverse(dtype=torch.float32)
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
        base_hidden = _apply_square_matrix_to_last_dim(hidden_states, self.inverse_matrix)
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


def attach_stage_j_block0_hooks(
    model,
    recorder: TraceRecorder,
    transform: SquareMonomialTransform | None,
    *,
    trace_layers: list[int],
    capture_embed_output: bool = True,
):
    handles = []
    inverse_matrix = None if transform is None else transform.inverse(dtype=torch.float32)

    def restore(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(torch.float32)
        if inverse_matrix is None:
            return tensor
        return _apply_square_matrix_to_last_dim(tensor, inverse_matrix)

    if capture_embed_output:
        handles.append(
            model.model.embed_tokens.register_forward_hook(
                lambda _, __, output: recorder.record("embed_out", restore(output))
            )
        )

    for layer_idx in trace_layers:
        layer = model.model.layers[layer_idx]
        prefix = f"layer_{layer_idx}"
        handles.append(
            layer.register_forward_pre_hook(
                lambda _, inputs, name=f"{prefix}_input": recorder.record(name, restore(inputs[0]))
            )
        )
        handles.append(
            layer.input_layernorm.register_forward_hook(
                lambda _, __, output, name=f"{prefix}_input_norm_out": recorder.record(name, restore(output))
            )
        )
        handles.append(
            layer.self_attn.q_proj.register_forward_hook(
                lambda _, __, output, name=f"{prefix}_q_proj_out": recorder.record(name, output)
            )
        )
        handles.append(
            layer.self_attn.k_proj.register_forward_hook(
                lambda _, __, output, name=f"{prefix}_k_proj_out": recorder.record(name, output)
            )
        )
        handles.append(
            layer.self_attn.v_proj.register_forward_hook(
                lambda _, __, output, name=f"{prefix}_v_proj_out": recorder.record(name, output)
            )
        )
        handles.append(
            layer.self_attn.o_proj.register_forward_hook(
                lambda _, __, output, name=f"{prefix}_attn_out": recorder.record(name, restore(output))
            )
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                lambda _, __, output, name=f"{prefix}_post_attn_norm_out": recorder.record(name, restore(output))
            )
        )
        handles.append(
            layer.mlp.register_forward_hook(
                lambda _, __, output, name=f"{prefix}_mlp_out": recorder.record(name, restore(output))
            )
        )
        handles.append(
            layer.register_forward_hook(
                lambda _, __, output, name=f"{prefix}_block_out": recorder.record(name, restore(output))
            )
        )

    def cleanup() -> None:
        for handle in handles:
            handle.remove()

    return cleanup


def _adapt_square_layer_inplace(
    layer,
    transform: SquareMonomialTransform,
) -> None:
    with torch.no_grad():
        layer.input_layernorm.weight.copy_(
            permute_rmsnorm_weight_for_square(layer.input_layernorm.weight.detach(), transform).to(layer.input_layernorm.weight.dtype)
        )
        layer.post_attention_layernorm.weight.copy_(
            permute_rmsnorm_weight_for_square(layer.post_attention_layernorm.weight.detach(), transform).to(layer.post_attention_layernorm.weight.dtype)
        )

        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            proj = getattr(layer.self_attn, proj_name)
            proj.weight.copy_(adapt_input_linear_weight_for_square(proj.weight.detach(), transform).to(proj.weight.dtype))

        layer.self_attn.o_proj.weight.copy_(
            adapt_output_linear_weight_for_square(layer.self_attn.o_proj.weight.detach(), transform).to(layer.self_attn.o_proj.weight.dtype)
        )
        if layer.self_attn.o_proj.bias is not None:
            layer.self_attn.o_proj.bias.copy_(
                adapt_output_bias_for_square(layer.self_attn.o_proj.bias.detach(), transform).to(layer.self_attn.o_proj.bias.dtype)
            )

        for proj_name in ["gate_proj", "up_proj"]:
            proj = getattr(layer.mlp, proj_name)
            proj.weight.copy_(adapt_input_linear_weight_for_square(proj.weight.detach(), transform).to(proj.weight.dtype))

        layer.mlp.down_proj.weight.copy_(
            adapt_output_linear_weight_for_square(layer.mlp.down_proj.weight.detach(), transform).to(layer.mlp.down_proj.weight.dtype)
        )
        if layer.mlp.down_proj.bias is not None:
            layer.mlp.down_proj.bias.copy_(
                adapt_output_bias_for_square(layer.mlp.down_proj.bias.detach(), transform).to(layer.mlp.down_proj.bias.dtype)
            )


def build_stage_j_square_model(
    baseline_model,
    tokenizer,
    *,
    adapted_layers: list[int],
    seed: int,
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
    global_scale: float = 1.0,
    recorder: TraceRecorder | None = None,
):
    if global_scale != 1.0:
        raise ValueError("Stage-J block0 prototype currently only supports global_scale=1.0")

    stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, seed=seed)
    stage_model = deepcopy(stage_a_model)
    transform = build_square_monomial_transform(
        hidden_size=baseline_model.config.hidden_size,
        seed=seed + 5000,
        global_scale=global_scale,
    )
    movable_ids = ordinary_token_ids(tokenizer)
    model_device = next(stage_model.parameters()).device
    original_head_weight = stage_model.lm_head.weight.detach().cpu().to(torch.float32).clone()
    original_head_bias = None
    if getattr(stage_model.lm_head, "bias", None) is not None:
        original_head_bias = stage_model.lm_head.bias.detach().cpu().to(torch.float32).clone()

    with torch.no_grad():
        embed_weight = stage_model.model.embed_tokens.weight.detach().cpu().to(torch.float32)
        embed_obf = obfuscate_embedding_with_square_transform_stage_a(
            embed_weight,
            transform,
            alpha_e=alpha_e,
            seed=seed + 101,
            movable_ids=movable_ids,
        )
        stage_model.model.embed_tokens.weight.copy_(embed_obf.to(stage_model.model.embed_tokens.weight.dtype))
        for layer_idx in adapted_layers:
            _adapt_square_layer_inplace(stage_model.model.layers[layer_idx], transform)

    handoff_layer = None if len(adapted_layers) == baseline_model.config.num_hidden_layers else max(adapted_layers) + 1
    if stage_model.model.embed_tokens.weight.data_ptr() == stage_model.lm_head.weight.data_ptr():
        untied_head = nn.Linear(
            stage_model.lm_head.in_features,
            stage_model.lm_head.out_features,
            bias=original_head_bias is not None,
        ).to(device=model_device, dtype=stage_model.lm_head.weight.dtype)
        with torch.no_grad():
            head_weight = original_head_weight
            if handoff_layer is None:
                head_weight = obfuscate_head_with_square_transform_stage_a(
                    head_weight,
                    transform,
                    alpha_h=alpha_h,
                    seed=seed + 202,
                    movable_ids=movable_ids,
                )
            untied_head.weight.copy_(head_weight.to(untied_head.weight.dtype))
            if original_head_bias is not None and untied_head.bias is not None:
                untied_head.bias.copy_(original_head_bias.to(untied_head.bias.dtype))
        stage_model.lm_head = untied_head
        if hasattr(stage_model.config, "tie_word_embeddings"):
            stage_model.config.tie_word_embeddings = False

    if handoff_layer is not None and handoff_layer < baseline_model.config.num_hidden_layers:
        original_handoff = stage_model.model.layers[handoff_layer]
        stage_model.model.layers[handoff_layer] = SquareDecoderLayerHandoff(
            layer_module=original_handoff,
            transform=transform,
            layer_idx=handoff_layer,
            recorder=recorder,
        )
    elif hasattr(stage_model.model, "norm") and stage_model.model.norm is not None:
        with torch.no_grad():
            stage_model.model.norm.weight.copy_(
                permute_rmsnorm_weight_for_square(stage_model.model.norm.weight.detach(), transform).to(stage_model.model.norm.weight.dtype)
            )

    stage_model.to(device=model_device)
    stage_model.eval()
    return stage_model, perm_vocab, inv_perm_vocab, transform


def build_stage_j_block0_model(
    baseline_model,
    tokenizer,
    *,
    seed: int,
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
    global_scale: float = 1.0,
    recorder: TraceRecorder | None = None,
):
    return build_stage_j_square_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        adapted_layers=[0],
        seed=seed,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
        global_scale=global_scale,
        recorder=recorder,
    )
