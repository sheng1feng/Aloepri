from __future__ import annotations

from pathlib import Path
from typing import Any

from src.stage_i_vllm import export_stage_i_vllm_checkpoint, summarize_token_partitions
from src.stage_j_block0 import build_stage_j_square_model


def build_standard_shape_full_bundle(
    *,
    baseline_model,
    tokenizer: Any,
    seed: int,
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
    global_scale: float = 1.0,
) -> dict[str, Any]:
    adapted_layers = list(range(baseline_model.config.num_hidden_layers))
    stage_model, perm_vocab, inv_perm_vocab, transform = build_stage_j_square_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        adapted_layers=adapted_layers,
        seed=seed,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
        global_scale=global_scale,
        recorder=None,
    )
    metadata = {
        "variant": "standard_shape_full_layer",
        "seed": seed,
        "alpha_e": alpha_e,
        "alpha_h": alpha_h,
        "global_scale": global_scale,
        "adapted_layers": adapted_layers,
        "square_transform": {
            "perm": transform.perm.tolist(),
            "inv_perm": transform.inv_perm.tolist(),
            "signs": transform.signs.tolist(),
            "global_scale": transform.global_scale,
        },
        **summarize_token_partitions(
            tokenizer=tokenizer,
            model_vocab_size=stage_model.get_input_embeddings().weight.shape[0],
            perm_vocab=perm_vocab,
        ),
    }
    return {
        "model": stage_model,
        "perm_vocab": perm_vocab,
        "inv_perm_vocab": inv_perm_vocab,
        "transform": transform,
        "metadata": metadata,
    }


def export_standard_shape_full_checkpoint(
    export_dir: str | Path,
    *,
    tokenizer: Any,
    baseline_model,
    seed: int,
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
    global_scale: float = 1.0,
) -> dict[str, Any]:
    bundle = build_standard_shape_full_bundle(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        seed=seed,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
        global_scale=global_scale,
    )
    paths = export_stage_i_vllm_checkpoint(
        export_dir,
        tokenizer=tokenizer,
        stage_a_model=bundle["model"],
        perm_vocab=bundle["perm_vocab"],
        inv_perm_vocab=bundle["inv_perm_vocab"],
        metadata=bundle["metadata"],
    )
    return {
        **bundle,
        **paths,
    }
