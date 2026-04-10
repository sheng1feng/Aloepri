from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_MAX_NEW_TOKENS, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.evaluator import max_abs_error, mean_abs_error
from src.key_manager import ordinary_token_ids
from src.model_loader import load_model_and_tokenizer, set_global_seed, tokenize_prompt
from src.stage_i_vllm import export_stage_i_vllm_checkpoint, summarize_token_partitions
from src.stage_j_block0 import (
    SquareDecoderLayerHandoff,
    _adapt_square_layer_inplace,
    obfuscate_embedding_with_square_transform_stage_a,
    obfuscate_head_with_square_transform_stage_a,
    permute_rmsnorm_weight_for_square,
)
from src.stage_b import prepare_stage_a_model
from src.stage_i_square import build_square_monomial_transform
from src.transforms import map_input_ids, restore_logits


GATE6_SENSITIVE_TERMS = [
    "privacy",
    "security",
    "token",
    "inference",
    "server",
    "client",
    "model",
    "prompt",
    "response",
    "obfuscation",
    "hidden state",
    "weights",
]


@dataclass(frozen=True)
class Gate6Case:
    name: str
    export_dir: str
    alpha_e: float
    alpha_h: float
    targeted_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_gate6_cases() -> list[Gate6Case]:
    return [
        Gate6Case("targeted_mild", "artifacts/stage_j_gate6_targeted_mild", alpha_e=0.2, alpha_h=0.1),
        Gate6Case("targeted_strong", "artifacts/stage_j_gate6_targeted_strong", alpha_e=0.5, alpha_h=0.2),
        Gate6Case("targeted_extreme", "artifacts/stage_j_gate6_targeted_extreme", alpha_e=5.0, alpha_h=2.0),
    ]


def security_sensitive_plain_ids(tokenizer) -> torch.Tensor:
    sensitive: set[int] = set()
    for term in GATE6_SENSITIVE_TERMS:
        ids = tokenizer(term, add_special_tokens=False)["input_ids"]
        sensitive.update(int(item) for item in ids if item < tokenizer.vocab_size)
    movable = set(int(item) for item in ordinary_token_ids(tokenizer).tolist())
    return torch.tensor(sorted(item for item in sensitive if item in movable), dtype=torch.long)


@torch.inference_mode()
def _greedy_generate_plain(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    current_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(current_ids)
        logits = model(input_ids=current_ids, attention_mask=attention_mask).logits.detach()
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        current_ids = torch.cat([current_ids, next_token.to(current_ids.device)], dim=1)
    return current_ids[:, input_ids.shape[1] :]


@torch.inference_mode()
def _greedy_generate_stage(model, input_ids: torch.Tensor, perm_vocab: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    current_ids = map_input_ids(input_ids.clone(), perm_vocab)
    generated_plain_tokens = []
    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(current_ids)
        logits_perm = model(input_ids=current_ids, attention_mask=attention_mask).logits.detach().cpu().to(torch.float32)
        restored = restore_logits(logits_perm, perm_vocab)
        next_plain = torch.argmax(restored[:, -1, :], dim=-1, keepdim=True)
        generated_plain_tokens.append(next_plain)
        next_stage = map_input_ids(next_plain, perm_vocab).to(current_ids.device)
        current_ids = torch.cat([current_ids, next_stage], dim=1)
    return torch.cat(generated_plain_tokens, dim=1)


def build_stage_j_targeted_sensitive_model(
    *,
    baseline_model,
    tokenizer,
    seed: int,
    alpha_e: float,
    alpha_h: float,
    noise_token_ids: torch.Tensor,
) -> tuple[Any, torch.Tensor, torch.Tensor, Any]:
    adapted_layers = list(range(baseline_model.config.num_hidden_layers))
    stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, seed=seed)
    from copy import deepcopy

    stage_model = deepcopy(stage_a_model)
    transform = build_square_monomial_transform(
        hidden_size=baseline_model.config.hidden_size,
        seed=seed + 5000,
        global_scale=1.0,
    )
    model_device = next(stage_model.parameters()).device
    original_head_weight = stage_model.lm_head.weight.detach().cpu().to(torch.float32).clone()
    original_head_bias = None
    if getattr(stage_model.lm_head, "bias", None) is not None:
        original_head_bias = stage_model.lm_head.bias.detach().cpu().to(torch.float32).clone()

    stage_noise_ids = perm_vocab[noise_token_ids]

    with torch.no_grad():
        embed_weight = stage_model.model.embed_tokens.weight.detach().cpu().to(torch.float32)
        embed_obf = obfuscate_embedding_with_square_transform_stage_a(
            embed_weight,
            transform,
            alpha_e=alpha_e,
            seed=seed + 101,
            movable_ids=stage_noise_ids,
        )
        stage_model.model.embed_tokens.weight.copy_(embed_obf.to(stage_model.model.embed_tokens.weight.dtype))
        for layer_idx in adapted_layers:
            _adapt_square_layer_inplace(stage_model.model.layers[layer_idx], transform)

    handoff_layer = None
    if stage_model.model.embed_tokens.weight.data_ptr() == stage_model.lm_head.weight.data_ptr():
        untied_head = torch.nn.Linear(
            stage_model.lm_head.in_features,
            stage_model.lm_head.out_features,
            bias=original_head_bias is not None,
        ).to(device=model_device, dtype=stage_model.lm_head.weight.dtype)
        with torch.no_grad():
            head_weight = obfuscate_head_with_square_transform_stage_a(
                original_head_weight,
                transform,
                alpha_h=alpha_h,
                seed=seed + 202,
                movable_ids=stage_noise_ids,
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
            recorder=None,
        )
    elif hasattr(stage_model.model, "norm") and stage_model.model.norm is not None:
        with torch.no_grad():
            stage_model.model.norm.weight.copy_(
                permute_rmsnorm_weight_for_square(stage_model.model.norm.weight.detach(), transform).to(stage_model.model.norm.weight.dtype)
            )

    stage_model.to(device=model_device)
    stage_model.eval()
    return stage_model, perm_vocab, inv_perm_vocab, transform


def evaluate_stage_j_accuracy(
    *,
    baseline_model,
    stage_model,
    tokenizer,
    perm_vocab: torch.Tensor,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    prompt_results: list[dict[str, Any]] = []
    for prompt in DEFAULT_PROMPTS:
        encoded = tokenize_prompt(tokenizer, prompt, device="cpu")
        baseline_logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)
        mapped_ids = map_input_ids(encoded["input_ids"], perm_vocab)
        observed_logits_perm = stage_model(input_ids=mapped_ids, attention_mask=encoded["attention_mask"]).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(observed_logits_perm, perm_vocab)
        baseline_generated_ids = _greedy_generate_plain(baseline_model, encoded["input_ids"], max_new_tokens)[0].cpu()
        stage_generated_ids = _greedy_generate_stage(stage_model, encoded["input_ids"], perm_vocab, max_new_tokens)[0].cpu()
        prompt_results.append(
            {
                "prompt": prompt,
                "final_logits_restored_max_abs_error": max_abs_error(baseline_logits, restored_logits),
                "final_logits_restored_mean_abs_error": mean_abs_error(baseline_logits, restored_logits),
                "greedy_first_token_match": bool(baseline_generated_ids[0].item() == stage_generated_ids[0].item()),
                "generated_ids_exact_match": baseline_generated_ids.tolist() == stage_generated_ids.tolist(),
                "generated_text_exact_match": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True) == tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
            }
        )
    prompt_count = len(prompt_results)
    return {
        "prompt_count": prompt_count,
        "avg_final_logits_restored_max_abs_error": float(sum(item["final_logits_restored_max_abs_error"] for item in prompt_results) / prompt_count),
        "generated_ids_exact_match_rate": float(sum(1 for item in prompt_results if item["generated_ids_exact_match"]) / prompt_count),
        "generated_text_exact_match_rate": float(sum(1 for item in prompt_results if item["generated_text_exact_match"]) / prompt_count),
    }


def ensure_gate6_artifact(case: Gate6Case) -> dict[str, Any]:
    export_dir = Path(case.export_dir)
    if (export_dir / "server" / "model.safetensors").exists() and (export_dir / "client" / "client_secret.pt").exists():
        metadata = json.loads((export_dir / "stage_i_metadata.json").read_text(encoding="utf-8"))
        accuracy_path = export_dir / "gate6_accuracy_summary.json"
        accuracy = json.loads(accuracy_path.read_text(encoding="utf-8")) if accuracy_path.exists() else {}
        return {"export_dir": export_dir, "metadata": metadata, "accuracy": accuracy}

    set_global_seed(DEFAULT_SEED)
    tokenizer, baseline_model = load_model_and_tokenizer(DEFAULT_MODEL_DIR, device="cpu", dtype="float32")
    sensitive_ids = security_sensitive_plain_ids(tokenizer)
    stage_model, perm_vocab, inv_perm_vocab, transform = build_stage_j_targeted_sensitive_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        seed=DEFAULT_SEED,
        alpha_e=case.alpha_e,
        alpha_h=case.alpha_h,
        noise_token_ids=sensitive_ids,
    )
    accuracy = evaluate_stage_j_accuracy(
        baseline_model=baseline_model,
        stage_model=stage_model,
        tokenizer=tokenizer,
        perm_vocab=perm_vocab,
    )
    metadata = {
        "stage": "J",
        "variant": "gate6_targeted_sensitive_square_transform",
        "model_dir": DEFAULT_MODEL_DIR,
        "seed": DEFAULT_SEED,
        "dtype": "float32",
        "alpha_e": case.alpha_e,
        "alpha_h": case.alpha_h,
        "global_scale": 1.0,
        "sensitive_terms": GATE6_SENSITIVE_TERMS,
        "sensitive_token_ids": sensitive_ids.tolist(),
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
    export_stage_i_vllm_checkpoint(
        export_dir,
        tokenizer=tokenizer,
        stage_a_model=stage_model,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        metadata=metadata,
    )
    (export_dir / "gate6_accuracy_summary.json").write_text(json.dumps(accuracy, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"export_dir": export_dir, "metadata": metadata, "accuracy": accuracy}
