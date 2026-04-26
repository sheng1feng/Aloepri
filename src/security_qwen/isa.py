from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.key_manager import ordinary_token_ids
from src.model_loader import resolve_torch_dtype
from src.security_qwen.ima import (
    _evaluate_inversion_predictions,
    _fit_ridge_regressor,
    _predict_ridge,
    load_ima_embedding_sources,
)
from src.security_qwen.metrics import classify_risk_level
from src.security_qwen.schema import SecurityEvalTarget, build_security_eval_payload
from src.stage_h_artifact import load_stage_h_artifact
from src.stage_h_pretrained import load_stage_h_pretrained


@dataclass(frozen=True)
class ISABaselineConfig:
    baseline_model_dir: str = DEFAULT_MODEL_DIR
    seed: int = DEFAULT_SEED
    observable_type: str = "hidden_state"
    observable_layer: int = 23
    sequence_length: int = 8
    train_sequences: int = 64
    val_sequences: int = 16
    test_sequences: int = 16
    candidate_pool_size: int = 2048
    topk: int = 10
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ridge_alphas"] = list(self.ridge_alphas)
        return payload


def default_isa_gate3_targets() -> list[str]:
    return [
        "stage_a_standard",
        "stage_h_full_obfuscated",
        "stage_j_stable_reference",
        "stage_j_tiny_a",
        "stage_k_reference",
        "stage_k_default",
    ]


def build_isa_template(
    target: SecurityEvalTarget,
    *,
    observable_type: str = "hidden_state",
    observable_layer: str = "planned",
) -> dict:
    primary_metric = None
    return build_security_eval_payload(
        attack="isa",
        target=target,
        config={
            "phase": "phase0",
            "dataset_name": "phase0_inversion_public_corpus",
            "seed": 20260323,
            "observable_type": observable_type,
            "observable_layer": observable_layer,
            "deployment_only": True,
        },
        metrics={
            "observable_type": observable_type,
            "observable_layer": observable_layer,
            "intermediate_top1_recovery_rate": primary_metric,
            "token_top10_recovery_rate": None,
        },
        summary={
            "status": "planned",
            "primary_metric_name": "intermediate_top1_recovery_rate",
            "primary_metric_value": primary_metric,
            "risk_level": classify_risk_level(primary_metric),
            "notes": "Phase 0 template only; ISA deployment-observable implementation not started.",
        },
        artifacts={},
    )


def _collect_sensitive_plain_ids(tokenizer) -> torch.Tensor:
    sensitive: set[int] = set()
    special_ids = set(tokenizer.all_special_ids)
    for prompt in DEFAULT_PROMPTS:
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for token_id in input_ids:
            if token_id not in special_ids and token_id < tokenizer.vocab_size:
                sensitive.add(int(token_id))
    return torch.tensor(sorted(sensitive), dtype=torch.long)


def _unique_preserve_order(items: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _sample_plain_sequences(
    *,
    tokenizer,
    sequence_length: int,
    train_sequences: int,
    val_sequences: int,
    test_sequences: int,
    candidate_pool_size: int,
    seed: int,
) -> dict[str, Any]:
    movable_plain_ids = ordinary_token_ids(tokenizer).tolist()
    sensitive_plain_ids = [item for item in _collect_sensitive_plain_ids(tokenizer).tolist() if item in set(movable_plain_ids)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def make_sequences(count: int) -> torch.Tensor:
        flat = torch.tensor(movable_plain_ids, dtype=torch.long)[
            torch.randint(low=0, high=len(movable_plain_ids), size=(count * sequence_length,), generator=generator)
        ]
        return flat.view(count, sequence_length)

    train_plain = make_sequences(train_sequences)
    val_plain = make_sequences(val_sequences)
    test_plain = make_sequences(test_sequences)
    if sensitive_plain_ids:
        sensitive_tensor = torch.tensor(sensitive_plain_ids[: min(len(sensitive_plain_ids), sequence_length)], dtype=torch.long)
        test_plain[0, : sensitive_tensor.numel()] = sensitive_tensor

    candidate_plain = _unique_preserve_order(
        test_plain.reshape(-1).tolist()
        + torch.tensor(movable_plain_ids, dtype=torch.long)[
            torch.randperm(len(movable_plain_ids), generator=generator)[:candidate_pool_size]
        ].tolist()
    )[:candidate_pool_size]

    return {
        "train_plain_ids": train_plain,
        "val_plain_ids": val_plain,
        "test_plain_ids": test_plain,
        "candidate_plain_ids": torch.tensor(candidate_plain, dtype=torch.long),
        "sensitive_plain_ids": torch.tensor(sensitive_plain_ids, dtype=torch.long),
    }


def _flatten_hidden_state(hidden_state: torch.Tensor) -> torch.Tensor:
    return hidden_state.reshape(-1, hidden_state.shape[-1])


def _flatten_attention_score(attn: torch.Tensor) -> torch.Tensor:
    # attn: [batch, heads, seq, seq]
    batch, heads, q_len, k_len = attn.shape
    per_token = attn.permute(0, 2, 1, 3).reshape(batch * q_len, heads * k_len)
    return per_token


def _get_standard_observable(
    *,
    model,
    input_ids: torch.Tensor,
    observable_type: str,
    observable_layer: int,
) -> torch.Tensor:
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "use_cache": False,
        "return_dict": True,
    }
    if observable_type == "hidden_state":
        outputs = model(**kwargs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[observable_layer + 1]
        return _flatten_hidden_state(hidden_state.detach().cpu().to(torch.float32))
    if observable_type == "attention_score":
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        outputs = model(**kwargs, output_attentions=True)
        attn = outputs.attentions[observable_layer]
        return _flatten_attention_score(attn.detach().cpu().to(torch.float32))
    raise ValueError(f"Unsupported observable_type: {observable_type}")


def _get_stage_h_observable(
    *,
    model,
    input_ids: torch.Tensor,
    observable_type: str,
    observable_layer: int,
) -> torch.Tensor:
    layers = model.stage_a_model.model.layers
    captured: dict[str, torch.Tensor] = {}

    if observable_type == "hidden_state":
        target_module = layers[observable_layer]

        def hook_fn(_, __, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured["observable"] = tensor.detach().cpu().to(torch.float32)

        handle = target_module.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
        handle.remove()
        return _flatten_hidden_state(captured["observable"])

    if observable_type == "attention_score":
        target_module = layers[observable_layer].self_attn

        def hook_fn(_, __, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                captured["observable"] = output[1].detach().cpu().to(torch.float32)

        handle = target_module.register_forward_hook(hook_fn)
        if hasattr(model.stage_a_model.config, "_attn_implementation"):
            model.stage_a_model.config._attn_implementation = "eager"
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                output_attentions=True,
            )
        handle.remove()
        if "observable" not in captured:
            raise RuntimeError("Failed to capture Stage-H attention scores")
        return _flatten_attention_score(captured["observable"])

    raise ValueError(f"Unsupported observable_type: {observable_type}")


def load_isa_model_bundle(
    *,
    target_name: str,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    baseline_embed, _, aux = load_ima_embedding_sources(
        target_name=target_name,
        baseline_model_dir=baseline_model_dir,
    )
    resolved = aux["resolved_target"]
    perm_vocab = aux["perm_vocab"]

    manifest = {}
    manifest_path = resolved.get("manifest_path")
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    if resolved["stage"] == "H":
        bundle = load_stage_h_artifact(resolved["resolved_root_dir"])
        model = bundle["stage_model"].eval()
        tokenizer = bundle["tokenizer"]
        extractor = _get_stage_h_observable
    elif manifest.get("bootstrap_source") == "artifacts/stage_h_pretrained":
        server_dir = resolved["server_dir"]
        if server_dir is None:
            raise FileNotFoundError(f"Target {target_name} has no server_dir")
        bundle = load_stage_h_pretrained(server_dir)
        model = bundle["stage_model"].eval()
        tokenizer = bundle["tokenizer"]
        extractor = _get_stage_h_observable
    else:
        server_dir = resolved["server_dir"]
        if server_dir is None:
            raise FileNotFoundError(f"Target {target_name} has no server_dir")
        tokenizer = AutoTokenizer.from_pretrained(server_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            server_dir,
            trust_remote_code=True,
            torch_dtype=resolve_torch_dtype("auto"),
        ).eval()
        extractor = _get_standard_observable

    return {
        "baseline_embed": baseline_embed,
        "perm_vocab": perm_vocab,
        "tokenizer": tokenizer,
        "model": model,
        "extractor": extractor,
        "resolved_target": resolved,
    }


def _extract_observable_samples(
    *,
    model,
    extractor,
    perm_vocab: torch.Tensor,
    plain_sequences: torch.Tensor,
    observable_type: str,
    observable_layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    obfuscated_sequences = perm_vocab[plain_sequences]
    features = extractor(
        model=model,
        input_ids=obfuscated_sequences,
        observable_type=observable_type,
        observable_layer=observable_layer,
    )
    labels = plain_sequences.reshape(-1).cpu()
    return features, labels


def run_isa_baseline(
    *,
    target_name: str,
    observable_type: str = "hidden_state",
    observable_layer: int = 23,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
    seed: int = DEFAULT_SEED,
    sequence_length: int = 8,
    train_sequences: int = 64,
    val_sequences: int = 16,
    test_sequences: int = 16,
    candidate_pool_size: int = 2048,
    topk: int = 10,
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0),
) -> dict[str, Any]:
    start_time = time.perf_counter()
    bundle = load_isa_model_bundle(target_name=target_name, baseline_model_dir=baseline_model_dir)
    baseline_embed = bundle["baseline_embed"]
    perm_vocab = bundle["perm_vocab"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    extractor = bundle["extractor"]
    resolved = bundle["resolved_target"]

    splits = _sample_plain_sequences(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
    )
    train_plain = splits["train_plain_ids"]
    val_plain = splits["val_plain_ids"]
    test_plain = splits["test_plain_ids"]
    candidate_plain_ids = splits["candidate_plain_ids"]
    sensitive_plain_ids = splits["sensitive_plain_ids"]

    x_train, y_train_ids = _extract_observable_samples(
        model=model,
        extractor=extractor,
        perm_vocab=perm_vocab,
        plain_sequences=train_plain,
        observable_type=observable_type,
        observable_layer=observable_layer,
    )
    x_val, y_val_ids = _extract_observable_samples(
        model=model,
        extractor=extractor,
        perm_vocab=perm_vocab,
        plain_sequences=val_plain,
        observable_type=observable_type,
        observable_layer=observable_layer,
    )
    x_test, y_test_ids = _extract_observable_samples(
        model=model,
        extractor=extractor,
        perm_vocab=perm_vocab,
        plain_sequences=test_plain,
        observable_type=observable_type,
        observable_layer=observable_layer,
    )

    y_train = baseline_embed[y_train_ids]
    y_val = baseline_embed[y_val_ids]
    val_candidate_plain_ids = torch.unique(torch.cat([y_val_ids, candidate_plain_ids], dim=0), sorted=False)

    best_alpha = None
    best_model = None
    best_val_top1 = -1.0
    alpha_scores: list[dict[str, Any]] = []
    for alpha in ridge_alphas:
        model_ridge = _fit_ridge_regressor(x_train, y_train, ridge_alpha=float(alpha))
        val_pred = _predict_ridge(model_ridge, x_val)
        val_metrics = _evaluate_inversion_predictions(
            predicted_embeddings=val_pred,
            true_plain_ids=y_val_ids,
            candidate_plain_ids=val_candidate_plain_ids,
            baseline_embed=baseline_embed,
            topk=topk,
        )
        alpha_scores.append({"ridge_alpha": float(alpha), **val_metrics})
        if val_metrics["token_top1_recovery_rate"] > best_val_top1:
            best_alpha = float(alpha)
            best_model = model_ridge
            best_val_top1 = val_metrics["token_top1_recovery_rate"]

    if best_model is None or best_alpha is None:
        raise RuntimeError("ISA failed to select a best ridge model")

    test_pred = _predict_ridge(best_model, x_test)
    test_metrics = _evaluate_inversion_predictions(
        predicted_embeddings=test_pred,
        true_plain_ids=y_test_ids,
        candidate_plain_ids=candidate_plain_ids,
        baseline_embed=baseline_embed,
        topk=topk,
    )
    sensitive_mask = torch.isin(y_test_ids, sensitive_plain_ids) if sensitive_plain_ids.numel() > 0 else torch.zeros_like(y_test_ids, dtype=torch.bool)
    sensitive_rate = None
    if bool(sensitive_mask.any()):
        candidate_embeddings = baseline_embed[candidate_plain_ids]
        pred_norm = test_pred / test_pred.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cand_norm = candidate_embeddings / candidate_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
        score_matrix = pred_norm @ cand_norm.T
        top1_indices = torch.topk(score_matrix, k=1, dim=1).indices.squeeze(1)
        predicted_plain = candidate_plain_ids[top1_indices]
        sensitive_rate = float(predicted_plain[sensitive_mask].eq(y_test_ids[sensitive_mask]).to(torch.float32).mean().item())

    runtime_seconds = time.perf_counter() - start_time
    payload = build_security_eval_payload(
        attack="isa",
        target=SecurityEvalTarget(
            stage=resolved["stage"],
            artifact_dir=resolved["artifact_dir"],
            profile=resolved["profile"],
            model_family="qwen",
            variant=resolved["variant"],
        ),
        config={
            "phase": "gate3_minimal",
            **ISABaselineConfig(
                baseline_model_dir=baseline_model_dir,
                seed=seed,
                observable_type=observable_type,
                observable_layer=observable_layer,
                sequence_length=sequence_length,
                train_sequences=train_sequences,
                val_sequences=val_sequences,
                test_sequences=test_sequences,
                candidate_pool_size=int(candidate_plain_ids.numel()),
                topk=topk,
                ridge_alphas=tuple(float(item) for item in ridge_alphas),
            ).to_dict(),
            "dataset_name": "phase0_synthetic_sequence_proxy",
            "selected_ridge_alpha": best_alpha,
            "alpha_scores": alpha_scores,
            "deployment_only": True,
        },
        metrics={
            "observable_type": observable_type,
            "observable_layer": observable_layer,
            "intermediate_top1_recovery_rate": test_metrics["token_top1_recovery_rate"],
            "token_top1_recovery_rate": test_metrics["token_top1_recovery_rate"],
            "token_top10_recovery_rate": test_metrics["token_top10_recovery_rate"],
            "embedding_cosine_similarity": test_metrics["embedding_cosine_similarity"],
            "sensitive_token_recovery_rate": sensitive_rate,
            "train_sample_count": int(x_train.shape[0]),
            "val_sample_count": int(x_val.shape[0]),
            "test_sample_count": int(x_test.shape[0]),
            "feature_dim": int(x_train.shape[1]),
            "attack_runtime_seconds": runtime_seconds,
        },
        summary={
            "status": "completed_minimal_baseline",
            "primary_metric_name": "intermediate_top1_recovery_rate",
            "primary_metric_value": test_metrics["token_top1_recovery_rate"],
            "risk_level": classify_risk_level(test_metrics["token_top1_recovery_rate"]),
            "notes": "Gate-3 minimal ridge inversion baseline on deployment-visible observables.",
        },
        artifacts={
            "resolved_target": resolved,
            "predicted_plain_ids_sample": test_metrics["predicted_plain_ids_sample"],
        },
    )
    return payload


def build_isa_comparison_payload(
    *,
    result_payloads: list[dict[str, Any]],
    observable_type: str,
    baseline_target_name: str = "stage_a_standard",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline_row: dict[str, Any] | None = None
    for payload in result_payloads:
        target = payload["target"]
        metrics = payload["metrics"]
        summary = payload["summary"]
        row = {
            "stage": target.get("stage"),
            "profile": target.get("profile"),
            "artifact_dir": target.get("artifact_dir"),
            "variant": target.get("variant"),
            "observable_type": metrics.get("observable_type"),
            "observable_layer": metrics.get("observable_layer"),
            "intermediate_top1_recovery_rate": metrics.get("intermediate_top1_recovery_rate"),
            "token_top10_recovery_rate": metrics.get("token_top10_recovery_rate"),
            "embedding_cosine_similarity": metrics.get("embedding_cosine_similarity"),
            "sensitive_token_recovery_rate": metrics.get("sensitive_token_recovery_rate"),
            "risk_level": summary.get("risk_level"),
            "status": summary.get("status"),
            "selected_ridge_alpha": payload.get("config", {}).get("selected_ridge_alpha"),
            "target_name": payload.get("artifacts", {}).get("resolved_target", {}).get("name"),
        }
        rows.append(row)
        if row["target_name"] == baseline_target_name:
            baseline_row = row

    baseline_top1 = baseline_row.get("intermediate_top1_recovery_rate") if baseline_row is not None else None
    for row in rows:
        row["vs_stage_a_top1_delta"] = None if baseline_top1 is None or row["intermediate_top1_recovery_rate"] is None else float(row["intermediate_top1_recovery_rate"] - baseline_top1)

    rows.sort(key=lambda item: (item["intermediate_top1_recovery_rate"] is None, -(item["intermediate_top1_recovery_rate"] or -1.0)))
    return {
        "format": "qwen_security_isa_comparison_v1",
        "observable_type": observable_type,
        "baseline_target_name": baseline_target_name,
        "row_count": len(rows),
        "rows": rows,
    }
