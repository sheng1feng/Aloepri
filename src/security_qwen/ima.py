from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.key_manager import ordinary_token_ids
from src.security_qwen.artifacts import resolve_security_target
from src.security_qwen.metrics import classify_risk_level
from src.security_qwen.schema import SecurityEvalTarget, build_security_eval_payload


@dataclass(frozen=True)
class IMABaselineConfig:
    baseline_model_dir: str = DEFAULT_MODEL_DIR
    seed: int = DEFAULT_SEED
    train_size: int = 1024
    val_size: int = 128
    test_size: int = 128
    candidate_pool_size: int = 2048
    topk: int = 10
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ridge_alphas"] = list(self.ridge_alphas)
        return payload


def default_ima_gate2_targets() -> list[str]:
    return [
        "stage_a_standard",
        "stage_h_full_obfuscated",
        "stage_j_stable_reference",
        "stage_j_tiny_a",
        "stage_k_stable_reference",
        "stage_k_tiny_a",
    ]


def build_ima_template(target: SecurityEvalTarget) -> dict:
    primary_metric = None
    return build_security_eval_payload(
        attack="ima",
        target=target,
        config={
            "phase": "phase0",
            "dataset_name": "phase0_inversion_public_corpus",
            "seed": DEFAULT_SEED,
            "prediction_targets": ["token_id", "embedding"],
        },
        metrics={
            "token_top1_recovery_rate": primary_metric,
            "token_top10_recovery_rate": None,
            "embedding_cosine_similarity": None,
        },
        summary={
            "status": "planned",
            "primary_metric_name": "token_top1_recovery_rate",
            "primary_metric_value": primary_metric,
            "risk_level": classify_risk_level(primary_metric),
            "notes": "Phase 0 template only; IMA implementation not started.",
        },
        artifacts={},
    )


def _load_standard_embedding(server_dir: str) -> torch.Tensor:
    state = load_file(str(Path(server_dir) / "model.safetensors"))
    return state["model.embed_tokens.weight"].to(torch.float32)


def _load_buffered_stage_embedding(server_dir: str) -> torch.Tensor:
    state = load_file(str(Path(server_dir) / "model.safetensors"))
    return state["buffer::stage_a_model.model.embed_tokens.weight"].to(torch.float32)


def _load_stage_h_embedding(resolved_root_dir: str) -> torch.Tensor:
    payload = torch.load(Path(resolved_root_dir) / "server_model_state.pt", map_location="cpu")
    buffers = payload.get("buffer_state", {})
    return torch.as_tensor(buffers["stage_a_model.model.embed_tokens.weight"], dtype=torch.float32)


def load_ima_embedding_sources(
    *,
    target_name: str,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    baseline_state = load_file(str(Path(baseline_model_dir) / "model.safetensors"))
    baseline_embed = baseline_state["model.embed_tokens.weight"].to(torch.float32)

    resolved = resolve_security_target(target_name)
    if resolved.stage == "H":
        observed_embed = _load_stage_h_embedding(resolved.resolved_root_dir)
    else:
        if resolved.server_dir is None:
            raise FileNotFoundError(f"Target {target_name} has no server_dir")
        state = load_file(str(Path(resolved.server_dir) / "model.safetensors"))
        if "buffer::stage_a_model.model.embed_tokens.weight" in state:
            observed_embed = state["buffer::stage_a_model.model.embed_tokens.weight"].to(torch.float32)
        else:
            observed_embed = state["model.embed_tokens.weight"].to(torch.float32)

    if resolved.client_secret_path is None:
        raise FileNotFoundError(f"Target {target_name} has no client_secret_path")
    secret = torch.load(resolved.client_secret_path, map_location="cpu")
    metadata = {}
    if resolved.metadata_path is not None and Path(resolved.metadata_path).exists():
        metadata = json.loads(Path(resolved.metadata_path).read_text(encoding="utf-8"))
    return baseline_embed, observed_embed, {
        "resolved_target": resolved.to_dict(),
        "perm_vocab": torch.as_tensor(secret["perm_vocab"], dtype=torch.long),
        "inv_perm_vocab": torch.as_tensor(secret["inv_perm_vocab"], dtype=torch.long),
        "metadata": metadata,
    }


def _collect_sensitive_plain_ids(tokenizer) -> torch.Tensor:
    sensitive: set[int] = set()
    special_ids = set(tokenizer.all_special_ids)
    for prompt in DEFAULT_PROMPTS:
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for token_id in input_ids:
            if token_id not in special_ids and token_id < tokenizer.vocab_size:
                sensitive.add(int(token_id))
    return torch.tensor(sorted(sensitive), dtype=torch.long)


def _unique_preserve_order(items: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output


def _sample_ima_splits(
    *,
    tokenizer,
    train_size: int,
    val_size: int,
    test_size: int,
    candidate_pool_size: int,
    seed: int,
    sensitive_plain_ids_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    movable_plain_ids = ordinary_token_ids(tokenizer).tolist()
    sensitive_source = sensitive_plain_ids_override if sensitive_plain_ids_override is not None else _collect_sensitive_plain_ids(tokenizer)
    sensitive_plain_ids = [item for item in torch.as_tensor(sensitive_source, dtype=torch.long).tolist() if item in set(movable_plain_ids)]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shuffled = torch.tensor(movable_plain_ids, dtype=torch.long)[torch.randperm(len(movable_plain_ids), generator=generator)].tolist()
    remainder = [item for item in shuffled if item not in sensitive_plain_ids]

    test_plain = _unique_preserve_order(sensitive_plain_ids + remainder[: max(test_size - len(sensitive_plain_ids), 0)])
    remainder = [item for item in remainder if item not in set(test_plain)]

    val_plain = remainder[:val_size]
    remainder = remainder[val_size:]
    train_plain = remainder[:train_size]
    remainder = remainder[train_size:]

    candidate_plain = _unique_preserve_order(test_plain + remainder[: max(candidate_pool_size - len(test_plain), 0)])

    return {
        "train_plain_ids": torch.tensor(train_plain, dtype=torch.long),
        "val_plain_ids": torch.tensor(val_plain, dtype=torch.long),
        "test_plain_ids": torch.tensor(test_plain, dtype=torch.long),
        "candidate_plain_ids": torch.tensor(candidate_plain, dtype=torch.long),
        "sensitive_plain_ids": torch.tensor(sensitive_plain_ids, dtype=torch.long),
    }


def _fit_ridge_regressor(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    ridge_alpha: float,
) -> dict[str, torch.Tensor]:
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_norm = (x_train - x_mean) / x_std
    y_norm = (y_train - y_mean) / y_std
    ones = torch.ones((x_norm.shape[0], 1), dtype=x_norm.dtype)
    x_aug = torch.cat([x_norm, ones], dim=1)
    dim = x_aug.shape[1]
    identity = torch.eye(dim, dtype=x_norm.dtype)
    identity[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + ridge_alpha * identity
    rhs = x_aug.T @ y_norm
    weight = torch.linalg.solve(lhs, rhs)
    return {
        "weight": weight,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def _predict_ridge(model: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    x_norm = (x - model["x_mean"]) / model["x_std"]
    ones = torch.ones((x_norm.shape[0], 1), dtype=x_norm.dtype)
    x_aug = torch.cat([x_norm, ones], dim=1)
    y_norm = x_aug @ model["weight"]
    return y_norm * model["y_std"] + model["y_mean"]


def _row_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    y_norm = y / y.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (x_norm * y_norm).sum(dim=1)


def _evaluate_inversion_predictions(
    *,
    predicted_embeddings: torch.Tensor,
    true_plain_ids: torch.Tensor,
    candidate_plain_ids: torch.Tensor,
    baseline_embed: torch.Tensor,
    topk: int,
) -> dict[str, Any]:
    candidate_embeddings = baseline_embed[candidate_plain_ids]
    pred_norm = predicted_embeddings / predicted_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cand_norm = candidate_embeddings / candidate_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
    score_matrix = pred_norm @ cand_norm.T
    k = min(topk, score_matrix.shape[1])
    topk_indices = torch.topk(score_matrix, k=k, dim=1).indices
    predicted_plain_ids = candidate_plain_ids[topk_indices]
    hits = predicted_plain_ids.eq(true_plain_ids.unsqueeze(1))
    true_embeddings = baseline_embed[true_plain_ids]
    cosine = _row_cosine_similarity(predicted_embeddings, true_embeddings)
    return {
        "token_top1_recovery_rate": float(hits[:, 0].to(torch.float32).mean().item()),
        "token_top10_recovery_rate": float(hits[:, : min(10, hits.shape[1])].any(dim=1).to(torch.float32).mean().item()),
        "embedding_cosine_similarity": float(cosine.mean().item()),
        "predicted_plain_ids_sample": predicted_plain_ids[:10, 0].tolist(),
    }


def run_ima_baseline(
    *,
    target_name: str,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
    seed: int = DEFAULT_SEED,
    train_size: int = 1024,
    val_size: int = 128,
    test_size: int = 128,
    candidate_pool_size: int = 2048,
    topk: int = 10,
    ridge_alphas: tuple[float, ...] = (1e-4, 1e-2, 1.0),
    sensitive_plain_ids_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    baseline_embed, observed_embed, aux = load_ima_embedding_sources(
        target_name=target_name,
        baseline_model_dir=baseline_model_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_dir, trust_remote_code=True)
    splits = _sample_ima_splits(
        tokenizer=tokenizer,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
        sensitive_plain_ids_override=sensitive_plain_ids_override,
    )
    perm_vocab = aux["perm_vocab"]

    train_plain_ids = splits["train_plain_ids"]
    val_plain_ids = splits["val_plain_ids"]
    test_plain_ids = splits["test_plain_ids"]
    candidate_plain_ids = splits["candidate_plain_ids"]
    sensitive_plain_ids = splits["sensitive_plain_ids"]

    x_train = observed_embed[perm_vocab[train_plain_ids]]
    y_train = baseline_embed[train_plain_ids]
    x_val = observed_embed[perm_vocab[val_plain_ids]]
    y_val = baseline_embed[val_plain_ids]
    x_test = observed_embed[perm_vocab[test_plain_ids]]

    val_candidate_plain_ids = torch.unique(torch.cat([val_plain_ids, candidate_plain_ids], dim=0), sorted=False)

    best_alpha = None
    best_val_top1 = -1.0
    best_model = None
    alpha_scores: list[dict[str, Any]] = []
    for alpha in ridge_alphas:
        model = _fit_ridge_regressor(x_train, y_train, ridge_alpha=float(alpha))
        val_pred = _predict_ridge(model, x_val)
        val_metrics = _evaluate_inversion_predictions(
            predicted_embeddings=val_pred,
            true_plain_ids=val_plain_ids,
            candidate_plain_ids=val_candidate_plain_ids,
            baseline_embed=baseline_embed,
            topk=topk,
        )
        alpha_scores.append({"ridge_alpha": float(alpha), **val_metrics})
        if val_metrics["token_top1_recovery_rate"] > best_val_top1:
            best_val_top1 = val_metrics["token_top1_recovery_rate"]
            best_alpha = float(alpha)
            best_model = model

    if best_model is None or best_alpha is None:
        raise RuntimeError("IMA failed to select a best ridge model")

    test_pred = _predict_ridge(best_model, x_test)
    test_metrics = _evaluate_inversion_predictions(
        predicted_embeddings=test_pred,
        true_plain_ids=test_plain_ids,
        candidate_plain_ids=candidate_plain_ids,
        baseline_embed=baseline_embed,
        topk=topk,
    )

    sensitive_mask = torch.isin(test_plain_ids, sensitive_plain_ids) if sensitive_plain_ids.numel() > 0 else torch.zeros_like(test_plain_ids, dtype=torch.bool)
    sensitive_rate = None
    if bool(sensitive_mask.any()):
        candidate_embeddings = baseline_embed[candidate_plain_ids]
        pred_norm = test_pred / test_pred.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cand_norm = candidate_embeddings / candidate_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-8)
        score_matrix = pred_norm @ cand_norm.T
        top1_indices = torch.topk(score_matrix, k=1, dim=1).indices.squeeze(1)
        predicted_plain = candidate_plain_ids[top1_indices]
        sensitive_rate = float(predicted_plain[sensitive_mask].eq(test_plain_ids[sensitive_mask]).to(torch.float32).mean().item())

    runtime_seconds = time.perf_counter() - start_time
    resolved = aux["resolved_target"]
    payload = build_security_eval_payload(
        attack="ima",
        target=SecurityEvalTarget(
            stage=resolved["stage"],
            artifact_dir=resolved["artifact_dir"],
            profile=resolved["profile"],
            model_family="qwen",
            variant=resolved["variant"],
        ),
        config={
            "phase": "gate2_minimal",
            **IMABaselineConfig(
                baseline_model_dir=baseline_model_dir,
                seed=seed,
                train_size=int(train_plain_ids.numel()),
                val_size=int(val_plain_ids.numel()),
                test_size=int(test_plain_ids.numel()),
                candidate_pool_size=int(candidate_plain_ids.numel()),
                topk=topk,
                ridge_alphas=tuple(float(item) for item in ridge_alphas),
            ).to_dict(),
            "dataset_name": "phase0_token_row_proxy_corpus",
            "prediction_targets": ["embedding", "token_id"],
            "selected_ridge_alpha": best_alpha,
            "alpha_scores": alpha_scores,
        },
        metrics={
            **test_metrics,
            "sensitive_token_recovery_rate": sensitive_rate,
            "train_size": int(train_plain_ids.numel()),
            "val_size": int(val_plain_ids.numel()),
            "test_size": int(test_plain_ids.numel()),
            "candidate_plain_token_count": int(candidate_plain_ids.numel()),
            "attack_runtime_seconds": runtime_seconds,
        },
        summary={
            "status": "completed_minimal_baseline",
            "primary_metric_name": "token_top1_recovery_rate",
            "primary_metric_value": test_metrics["token_top1_recovery_rate"],
            "risk_level": classify_risk_level(test_metrics["token_top1_recovery_rate"]),
            "notes": "Gate-2 minimal ridge inversion baseline on obfuscated embedding rows.",
        },
        artifacts={
            "resolved_target": resolved,
            "predicted_plain_ids_sample": test_metrics["predicted_plain_ids_sample"],
        },
    )
    return payload


def build_ima_comparison_payload(
    *,
    result_payloads: list[dict[str, Any]],
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
            "token_top1_recovery_rate": metrics.get("token_top1_recovery_rate"),
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

    baseline_top1 = baseline_row.get("token_top1_recovery_rate") if baseline_row is not None else None
    baseline_cos = baseline_row.get("embedding_cosine_similarity") if baseline_row is not None else None
    for row in rows:
        row["vs_stage_a_top1_delta"] = None if baseline_top1 is None or row["token_top1_recovery_rate"] is None else float(row["token_top1_recovery_rate"] - baseline_top1)
        row["vs_stage_a_cosine_delta"] = None if baseline_cos is None or row["embedding_cosine_similarity"] is None else float(row["embedding_cosine_similarity"] - baseline_cos)

    rows.sort(key=lambda item: (item["token_top1_recovery_rate"] is None, -(item["token_top1_recovery_rate"] or -1.0)))
    return {
        "format": "qwen_security_ima_comparison_v1",
        "baseline_target_name": baseline_target_name,
        "row_count": len(rows),
        "rows": rows,
    }
