from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.key_manager import ordinary_token_ids
from src.security_qwen.artifacts import resolve_security_target
from src.security_qwen.metrics import classify_risk_level
from src.security_qwen.schema import SecurityEvalTarget, build_security_eval_payload


@dataclass(frozen=True)
class VMABaselineConfig:
    baseline_model_dir: str = DEFAULT_MODEL_DIR
    seed: int = DEFAULT_SEED
    eval_size: int = 256
    candidate_pool_size: int = 4096
    feature_bins: int = 64
    topk: int = 10
    layer_indices: tuple[int, ...] = (0, 11, 23)
    use_projection_sources: bool = True
    include_direct_sources: bool = True
    projection_kinds: tuple[str, ...] = ("q", "k", "v", "gate", "up")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layer_indices"] = list(self.layer_indices)
        payload["projection_kinds"] = list(self.projection_kinds)
        return payload


def default_vma_gate1_targets() -> list[str]:
    return [
        "stage_a_standard",
        "stage_h_full_obfuscated",
        "stage_j_stable_reference",
        "stage_j_tiny_a",
        "stage_k_stable_reference",
        "stage_k_tiny_a",
    ]


def build_vma_template(target: SecurityEvalTarget) -> dict:
    primary_metric = None
    return build_security_eval_payload(
        attack="vma",
        target=target,
        config={
            "phase": "phase0",
            "dataset_name": "phase0_smoke_prompts",
            "seed": DEFAULT_SEED,
            "matching_strategy": "planned_voting_across_weight_pairs",
        },
        metrics={
            "token_top1_recovery_rate": primary_metric,
            "token_top10_recovery_rate": None,
            "sensitive_token_recovery_rate": None,
        },
        summary={
            "status": "planned",
            "primary_metric_name": "token_top1_recovery_rate",
            "primary_metric_value": primary_metric,
            "risk_level": classify_risk_level(primary_metric),
            "notes": "Phase 0 template only; VMA implementation not started.",
        },
        artifacts={},
    )


def _load_stage_h_server_weights(resolved_root_dir: str) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(resolved_root_dir) / "server_model_state.pt", map_location="cpu")
    buffers = payload.get("buffer_state", {})
    weights: dict[str, torch.Tensor] = {}
    if "stage_a_model.model.embed_tokens.weight" in buffers:
        weights["embed"] = torch.as_tensor(buffers["stage_a_model.model.embed_tokens.weight"], dtype=torch.float32)
    if "stage_a_model.lm_head.weight" in buffers:
        weights["head"] = torch.as_tensor(buffers["stage_a_model.lm_head.weight"], dtype=torch.float32)
    if "head" not in weights and "embed" in weights:
        weights["head"] = weights["embed"]
    return weights


def _load_standard_server_weights(server_dir: str) -> dict[str, torch.Tensor]:
    state = load_file(str(Path(server_dir) / "model.safetensors"))
    weights: dict[str, torch.Tensor] = {}
    if "model.embed_tokens.weight" in state:
        weights["embed"] = state["model.embed_tokens.weight"].to(torch.float32)
    if "lm_head.weight" in state:
        weights["head"] = state["lm_head.weight"].to(torch.float32)
    if "head" not in weights and "embed" in weights:
        weights["head"] = weights["embed"]
    return weights


def _infer_layer_indices_from_standard_state(state: dict[str, torch.Tensor]) -> list[int]:
    pattern = re.compile(r"^model\.layers\.(\d+)\.")
    indices = sorted(
        {
            int(match.group(1))
            for key in state.keys()
            for match in [pattern.match(key)]
            if match is not None
        }
    )
    return indices


def _infer_layer_indices_from_stage_h_buffers(buffers: dict[str, torch.Tensor]) -> list[int]:
    pattern = re.compile(r"^stage_a_model\.model\.layers\.(\d+)\.")
    indices = sorted(
        {
            int(match.group(1))
            for key in buffers.keys()
            for match in [pattern.match(key)]
            if match is not None
        }
    )
    return indices


def _default_layer_indices(all_indices: list[int]) -> list[int]:
    if not all_indices:
        return []
    selected = {all_indices[0], all_indices[len(all_indices) // 2], all_indices[-1]}
    return sorted(selected)


def infer_vma_default_projection_layers(
    *,
    target_name: str,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
) -> list[int]:
    baseline_state = load_file(str(Path(baseline_model_dir) / "model.safetensors"))
    baseline_layers = _default_layer_indices(_infer_layer_indices_from_standard_state(baseline_state))
    resolved = resolve_security_target(target_name)
    if resolved.stage == "H":
        payload = torch.load(Path(resolved.resolved_root_dir) / "server_model_state.pt", map_location="cpu")
        observed_layers = _default_layer_indices(_infer_layer_indices_from_stage_h_buffers(payload.get("buffer_state", {})))
    else:
        if resolved.server_dir is None:
            return baseline_layers
        observed_state = load_file(str(Path(resolved.server_dir) / "model.safetensors"))
        observed_layers = _default_layer_indices(_infer_layer_indices_from_standard_state(observed_state))
    return sorted(set(baseline_layers) & set(observed_layers))


def _load_standard_projection_weights(
    state: dict[str, torch.Tensor],
    layer_indices: list[int],
) -> dict[str, torch.Tensor]:
    projections: dict[str, torch.Tensor] = {}
    mapping = {
        "q": "self_attn.q_proj.weight",
        "k": "self_attn.k_proj.weight",
        "v": "self_attn.v_proj.weight",
        "gate": "mlp.gate_proj.weight",
        "up": "mlp.up_proj.weight",
    }
    for layer_idx in layer_indices:
        for short_name, suffix in mapping.items():
            key = f"model.layers.{layer_idx}.{suffix}"
            if key in state:
                projections[f"layer_{layer_idx}_{short_name}"] = state[key].to(torch.float32)
    return projections


def _load_stage_h_projection_weights(
    resolved_root_dir: str,
    layer_indices: list[int],
) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(resolved_root_dir) / "server_model_state.pt", map_location="cpu")
    buffers = payload.get("buffer_state", {})
    projections: dict[str, torch.Tensor] = {}
    mapping = {
        "q": "self_attn.q_weight",
        "k": "self_attn.k_weight",
        "v": "self_attn.v_weight",
        "gate": "mlp.gate_weight",
        "up": "mlp.up_weight",
    }
    for layer_idx in layer_indices:
        for short_name, suffix in mapping.items():
            key = f"stage_a_model.model.layers.{layer_idx}.{suffix}"
            if key in buffers:
                projections[f"layer_{layer_idx}_{short_name}"] = torch.as_tensor(buffers[key], dtype=torch.float32)
    return projections


def load_vma_weight_sources(
    *,
    target_name: str,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    baseline_state = load_file(str(Path(baseline_model_dir) / "model.safetensors"))
    baseline_layer_indices = _default_layer_indices(_infer_layer_indices_from_standard_state(baseline_state))
    baseline_weights = {
        "embed": baseline_state["model.embed_tokens.weight"].to(torch.float32),
        "head": baseline_state["model.embed_tokens.weight"].to(torch.float32),
    }
    baseline_projection_weights = _load_standard_projection_weights(baseline_state, baseline_layer_indices)

    resolved = resolve_security_target(target_name)
    if resolved.stage == "H":
        observed_weights = _load_stage_h_server_weights(resolved.resolved_root_dir)
        raw_payload = torch.load(Path(resolved.resolved_root_dir) / "server_model_state.pt", map_location="cpu")
        observed_layer_indices = _default_layer_indices(_infer_layer_indices_from_stage_h_buffers(raw_payload.get("buffer_state", {})))
        observed_projection_weights = _load_stage_h_projection_weights(resolved.resolved_root_dir, observed_layer_indices)
    else:
        if resolved.server_dir is None:
            raise FileNotFoundError(f"Target {target_name} has no server_dir: {resolved}")
        observed_weights = _load_standard_server_weights(resolved.server_dir)
        observed_state = load_file(str(Path(resolved.server_dir) / "model.safetensors"))
        observed_layer_indices = _default_layer_indices(_infer_layer_indices_from_standard_state(observed_state))
        observed_projection_weights = _load_standard_projection_weights(observed_state, observed_layer_indices)

    if "embed" not in observed_weights:
        raise KeyError(f"Target {target_name} does not expose an embeddable weight source")

    if resolved.client_secret_path is None:
        raise FileNotFoundError(f"Target {target_name} has no client_secret_path for evaluation")
    secret = torch.load(resolved.client_secret_path, map_location="cpu")

    metadata = {}
    if resolved.metadata_path is not None and Path(resolved.metadata_path).exists():
        metadata = json.loads(Path(resolved.metadata_path).read_text(encoding="utf-8"))

    common_projection_names = sorted(set(baseline_projection_weights.keys()) & set(observed_projection_weights.keys()))
    baseline_projection_weights = {name: baseline_projection_weights[name] for name in common_projection_names}
    observed_projection_weights = {name: observed_projection_weights[name] for name in common_projection_names}

    return baseline_weights, observed_weights, baseline_projection_weights, observed_projection_weights, {
        "resolved_target": resolved.to_dict(),
        "perm_vocab": torch.as_tensor(secret["perm_vocab"], dtype=torch.long),
        "inv_perm_vocab": torch.as_tensor(secret["inv_perm_vocab"], dtype=torch.long),
        "metadata": metadata,
        "layer_indices": common_projection_names,
    }


def _sorted_quantile_features(matrix: torch.Tensor, bins: int) -> torch.Tensor:
    matrix = torch.as_tensor(matrix, dtype=torch.float32)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    sorted_rows, _ = torch.sort(matrix, dim=1)
    positions = torch.linspace(0, sorted_rows.shape[1] - 1, steps=bins)
    positions = positions.round().to(torch.long)
    features = sorted_rows[:, positions]
    features = features - features.mean(dim=1, keepdim=True)
    norms = features.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return features / norms


def _rowwise_standardize(score_matrix: torch.Tensor) -> torch.Tensor:
    mean = score_matrix.mean(dim=1, keepdim=True)
    std = score_matrix.std(dim=1, keepdim=True).clamp_min(1e-8)
    return (score_matrix - mean) / std


def _collect_sensitive_plain_ids(tokenizer) -> torch.Tensor:
    sensitive: set[int] = set()
    special_ids = set(tokenizer.all_special_ids)
    for prompt in DEFAULT_PROMPTS:
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for token_id in input_ids:
            if token_id not in special_ids and token_id < tokenizer.vocab_size:
                sensitive.add(int(token_id))
    return torch.tensor(sorted(sensitive), dtype=torch.long)


def _sample_eval_and_candidate_sets(
    *,
    tokenizer,
    perm_vocab: torch.Tensor,
    inv_perm_vocab: torch.Tensor,
    eval_size: int,
    candidate_pool_size: int,
    seed: int,
    sensitive_plain_ids_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    movable_plain_ids = ordinary_token_ids(tokenizer)
    movable_plain_set = set(int(item) for item in movable_plain_ids.tolist())
    sensitive_source = sensitive_plain_ids_override if sensitive_plain_ids_override is not None else _collect_sensitive_plain_ids(tokenizer)
    sensitive_plain_ids = torch.tensor(
        [item for item in torch.as_tensor(sensitive_source, dtype=torch.long).tolist() if item in movable_plain_set],
        dtype=torch.long,
    )
    sensitive_obfuscated_ids = perm_vocab[sensitive_plain_ids] if sensitive_plain_ids.numel() > 0 else torch.empty(0, dtype=torch.long)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    movable_obfuscated_ids = perm_vocab[movable_plain_ids]
    candidate_needed = max(candidate_pool_size, sensitive_plain_ids.numel())
    eval_needed = max(eval_size, sensitive_obfuscated_ids.numel())

    shuffled_eval = movable_obfuscated_ids[torch.randperm(movable_obfuscated_ids.numel(), generator=generator)]
    eval_extra = shuffled_eval[: max(eval_needed - sensitive_obfuscated_ids.numel(), 0)]
    eval_obfuscated_ids = torch.unique(torch.cat([sensitive_obfuscated_ids, eval_extra], dim=0), sorted=False)
    eval_obfuscated_ids = eval_obfuscated_ids[:eval_needed]

    true_plain_ids = inv_perm_vocab[eval_obfuscated_ids]
    shuffled_candidates = movable_plain_ids[torch.randperm(movable_plain_ids.numel(), generator=generator)]
    candidate_extra = shuffled_candidates[: max(candidate_needed - true_plain_ids.numel(), 0)]
    candidate_plain_ids = torch.unique(torch.cat([true_plain_ids, candidate_extra], dim=0), sorted=False)
    candidate_plain_ids = candidate_plain_ids[:candidate_needed]
    return eval_obfuscated_ids, candidate_plain_ids, sensitive_plain_ids


def _topk_hits(score_matrix: torch.Tensor, candidate_plain_ids: torch.Tensor, true_plain_ids: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = min(topk, score_matrix.shape[1])
    topk_indices = torch.topk(score_matrix, k=k, dim=1).indices
    predicted_plain_ids = candidate_plain_ids[topk_indices]
    top1 = predicted_plain_ids[:, 0]
    hits = predicted_plain_ids.eq(true_plain_ids.unsqueeze(1))
    return top1, hits


def _projection_layers_from_names(names: list[str]) -> list[int]:
    layers: list[int] = []
    for name in names:
        if name.startswith("layer_"):
            try:
                layer_idx = int(name.split("_")[1])
            except Exception:
                continue
            if layer_idx not in layers:
                layers.append(layer_idx)
    return layers


def _filter_projection_names(
    names: list[str],
    *,
    layer_indices: tuple[int, ...] | None,
    projection_kinds: tuple[str, ...] | None,
) -> list[str]:
    filtered = list(names)
    if layer_indices is not None:
        allowed_prefixes = {f"layer_{index}_" for index in layer_indices}
        filtered = [name for name in filtered if any(name.startswith(prefix) for prefix in allowed_prefixes)]
    if projection_kinds is not None:
        allowed_suffixes = {f"_{kind}" for kind in projection_kinds}
        filtered = [name for name in filtered if any(name.endswith(suffix) for suffix in allowed_suffixes)]
    return filtered


def run_vma_baseline(
    *,
    target_name: str,
    baseline_model_dir: str = DEFAULT_MODEL_DIR,
    seed: int = DEFAULT_SEED,
    eval_size: int = 256,
    candidate_pool_size: int = 4096,
    feature_bins: int = 64,
    topk: int = 10,
    layer_indices: tuple[int, ...] | None = None,
    use_projection_sources: bool = True,
    include_direct_sources: bool = True,
    projection_kinds: tuple[str, ...] | None = None,
    sensitive_plain_ids_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    baseline_weights, observed_weights, baseline_projection_weights, observed_projection_weights, aux = load_vma_weight_sources(
        target_name=target_name,
        baseline_model_dir=baseline_model_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(baseline_model_dir, trust_remote_code=True)
    perm_vocab = aux["perm_vocab"]
    inv_perm_vocab = aux["inv_perm_vocab"]

    eval_obfuscated_ids, candidate_plain_ids, sensitive_plain_ids = _sample_eval_and_candidate_sets(
        tokenizer=tokenizer,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        eval_size=eval_size,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
        sensitive_plain_ids_override=sensitive_plain_ids_override,
    )
    true_plain_ids = inv_perm_vocab[eval_obfuscated_ids]

    source_scores: dict[str, torch.Tensor] = {}
    source_metrics: dict[str, Any] = {}
    combined_score = None
    observed_embed_rows = observed_weights["embed"][eval_obfuscated_ids]
    baseline_embed_rows = baseline_weights["embed"][candidate_plain_ids]

    if include_direct_sources:
        for source_name, observed_matrix in observed_weights.items():
            baseline_matrix = baseline_weights["embed"] if source_name == "head" else baseline_weights[source_name]
            observed_rows = observed_matrix[eval_obfuscated_ids]
            baseline_rows = baseline_matrix[candidate_plain_ids]
            observed_features = _sorted_quantile_features(observed_rows, feature_bins)
            baseline_features = _sorted_quantile_features(baseline_rows, feature_bins)
            score_matrix = _rowwise_standardize(observed_features @ baseline_features.T)
            source_scores[source_name] = score_matrix
            combined_score = score_matrix if combined_score is None else combined_score + score_matrix

            source_top1, source_hits = _topk_hits(score_matrix, candidate_plain_ids, true_plain_ids, max(topk, 10))
            source_metrics[source_name] = {
                "token_top1_recovery_rate": float(source_hits[:, 0].to(torch.float32).mean().item()),
                "token_top10_recovery_rate": float(source_hits[:, : min(10, source_hits.shape[1])].any(dim=1).to(torch.float32).mean().item()),
                "predicted_plain_ids_sample": source_top1[:10].tolist(),
            }

    active_projection_names = sorted(set(baseline_projection_weights.keys()) & set(observed_projection_weights.keys()))
    if not use_projection_sources:
        active_projection_names = []
    else:
        active_projection_names = _filter_projection_names(
            active_projection_names,
            layer_indices=layer_indices,
            projection_kinds=projection_kinds,
        )

    for source_name in active_projection_names:
        observed_proj = observed_projection_weights[source_name]
        baseline_proj = baseline_projection_weights[source_name]
        observed_projected = observed_embed_rows @ observed_proj.T
        baseline_projected = baseline_embed_rows @ baseline_proj.T
        observed_features = _sorted_quantile_features(observed_projected, feature_bins)
        baseline_features = _sorted_quantile_features(baseline_projected, feature_bins)
        score_matrix = _rowwise_standardize(observed_features @ baseline_features.T)
        source_scores[source_name] = score_matrix
        combined_score = score_matrix if combined_score is None else combined_score + score_matrix

        source_top1, source_hits = _topk_hits(score_matrix, candidate_plain_ids, true_plain_ids, max(topk, 10))
        source_metrics[source_name] = {
            "token_top1_recovery_rate": float(source_hits[:, 0].to(torch.float32).mean().item()),
            "token_top10_recovery_rate": float(source_hits[:, : min(10, source_hits.shape[1])].any(dim=1).to(torch.float32).mean().item()),
            "predicted_plain_ids_sample": source_top1[:10].tolist(),
        }

    if combined_score is None:
        raise RuntimeError(f"No VMA sources available for target {target_name}")

    _, combined_hits = _topk_hits(combined_score, candidate_plain_ids, true_plain_ids, max(topk, 100))
    top1_rate = float(combined_hits[:, 0].to(torch.float32).mean().item())
    top10_rate = float(combined_hits[:, : min(10, combined_hits.shape[1])].any(dim=1).to(torch.float32).mean().item())
    top100_rate = float(combined_hits[:, : min(100, combined_hits.shape[1])].any(dim=1).to(torch.float32).mean().item())

    sensitive_mask = torch.isin(true_plain_ids, sensitive_plain_ids) if sensitive_plain_ids.numel() > 0 else torch.zeros_like(true_plain_ids, dtype=torch.bool)
    sensitive_rate = None
    if bool(sensitive_mask.any()):
        sensitive_rate = float(combined_hits[sensitive_mask, 0].to(torch.float32).mean().item())

    target = resolve_security_target(target_name)
    payload = build_security_eval_payload(
        attack="vma",
        target=SecurityEvalTarget(
            stage=target.stage,
            artifact_dir=target.artifact_dir,
            profile=target.profile,
            model_family="qwen",
            variant=target.variant,
        ),
        config={
            "phase": "gate1_minimal",
            "dataset_name": "phase0_smoke_prompts",
            **VMABaselineConfig(
                baseline_model_dir=baseline_model_dir,
                seed=seed,
                eval_size=int(eval_obfuscated_ids.numel()),
                candidate_pool_size=int(candidate_plain_ids.numel()),
                feature_bins=feature_bins,
                topk=topk,
                layer_indices=tuple(_projection_layers_from_names(active_projection_names)),
                use_projection_sources=use_projection_sources,
                include_direct_sources=include_direct_sources,
                projection_kinds=tuple(projection_kinds) if projection_kinds is not None else tuple(sorted({name.split('_')[-1] for name in active_projection_names})),
            ).to_dict(),
            "matching_strategy": "sorted_quantile_signature_zscore_voting_with_projections" if use_projection_sources else "sorted_quantile_signature_zscore_voting_direct_only",
            "source_names": sorted(observed_weights.keys()) if include_direct_sources else [],
            "projection_source_names": active_projection_names,
        },
        metrics={
            "token_top1_recovery_rate": top1_rate,
            "token_top10_recovery_rate": top10_rate,
            "token_top100_recovery_rate": top100_rate,
            "sensitive_token_recovery_rate": sensitive_rate,
            "evaluated_obfuscated_token_count": int(eval_obfuscated_ids.numel()),
            "candidate_plain_token_count": int(candidate_plain_ids.numel()),
            "sensitive_token_count": int(sensitive_plain_ids.numel()),
            "source_metrics": source_metrics,
            "projection_source_count": len(active_projection_names),
            "total_source_count": len(source_metrics),
        },
        summary={
            "status": "completed_minimal_baseline",
            "primary_metric_name": "token_top1_recovery_rate",
            "primary_metric_value": top1_rate,
            "risk_level": classify_risk_level(top1_rate),
            "notes": "Gate-1 strengthened baseline using sorted quantile signatures, projection-derived sources, and row-wise standardized voting."
            if use_projection_sources
            else "Gate-1 direct-only baseline using embed/head sorted quantile signatures and row-wise standardized voting.",
        },
        artifacts={
            "resolved_target": target.to_dict(),
            "source_names": sorted(observed_weights.keys()) if include_direct_sources else [],
            "projection_source_names": active_projection_names,
        },
    )
    return payload


def build_vma_comparison_payload(
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
            "token_top100_recovery_rate": metrics.get("token_top100_recovery_rate"),
            "sensitive_token_recovery_rate": metrics.get("sensitive_token_recovery_rate"),
            "risk_level": summary.get("risk_level"),
            "status": summary.get("status"),
            "target_name": payload.get("artifacts", {}).get("resolved_target", {}).get("name"),
        }
        rows.append(row)
        if row["target_name"] == baseline_target_name:
            baseline_row = row

    baseline_top1 = baseline_row.get("token_top1_recovery_rate") if baseline_row is not None else None
    baseline_top10 = baseline_row.get("token_top10_recovery_rate") if baseline_row is not None else None

    for row in rows:
        top1 = row["token_top1_recovery_rate"]
        top10 = row["token_top10_recovery_rate"]
        row["vs_stage_a_top1_delta"] = None if baseline_top1 is None or top1 is None else float(top1 - baseline_top1)
        row["vs_stage_a_top10_delta"] = None if baseline_top10 is None or top10 is None else float(top10 - baseline_top10)

    rows.sort(
        key=lambda item: (
            item["token_top1_recovery_rate"] is None,
            -(item["token_top1_recovery_rate"] or -1.0),
            item["stage"] or "",
            item["profile"] or "",
        )
    )

    return {
        "format": "qwen_security_vma_comparison_v1",
        "baseline_target_name": baseline_target_name,
        "row_count": len(rows),
        "rows": rows,
    }


def build_vma_source_attribution_payload(
    *,
    target_name: str,
    result_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in result_payloads:
        metrics = payload["metrics"]
        config = payload["config"]
        rows.append(
            {
                "label": config.get("attribution_label"),
                "include_direct_sources": config.get("include_direct_sources"),
                "projection_kinds": config.get("projection_kinds"),
                "layer_indices": config.get("layer_indices"),
                "token_top1_recovery_rate": metrics.get("token_top1_recovery_rate"),
                "token_top10_recovery_rate": metrics.get("token_top10_recovery_rate"),
                "projection_source_count": metrics.get("projection_source_count"),
                "total_source_count": metrics.get("total_source_count"),
                "risk_level": payload["summary"].get("risk_level"),
            }
        )
    rows.sort(key=lambda item: (item["token_top1_recovery_rate"] is None, -(item["token_top1_recovery_rate"] or -1.0)))
    return {
        "format": "qwen_security_vma_source_attribution_v1",
        "target_name": target_name,
        "row_count": len(rows),
        "rows": rows,
    }


def build_vma_layer_ablation_payload(
    *,
    target_name: str,
    result_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in result_payloads:
        metrics = payload["metrics"]
        config = payload["config"]
        rows.append(
            {
                "label": config.get("ablation_label"),
                "layer_indices": config.get("layer_indices"),
                "include_direct_sources": config.get("include_direct_sources"),
                "projection_kinds": config.get("projection_kinds"),
                "token_top1_recovery_rate": metrics.get("token_top1_recovery_rate"),
                "token_top10_recovery_rate": metrics.get("token_top10_recovery_rate"),
                "projection_source_count": metrics.get("projection_source_count"),
                "risk_level": payload["summary"].get("risk_level"),
            }
        )
    rows.sort(key=lambda item: (item["token_top1_recovery_rate"] is None, -(item["token_top1_recovery_rate"] or -1.0)))
    return {
        "format": "qwen_security_vma_layer_ablation_v1",
        "target_name": target_name,
        "row_count": len(rows),
        "rows": rows,
    }
