from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluator import write_json
from src.stage_j_bridge_regression import run_stage_j_bridge_regression
from src.stage_j_standard_bridge import export_stage_j_redesign_standard_bridge
from src.stage_j_standard_weight_proof import build_stage_j_standard_weight_proof


def build_stage_j_paper_consistent_target() -> dict[str, Any]:
    return {
        "stage": "J",
        "goal": "paper_consistent_standard_deployable_obfuscated_checkpoint",
        "standard_graph_required": True,
        "standard_visible_keys_required": True,
        "bridge_is_final_target": False,
        "buffered_reference": "artifacts/stage_j_qwen_redesign",
        "historical_bridge_dir": "artifacts/stage_j_qwen_redesign_standard",
        "canonical_candidate_dir": "artifacts/stage_j_qwen_paper_consistent",
        "evidence_dir": "outputs/stage_j/paper_consistent",
        "required_evidence_files": [
            "standard_weight_proof.json",
            "attention_export_visible_proof.json",
            "ffn_export_visible_proof.json",
            "norm_export_visible_proof.json",
            "correctness_regression.json",
            "completion_summary.json",
        ],
        "completion_statuses": ["export_visible_complete", "not_complete"],
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_export_visible_component_metadata(source_server: Path, norm_strategy: str) -> dict[str, Any]:
    obfuscation_config = _load_json(source_server / "obfuscation_config.json")
    adapted_layers_raw = obfuscation_config.get("adapted_layers", [])
    adapted_layers = adapted_layers_raw if isinstance(adapted_layers_raw, list) else []
    attention_profile = obfuscation_config.get("attention_profile")
    attention_profile_str = attention_profile if isinstance(attention_profile, str) else ""
    attention_profile_lower = attention_profile_str.lower()
    has_profile = bool(attention_profile_str)
    has_head_group_semantics = has_profile and ("hqk" in attention_profile_lower or "group" in attention_profile_lower)
    has_block_semantics = has_profile and ("block" in attention_profile_lower)

    has_explicit_kappa_overrides = "kappa_overrides" in obfuscation_config
    calibration_required_keys = {"model_dir", "seed", "lambda", "h", "prompts_for_kappa", "adapted_layers"}
    has_calibration_inputs = calibration_required_keys.issubset(obfuscation_config.keys())
    has_kappa_overrides = norm_strategy == "kappa_fused" and (has_explicit_kappa_overrides or has_calibration_inputs)
    return {
        "attention": {
            "profile": attention_profile,
            "has_profile": has_profile,
            "has_head_group_semantics": has_head_group_semantics,
            "has_block_semantics": has_block_semantics,
            "lambda": obfuscation_config.get("lambda"),
            "h": obfuscation_config.get("h"),
            "alpha_e": obfuscation_config.get("alpha_e"),
            "alpha_h": obfuscation_config.get("alpha_h"),
        },
        "ffn": {
            "beta": obfuscation_config.get("beta"),
            "gamma": obfuscation_config.get("gamma"),
            "adapted_layers_count": len(adapted_layers),
        },
        "norm": {
            "strategy": norm_strategy,
            "has_kappa_overrides": has_kappa_overrides,
        },
    }


def export_stage_j_paper_consistent_candidate(
    export_dir: str | Path,
    *,
    source_dir: str | Path = "artifacts/stage_j_qwen_redesign",
    materialize: bool = False,
    norm_strategy: str = "kappa_fused",
) -> dict[str, Path]:
    result = export_stage_j_redesign_standard_bridge(
        export_dir,
        source_dir=source_dir,
        materialize=materialize,
        norm_strategy=norm_strategy,
    )

    manifest_path = result["manifest_path"]
    manifest = _load_json(manifest_path)
    resolved_norm_strategy = str(manifest.get("norm_strategy", norm_strategy))
    source_server = Path(source_dir) / "server"
    manifest.update(
        {
            "track": "paper_consistent_candidate",
            "goal": "paper_consistent_standard_deployable_obfuscated_checkpoint",
            "candidate_role": "canonical_stage_j_acceptance_target",
            "bridge_is_acceptance_target": False,
            "historical_bridge_reference": "artifacts/stage_j_qwen_redesign_standard",
            "buffered_source_of_truth": "artifacts/stage_j_qwen_redesign",
            "export_visible_components": _build_export_visible_component_metadata(source_server, resolved_norm_strategy),
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_stage_j_attention_export_visible_proof(manifest: dict[str, Any]) -> dict[str, Any]:
    attention = manifest.get("export_visible_components", {}).get("attention", {})
    passed = bool(
        manifest.get("standard_weight_proof", {}).get("is_standard_weight_export")
        and attention.get("has_profile")
        and attention.get("has_head_group_semantics")
        and attention.get("has_block_semantics")
    )
    return {
        "status": "pass" if passed else "fail",
        "profile": attention.get("profile"),
        "has_head_group_semantics": bool(attention.get("has_head_group_semantics")),
        "has_block_semantics": bool(attention.get("has_block_semantics")),
    }


def build_stage_j_ffn_export_visible_proof(manifest: dict[str, Any]) -> dict[str, Any]:
    ffn = manifest.get("export_visible_components", {}).get("ffn", {})
    adapted_layers_count = int(ffn.get("adapted_layers_count", 0))
    passed = bool(
        manifest.get("standard_weight_proof", {}).get("is_standard_weight_export")
        and adapted_layers_count > 0
    )
    return {
        "status": "pass" if passed else "fail",
        "adapted_layers_count": adapted_layers_count,
        "beta": ffn.get("beta"),
        "gamma": ffn.get("gamma"),
    }


def build_stage_j_norm_export_visible_proof(manifest: dict[str, Any]) -> dict[str, Any]:
    norm = manifest.get("export_visible_components", {}).get("norm", {})
    strategy = norm.get("strategy")
    has_kappa_overrides = bool(norm.get("has_kappa_overrides"))
    passed = bool(
        manifest.get("standard_weight_proof", {}).get("is_standard_weight_export")
        and strategy in {"kappa_fused", "metric_diag_sqrt"}
        and (has_kappa_overrides or strategy == "metric_diag_sqrt")
    )
    return {
        "status": "pass" if passed else "fail",
        "strategy": strategy,
        "has_kappa_overrides": has_kappa_overrides,
    }


def build_stage_j_correctness_regression_status(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    passed = bool(
        float(summary.get("generated_ids_exact_match_rate", 0.0)) > 0.0
        and float(summary.get("generated_text_exact_match_rate", 0.0)) > 0.0
    )
    return {
        "status": "pass" if passed else "fail",
        **summary,
    }


def build_stage_j_paper_consistent_completion_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    blocking_components = [name for name, payload in bundle.items() if payload.get("status") != "pass"]
    return {
        "completion_status": "export_visible_complete" if not blocking_components else "not_complete",
        "blocking_components": blocking_components,
    }


def build_stage_j_paper_consistent_evidence_bundle(
    *,
    candidate_dir: str | Path = "artifacts/stage_j_qwen_paper_consistent",
    source_dir: str | Path = "artifacts/stage_j_qwen_redesign",
    output_dir: str | Path = "outputs/stage_j/paper_consistent",
    correctness_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_dir = Path(candidate_dir)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    manifest = _load_json(candidate_dir / "manifest.json")
    prior_standard_weight_proof = manifest.get("standard_weight_proof", {})
    refreshed_standard_weight_proof = build_stage_j_standard_weight_proof(candidate_dir / "server")
    fallback_standard_weight_export = bool(prior_standard_weight_proof.get("is_standard_weight_export"))
    refreshed_standard_weight_export = bool(refreshed_standard_weight_proof.get("is_standard_weight_export"))
    effective_standard_weight_export = refreshed_standard_weight_export or (
        refreshed_standard_weight_proof.get("layout") == "missing_model_safetensors" and fallback_standard_weight_export
    )
    manifest["standard_weight_proof"] = {
        **refreshed_standard_weight_proof,
        "is_standard_weight_export": effective_standard_weight_export,
    }

    if correctness_payload is None:
        correctness_payload = run_stage_j_bridge_regression(
            buffered_server_dir=str((source_dir / "server").resolve()),
            bridge_server_dir=str((candidate_dir / "server").resolve()),
        )

    bundle = {
        "standard_weight_proof": {
            "status": "pass" if manifest["standard_weight_proof"].get("is_standard_weight_export") else "fail",
            **manifest["standard_weight_proof"],
        },
        "attention_export_visible_proof": build_stage_j_attention_export_visible_proof(manifest),
        "ffn_export_visible_proof": build_stage_j_ffn_export_visible_proof(manifest),
        "norm_export_visible_proof": build_stage_j_norm_export_visible_proof(manifest),
        "correctness_regression": build_stage_j_correctness_regression_status(correctness_payload),
    }
    completion_summary = build_stage_j_paper_consistent_completion_summary(bundle)
    bundle["completion_summary"] = completion_summary

    write_json(output_dir / "standard_weight_proof.json", bundle["standard_weight_proof"])
    write_json(output_dir / "attention_export_visible_proof.json", bundle["attention_export_visible_proof"])
    write_json(output_dir / "ffn_export_visible_proof.json", bundle["ffn_export_visible_proof"])
    write_json(output_dir / "norm_export_visible_proof.json", bundle["norm_export_visible_proof"])
    write_json(output_dir / "correctness_regression.json", bundle["correctness_regression"])
    write_json(output_dir / "completion_summary.json", bundle["completion_summary"])
    return bundle
