from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.stage_j_standard_bridge import export_stage_j_redesign_standard_bridge


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
    has_kappa_overrides = "kappa_overrides" in obfuscation_config
    return {
        "attention": {
            "profile": obfuscation_config.get("attention_profile"),
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
