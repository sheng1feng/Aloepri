from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_redesigned_expression_audit() -> dict[str, Any]:
    stage_h_cfg = _load_json("artifacts/stage_h_pretrained/server/obfuscation_config.json")
    stage_j_manifest = _load_json("artifacts/stage_j_qwen_redesign/manifest.json")
    stage_k_catalog = _load_json("artifacts/stage_k_release/catalog.json")

    profiles = [item for item in stage_k_catalog.get("profiles", []) if isinstance(item, dict)]
    profile_sources = [item.get("source_dir") for item in profiles if isinstance(item.get("source_dir"), str)]

    stage_h_source = {
        "attention_profile_present": bool(stage_h_cfg.get("attention_profile")),
        "attention_profile": stage_h_cfg.get("attention_profile"),
        "keymat_parameters_present": all(key in stage_h_cfg for key in ["lambda", "h"]),
        "keymat_lambda": stage_h_cfg.get("lambda"),
        "keymat_h": stage_h_cfg.get("h"),
        "adapted_layers_count": len(stage_h_cfg.get("adapted_layers", [])),
        "full_layer_adaptation": len(stage_h_cfg.get("adapted_layers", [])) > 1,
    }

    stage_j = {
        "bootstraps_from_stage_h_pretrained": stage_j_manifest.get("bootstrap_source") == "artifacts/stage_h_pretrained",
        "source_stages": stage_j_manifest.get("source_stages", []),
        "has_component_level_expression_manifest": "component_expression" in stage_j_manifest,
        "server_dir_present": Path("artifacts/stage_j_qwen_redesign/server").exists(),
        "client_dir_present": Path("artifacts/stage_j_qwen_redesign/client").exists(),
    }

    stage_k = {
        "points_to_redesigned_stage_j": stage_k_catalog.get("stage_lineage") == "redesigned_qwen_stage_j"
        and all(source == "artifacts/stage_j_qwen_redesign" for source in profile_sources),
        "profile_sources": profile_sources,
        "has_expression_metadata_in_catalog": any("manifest" in item for item in profiles),
    }

    verdict = {
        "expression_enters_bootstrap_source": stage_h_source["attention_profile_present"] and stage_h_source["keymat_parameters_present"],
        "expression_is_proven_in_stage_j_export": stage_j["has_component_level_expression_manifest"],
        "expression_is_carried_into_stage_k_release": stage_k["has_expression_metadata_in_catalog"],
    }

    summary = {
        "status": "bootstrap_expression_present_but_not_yet_proven_in_stage_j_manifest"
        if verdict["expression_enters_bootstrap_source"] and not verdict["expression_is_proven_in_stage_j_export"]
        else "expression_audit_inconclusive",
        "next_action": "materialize_component_level_expression_metadata_and_then_re-run_vma",
    }

    return {
        "format": "qwen_stage_hk_expression_audit_v1",
        "stage_h_source": stage_h_source,
        "stage_j": stage_j,
        "stage_k": stage_k,
        "verdict": verdict,
        "summary": summary,
    }
