from __future__ import annotations

from typing import Any


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
