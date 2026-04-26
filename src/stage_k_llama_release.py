from __future__ import annotations

from pathlib import Path

from src.stage_k_release import StageKProfile, export_stage_k_release


def default_stage_k_llama_profiles() -> list[StageKProfile]:
    return [
        StageKProfile(
            name="stable_reference",
            source_dir="artifacts/stage_j_llama_real_full_square",
            description="Zero-noise Llama-3.2-3B standard-shape full-layer checkpoint.",
            recommended_use="Regression baseline, correctness debugging, deterministic demos.",
            correctness_evidence_file="outputs/stage_j_llama/real_remote_validation.json",
        ),
        StageKProfile(
            name="tiny_a",
            source_dir="artifacts/stage_j_llama_real_full_square_tiny_a",
            description="Recommended non-zero noise Llama-3.2-3B standard-shape full-layer checkpoint.",
            recommended_use="Default delivery profile for Llama-3.2-3B when non-zero obfuscation noise is required.",
            correctness_evidence_file="outputs/stage_j_llama/real_tiny_a_remote_validation.json",
        ),
    ]


def export_stage_k_llama_release(
    export_dir: str | Path,
    *,
    materialize: bool = False,
) -> dict:
    return export_stage_k_release(
        export_dir,
        profiles=default_stage_k_llama_profiles(),
        materialize=materialize,
        recommended_profile="tiny_a",
        reference_profile="stable_reference",
        title="Stage-K Llama Release",
    )
