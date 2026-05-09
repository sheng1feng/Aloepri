from __future__ import annotations

import json
from pathlib import Path

from src.stage_k_correctness import (
    build_stage_k_correctness_summary,
    resolve_stage_k_profile_paths,
    summarize_prompt_results,
)


def test_resolve_stage_k_profile_paths_reads_catalog_profile_dirs(tmp_path: Path) -> None:
    release_dir = tmp_path / "stage_k_release"
    (release_dir / "profiles" / "default" / "server").mkdir(parents=True)
    (release_dir / "profiles" / "default" / "client").mkdir(parents=True)
    (release_dir / "profiles" / "default" / "manifest.json").write_text(
        json.dumps(
            {
                "buffered_source_of_truth": "artifacts/stage_j_qwen_redesign",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (release_dir / "catalog.json").write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "default",
                        "server_dir": "profiles/default/server",
                        "client_secret": "profiles/default/client/client_secret.pt",
                        "correctness_evidence_file": "outputs/stage_k_release/correctness/default.json",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    resolved = resolve_stage_k_profile_paths(release_dir, "default")

    assert resolved["server_dir"] == str(release_dir / "profiles" / "default" / "server")
    assert resolved["client_secret"] == str(release_dir / "profiles" / "default" / "client" / "client_secret.pt")
    assert resolved["correctness_evidence_file"] == "outputs/stage_k_release/correctness/default.json"
    assert resolved["buffered_server_dir"] == "artifacts/stage_j_qwen_redesign/server"


def test_summarize_prompt_results_uses_stage_j_compatible_metrics() -> None:
    summary = summarize_prompt_results(
        [
            {
                "full_logits_max_abs_error": 2.0,
                "full_logits_mean_abs_error": 1.0,
                "last_token_logits_max_abs_error": 0.4,
                "last_token_logits_mean_abs_error": 0.2,
                "greedy_first_token_match": True,
                "generated_ids_exact_match": True,
                "generated_text_exact_match": False,
                "baseline_has_nan_or_inf": False,
                "stage_k_has_nan_or_inf": False,
            },
            {
                "full_logits_max_abs_error": 4.0,
                "full_logits_mean_abs_error": 2.0,
                "last_token_logits_max_abs_error": 0.2,
                "last_token_logits_mean_abs_error": 0.1,
                "greedy_first_token_match": False,
                "generated_ids_exact_match": True,
                "generated_text_exact_match": True,
                "baseline_has_nan_or_inf": False,
                "stage_k_has_nan_or_inf": False,
            },
        ]
    )

    assert summary["prompt_count"] == 2
    assert summary["avg_restored_full_logits_max_abs_error"] == 3.0
    assert summary["avg_restored_full_logits_mean_abs_error"] == 1.5
    assert summary["avg_restored_last_token_max_abs_error"] == 0.30000000000000004
    assert summary["avg_restored_last_token_mean_abs_error"] == 0.15000000000000002
    assert summary["greedy_first_token_match_rate"] == 0.5
    assert summary["generated_ids_exact_match_rate"] == 1.0
    assert summary["generated_text_exact_match_rate"] == 0.5


def test_build_stage_k_correctness_summary_wraps_profile_results() -> None:
    release_dir = "artifacts/stage_k_release"
    output_dir = "outputs/stage_k_release/correctness"
    summary = build_stage_k_correctness_summary(
        release_dir=release_dir,
        output_dir=output_dir,
        profile_results={
            "default": {
                "status": "pass",
                "summary": {
                    "status": "pass",
                    "prompt_count": 5,
                    "generated_ids_exact_match_rate": 1.0,
                    "generated_text_exact_match_rate": 1.0,
                },
            },
            "reference": {
                "status": "pass",
                "summary": {
                    "status": "pass",
                    "prompt_count": 5,
                    "generated_ids_exact_match_rate": 1.0,
                    "generated_text_exact_match_rate": 1.0,
                },
            },
        },
    )

    assert summary["release_dir"] == release_dir
    assert summary["profiles"] == ["default", "reference"]
    assert summary["completion_status"] == "complete"
    assert summary["blocking_profiles"] == []
    assert summary["profile_evidence_files"]["default"] == f"{output_dir}/default.json"
    assert summary["profile_evidence_files"]["reference"] == f"{output_dir}/reference.json"
    assert summary["profile_summaries"]["default"]["generated_ids_exact_match_rate"] == 1.0
