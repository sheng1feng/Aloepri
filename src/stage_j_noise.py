from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageJNoiseCase:
    name: str
    alpha_e: float
    alpha_h: float
    global_scale: float = 1.0


def default_stage_j_noise_cases() -> list[StageJNoiseCase]:
    return [
        StageJNoiseCase("stable_reference", alpha_e=0.0, alpha_h=0.0),
        StageJNoiseCase("tiny_a", alpha_e=0.02, alpha_h=0.01),
        StageJNoiseCase("tiny_b", alpha_e=0.05, alpha_h=0.02),
        StageJNoiseCase("small_a", alpha_e=0.1, alpha_h=0.05),
        StageJNoiseCase("small_b", alpha_e=0.15, alpha_h=0.05),
        StageJNoiseCase("small_c", alpha_e=0.1, alpha_h=0.1),
        StageJNoiseCase("paper_like", alpha_e=1.0, alpha_h=0.2),
    ]


def rank_stage_j_noise_cases(results: list[dict]) -> list[dict]:
    def score(item: dict):
        summary = item["summary"]
        return (
            float(summary.get("generated_text_exact_match_rate", 0.0)),
            float(summary.get("generated_ids_exact_match_rate", 0.0)),
            float(summary.get("greedy_first_token_match_rate", 0.0)),
            -float(summary.get("avg_final_logits_restored_max_abs_error", 1e9)),
            -float(summary.get("avg_layer_23_block_out_max_abs_error", 1e9)),
        )

    return sorted(results, key=score, reverse=True)
