from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseCalibrationCase:
    name: str
    alpha_e: float
    alpha_h: float
    lam: float
    h: int
    beta: int
    gamma: float


def default_noise_cases() -> list[NoiseCalibrationCase]:
    return [
        NoiseCalibrationCase("stable_reference", alpha_e=0.0, alpha_h=0.0, lam=0.1, h=32, beta=4, gamma=1e3),
        NoiseCalibrationCase("paper_default", alpha_e=1.0, alpha_h=0.2, lam=0.3, h=128, beta=8, gamma=1e3),
        NoiseCalibrationCase("fine_a", alpha_e=0.1, alpha_h=0.05, lam=0.3, h=128, beta=8, gamma=1e3),
        NoiseCalibrationCase("fine_b", alpha_e=0.15, alpha_h=0.05, lam=0.3, h=128, beta=8, gamma=1e3),
        NoiseCalibrationCase("fine_c", alpha_e=0.1, alpha_h=0.1, lam=0.3, h=128, beta=8, gamma=1e3),
        NoiseCalibrationCase("mild_a", alpha_e=0.25, alpha_h=0.1, lam=0.3, h=128, beta=8, gamma=1e3),
        NoiseCalibrationCase("mild_b", alpha_e=0.5, alpha_h=0.1, lam=0.3, h=128, beta=8, gamma=1e3),
        NoiseCalibrationCase("mild_c", alpha_e=0.5, alpha_h=0.2, lam=0.3, h=128, beta=8, gamma=1e3),
        NoiseCalibrationCase("paper_lambda_low", alpha_e=1.0, alpha_h=0.2, lam=0.1, h=128, beta=8, gamma=1e3),
        NoiseCalibrationCase("paper_lambda_high", alpha_e=1.0, alpha_h=0.2, lam=1.0, h=128, beta=8, gamma=1e3),
    ]


def rank_noise_cases(results: list[dict]) -> list[dict]:
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
