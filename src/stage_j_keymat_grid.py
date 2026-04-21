from __future__ import annotations

from typing import Any

from src.stage_j_keymat_search import evaluate_keymat_candidate, rank_keymat_candidates


def evaluate_keymat_grid(
    *,
    hidden_size: int,
    expansion_sizes: list[int],
    lams: list[float],
    families: list[str],
    seed_start: int,
    num_candidates: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in families:
        for expansion_size in expansion_sizes:
            for lam in lams:
                candidates = [
                    evaluate_keymat_candidate(
                        hidden_size=hidden_size,
                        expansion_size=expansion_size,
                        lam=lam,
                        seed=seed_start + offset,
                        family=family,
                    )
                    for offset in range(num_candidates)
                ]
                ranked = rank_keymat_candidates(candidates)
                best = ranked[0]
                rows.append(
                    {
                        "family": family,
                        "expansion_size": expansion_size,
                        "lam": lam,
                        "best_seed": best["seed"],
                        "best_offdiag_ratio": best["offdiag_ratio"],
                        "best_condition_number": best["condition_number"],
                    }
                )
    return rows


def rank_keymat_grid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda item: float(item["best_offdiag_ratio"]))
