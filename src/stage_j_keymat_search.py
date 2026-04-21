from __future__ import annotations

from typing import Any

import torch

from src.keymat import build_keymat_transform, check_keymat_inverse


def evaluate_keymat_candidate(
    *,
    hidden_size: int,
    expansion_size: int,
    lam: float,
    seed: int,
) -> dict[str, Any]:
    transform = build_keymat_transform(
        d=hidden_size,
        h=expansion_size,
        lam=lam,
        init_seed=seed,
        dtype=torch.float32,
    )
    q = transform.inverse.to(torch.float32)
    metric = q @ q.T
    diag = torch.diagonal(metric)
    offdiag = metric - torch.diag_embed(diag)
    offdiag_ratio = float(offdiag.norm().item() / metric.norm().item())
    inverse_check = check_keymat_inverse(transform.key, transform.inverse)
    return {
        "seed": seed,
        "hidden_size": hidden_size,
        "expansion_size": expansion_size,
        "lam": lam,
        "offdiag_ratio": offdiag_ratio,
        "max_abs_inverse_error": float(inverse_check["max_abs_error"]),
        "condition_number": float(inverse_check["condition_number"]),
    }


def rank_keymat_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            float(item["offdiag_ratio"]),
            float(item["max_abs_inverse_error"]),
            float(item.get("condition_number", float("inf"))),
        ),
    )
