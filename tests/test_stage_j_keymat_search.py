from src.stage_j_keymat_search import rank_keymat_candidates


def test_rank_keymat_candidates_prefers_lower_offdiag_ratio() -> None:
    ranked = rank_keymat_candidates(
        [
            {"seed": 1, "offdiag_ratio": 0.9, "max_abs_inverse_error": 1e-6},
            {"seed": 2, "offdiag_ratio": 0.3, "max_abs_inverse_error": 1e-6},
        ]
    )
    assert ranked[0]["seed"] == 2
