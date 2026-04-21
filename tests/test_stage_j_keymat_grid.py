from src.stage_j_keymat_grid import rank_keymat_grid_rows


def test_rank_keymat_grid_rows_prefers_lower_offdiag_ratio() -> None:
    ranked = rank_keymat_grid_rows(
        [
            {"lam": 0.3, "expansion_size": 128, "best_offdiag_ratio": 0.9},
            {"lam": 0.1, "expansion_size": 64, "best_offdiag_ratio": 0.4},
        ]
    )
    assert ranked[0]["lam"] == 0.1
