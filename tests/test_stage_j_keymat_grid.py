from src.stage_j_keymat_grid import evaluate_keymat_grid, rank_keymat_grid_rows


def test_rank_keymat_grid_rows_prefers_lower_offdiag_ratio() -> None:
    ranked = rank_keymat_grid_rows(
        [
            {"family": "algorithm1", "lam": 0.3, "expansion_size": 128, "best_offdiag_ratio": 0.9},
            {"family": "diag_friendly", "lam": 0.1, "expansion_size": 64, "best_offdiag_ratio": 0.4},
        ]
    )
    assert ranked[0]["family"] == "diag_friendly"


def test_evaluate_keymat_grid_preserves_family_dimension() -> None:
    rows = evaluate_keymat_grid(
        hidden_size=8,
        expansion_sizes=[4],
        lams=[0.3],
        families=["algorithm1", "diag_friendly"],
        seed_start=123,
        num_candidates=1,
    )
    assert {row["family"] for row in rows} == {"algorithm1", "diag_friendly"}
