import torch

from src.keymat import (
    build_keymat_transform,
    check_keymat_inverse,
    init_keymat_bases,
    sample_null_columns,
    sample_null_rows,
)


def test_keymat_inverse_small_matrix() -> None:
    transform = build_keymat_transform(
        d=16,
        h=4,
        lam=0.1,
        init_seed=123,
        key_seed=124,
        inv_seed=125,
    )
    metrics = check_keymat_inverse(transform.key, transform.inverse, tol=1e-5)
    assert metrics["passes_tolerance"] is True
    assert float(metrics["max_abs_error"]) < 1e-5


def test_nullspace_samples_cancel_cross_terms() -> None:
    bases = init_keymat_bases(d=16, h=4, lam=0.1, seed=321)
    c = sample_null_columns(bases.f.T, out_rows=bases.hidden_size, seed=333)
    d = sample_null_rows(bases.e, out_cols=bases.hidden_size, seed=444)
    assert torch.allclose(c @ bases.f, torch.zeros((bases.hidden_size, bases.hidden_size), dtype=torch.float64), atol=1e-8)
    assert torch.allclose(bases.e @ d, torch.zeros((bases.hidden_size, bases.hidden_size), dtype=torch.float64), atol=1e-8)
