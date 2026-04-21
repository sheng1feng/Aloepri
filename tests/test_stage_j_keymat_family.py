from src.stage_j_keymat_family import build_diag_friendly_keymat_transform
from src.stage_j_keymat_search import evaluate_keymat_candidate
from src.stage_f import build_default_stage_f_keymat


def test_diag_friendly_keymat_transform_has_diagonal_friendly_metric() -> None:
    transform = build_diag_friendly_keymat_transform(
        hidden_size=16,
        expansion_size=4,
        seed=123,
    )
    q = transform.inverse
    metric = q @ q.T
    diag = metric.diag()
    offdiag = metric - diag.diag()
    ratio = float(offdiag.norm().item() / metric.norm().item())
    assert ratio < 1e-4


def test_default_algorithm1_candidate_is_less_norm_friendly_than_diag_friendly() -> None:
    baseline = evaluate_keymat_candidate(
        hidden_size=16,
        expansion_size=4,
        lam=0.3,
        seed=123,
    )
    structured = evaluate_keymat_candidate(
        hidden_size=16,
        expansion_size=4,
        lam=0.3,
        seed=123,
        family="diag_friendly",
    )
    assert structured["offdiag_ratio"] < baseline["offdiag_ratio"]


def test_build_default_stage_f_keymat_supports_diag_friendly_family() -> None:
    class _Baseline:
        class config:
            hidden_size = 16

    transform = build_default_stage_f_keymat(
        _Baseline(),
        lam=0.3,
        h=4,
        seed=123,
        family="diag_friendly",
    )
    q = transform.inverse
    metric = q @ q.T
    ratio = float((metric - metric.diag().diag()).norm().item() / metric.norm().item())
    assert ratio < 1e-4
