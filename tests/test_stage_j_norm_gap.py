from __future__ import annotations

import torch

from src.stage_j_norm_gap import summarize_metric_matrix


def test_summarize_metric_matrix_marks_diagonal_matrix_as_bridgeable() -> None:
    matrix = torch.diag(torch.tensor([4.0, 9.0, 16.0], dtype=torch.float32))
    payload = summarize_metric_matrix(matrix)
    assert payload["offdiag_ratio"] == 0.0
    assert payload["standard_rmsnorm_equivalent"] is True


def test_summarize_metric_matrix_marks_dense_matrix_as_not_bridgeable() -> None:
    matrix = torch.ones((3, 3), dtype=torch.float32)
    payload = summarize_metric_matrix(matrix)
    assert payload["offdiag_ratio"] > 0.0
    assert payload["standard_rmsnorm_equivalent"] is False
