import torch

from src.security_qwen.vma import _sorted_quantile_features, _topk_hits


def test_sorted_quantile_features_is_row_normalized() -> None:
    matrix = torch.tensor([[3.0, 1.0, 2.0], [4.0, 2.0, 0.0]], dtype=torch.float32)
    features = _sorted_quantile_features(matrix, bins=3)
    assert features.shape == (2, 3)
    assert torch.allclose(features[0].norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(features[1].norm(), torch.tensor(1.0), atol=1e-5)


def test_topk_hits_recovers_true_ids_from_score_matrix() -> None:
    score_matrix = torch.tensor([[0.2, 0.9, 0.1], [0.8, 0.1, 0.3]], dtype=torch.float32)
    candidate_plain_ids = torch.tensor([10, 11, 12], dtype=torch.long)
    true_plain_ids = torch.tensor([11, 10], dtype=torch.long)
    top1, hits = _topk_hits(score_matrix, candidate_plain_ids, true_plain_ids, topk=2)
    assert top1.tolist() == [11, 10]
    assert hits[:, 0].tolist() == [True, True]
