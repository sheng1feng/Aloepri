from src.stage_j_attention_gap import summarize_attention_metadata

import torch


def test_summarize_attention_metadata_detects_non_identity_structure() -> None:
    payload = summarize_attention_metadata(
        q_feature_inv_order=torch.tensor([1, 0]),
        kv_feature_inv_order=torch.tensor([0, 1]),
        q_dense_inverse=torch.tensor([[1.0, 0.2], [0.0, 1.0]]),
        k_dense_inverse=torch.eye(2),
    )
    assert payload["q_order_is_identity"] is False
    assert payload["q_dense_identity_max_abs_error"] > 0.0
