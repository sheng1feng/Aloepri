import torch

from src.gqa_layout import GQALayout


def test_gqa_layout_permutations() -> None:
    layout = GQALayout(num_query_heads=14, num_kv_heads=2)
    query = torch.arange(14 * 3, dtype=torch.float32).view(1, 14, 1, 3)
    grouped = layout.reshape_query_groups(query)
    tau_kv = torch.tensor([1, 0], dtype=torch.long)
    tau_group = torch.tensor([6, 5, 4, 3, 2, 1, 0], dtype=torch.long)
    permuted = layout.permute_query_groups(grouped, tau_kv, tau_group)
    restored = layout.invert_query_groups(permuted, tau_kv, tau_group)
    assert torch.equal(layout.merge_query_groups(restored), query)
