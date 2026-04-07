import torch

from src.obfuscate_embed_head import permute_embedding_weight, permute_lm_head_weight


def test_permute_embedding_weight() -> None:
    weight = torch.tensor(
        [
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
            [4.0, 4.1, 4.2],
        ]
    )
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted = permute_embedding_weight(weight, perm)
    for original_id in range(weight.shape[0]):
        assert torch.equal(permuted[perm[original_id]], weight[original_id])


def test_permute_lm_head_weight() -> None:
    weight = torch.tensor(
        [
            [5.0, 5.1],
            [6.0, 6.1],
            [7.0, 7.1],
            [8.0, 8.1],
        ]
    )
    perm = torch.tensor([1, 3, 0, 2], dtype=torch.long)
    permuted = permute_lm_head_weight(weight, perm)
    for original_id in range(weight.shape[0]):
        assert torch.equal(permuted[perm[original_id]], weight[original_id])

