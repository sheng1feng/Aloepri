import torch

from src.hidden_keys import build_hidden_transform
from src.transforms import map_input_ids, restore_logits, unmap_output_ids
from src.transforms import apply_hidden_transform, apply_inverse_hidden_transform


def test_map_input_ids() -> None:
    perm = torch.tensor([2, 0, 1, 3], dtype=torch.long)
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    mapped = map_input_ids(input_ids, perm)
    assert torch.equal(mapped, torch.tensor([[2, 0, 1, 3]], dtype=torch.long))


def test_unmap_output_ids_round_trip() -> None:
    perm = torch.tensor([2, 0, 1, 3], dtype=torch.long)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    mapped = map_input_ids(input_ids, perm)
    restored = unmap_output_ids(mapped, inv)
    assert torch.equal(restored, input_ids)


def test_restore_logits_toy_case() -> None:
    perm = torch.tensor([2, 0, 1], dtype=torch.long)
    logits_perm = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    restored = restore_logits(logits_perm, perm)
    expected = torch.tensor([30.0, 10.0, 20.0], dtype=torch.float32)
    assert torch.equal(restored, expected)


def test_apply_hidden_transform_last_dim_only() -> None:
    perm = torch.tensor([1, 2, 0], dtype=torch.long)
    scale = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    transform = build_hidden_transform(perm, scale)
    hidden = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    obf = apply_hidden_transform(hidden, transform)
    restored = apply_inverse_hidden_transform(obf, transform)
    assert torch.allclose(restored, hidden)
