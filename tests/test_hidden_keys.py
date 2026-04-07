import torch

from src.hidden_keys import (
    build_hidden_transform,
    generate_hidden_permutation,
    generate_hidden_scaling,
    invert_hidden_transform,
    validate_hidden_transform,
)
from src.transforms import apply_hidden_transform, apply_inverse_hidden_transform


def test_hidden_transform_invertibility() -> None:
    perm = generate_hidden_permutation(8, seed=7)
    scale = generate_hidden_scaling(8, (0.95, 1.05), seed=11)
    transform = build_hidden_transform(perm, scale)
    inv_transform = invert_hidden_transform(transform)
    assert validate_hidden_transform(transform, inv_transform)


def test_hidden_transform_round_trip_toy() -> None:
    perm = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    scale = torch.tensor([1.0, 0.9, 1.1, 1.2], dtype=torch.float32)
    transform = build_hidden_transform(perm, scale)
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    obf = apply_hidden_transform(hidden, transform)
    restored = apply_inverse_hidden_transform(obf, transform)
    assert torch.allclose(restored, hidden)

