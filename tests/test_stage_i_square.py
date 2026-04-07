import torch

from src.stage_i_square import (
    build_square_monomial_transform,
    check_square_monomial_inverse,
    obfuscate_embedding_with_square_transform,
    obfuscate_head_with_square_transform,
)


def test_square_monomial_transform_is_invertible() -> None:
    transform = build_square_monomial_transform(hidden_size=8, seed=123, global_scale=1.0)
    metrics = check_square_monomial_inverse(transform)
    assert metrics["passes_tolerance"] is True
    assert float(metrics["max_abs_error"]) < 1e-8


def test_square_monomial_embed_head_obfuscation_preserves_shape() -> None:
    transform = build_square_monomial_transform(hidden_size=8, seed=321, global_scale=1.0)
    embed = torch.randn(10, 8)
    head = torch.randn(10, 8)
    embed_obf = obfuscate_embedding_with_square_transform(embed, transform, alpha_e=0.0, seed=1)
    head_obf = obfuscate_head_with_square_transform(head, transform, alpha_h=0.0, seed=2)
    assert list(embed_obf.shape) == [10, 8]
    assert list(head_obf.shape) == [10, 8]
