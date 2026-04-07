import torch

from src.stage_i_square import build_square_monomial_transform
from src.stage_j_block0 import (
    adapt_input_linear_weight_for_square,
    adapt_output_linear_weight_for_square,
    permute_rmsnorm_weight_for_square,
)


def test_square_norm_weight_permutation_preserves_shape() -> None:
    transform = build_square_monomial_transform(hidden_size=8, seed=11, global_scale=1.0)
    weight = torch.arange(8, dtype=torch.float32)
    permuted = permute_rmsnorm_weight_for_square(weight, transform)
    assert list(permuted.shape) == [8]


def test_square_linear_adaptation_preserves_shapes() -> None:
    transform = build_square_monomial_transform(hidden_size=8, seed=22, global_scale=1.0)
    in_proj = torch.randn(4, 8)
    out_proj = torch.randn(8, 4)
    adapted_in = adapt_input_linear_weight_for_square(in_proj, transform)
    adapted_out = adapt_output_linear_weight_for_square(out_proj, transform)
    assert list(adapted_in.shape) == [4, 8]
    assert list(adapted_out.shape) == [8, 4]
