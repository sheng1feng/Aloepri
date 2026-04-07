import torch

from src.attention_keys import (
    build_attention_complex_config,
    generate_block_perm,
    generate_h_qk,
    generate_r_qk,
    generate_tau_group,
    generate_tau_kv,
)


def test_r_qk_respects_2x2_blocks() -> None:
    matrix = generate_r_qk(8, seed=7)
    half = 4
    for i in range(half):
        idx = torch.tensor([i, i + half])
        block = matrix.index_select(0, idx).index_select(1, idx)
        assert torch.allclose(block.T @ block, torch.eye(2), atol=1e-5)
    assert torch.allclose(matrix.T @ matrix, torch.eye(8), atol=1e-5)


def test_h_qk_is_paired() -> None:
    matrix = generate_h_qk(8, (0.95, 1.05), seed=11)
    diag = torch.diag(matrix)
    half = diag.numel() // 2
    assert torch.equal(diag[:half], diag[half:])


def test_block_perm_is_local_windowed() -> None:
    block_perm, matrix = generate_block_perm(num_blocks=8, beta=2, gamma=1e3, rope_base=10000.0, seed=13)
    assert matrix.shape == (16, 16)
    for start in range(0, 8, 2):
        window = block_perm[start : start + 2].tolist()
        assert all(start <= idx < min(start + 2, 8) for idx in window)


def test_tau_permutations_are_valid() -> None:
    tau_kv, inv_tau_kv = generate_tau_kv(2, seed=17)
    tau_group, inv_tau_group = generate_tau_group(7, seed=19)
    assert torch.equal(inv_tau_kv[tau_kv], torch.arange(2))
    assert torch.equal(inv_tau_group[tau_group], torch.arange(7))


def test_build_attention_complex_config() -> None:
    config = build_attention_complex_config(
        profile="rqk_hqk_block_taukv_taugroup",
        head_dim=64,
        num_kv_heads=2,
        num_groups=7,
        seed=23,
        beta=4,
    )
    assert config.intra_head.q_matrix.shape == (64, 64)
    assert config.inter_head.tau_kv is not None
    assert config.inter_head.tau_group is not None
