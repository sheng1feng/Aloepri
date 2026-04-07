import torch

from src.keymat import build_keymat_transform
from src.keymat_embed_head import obfuscate_embedding_with_keymat, obfuscate_head_with_keymat


def test_keymat_embed_head_exact_round_trip_without_noise() -> None:
    embed_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    head_weight = torch.tensor(
        [
            [0.5, 0.1, -0.2],
            [-0.3, 0.4, 0.2],
            [0.1, -0.2, 0.7],
            [0.0, 0.3, -0.1],
        ],
        dtype=torch.float32,
    )
    transform = build_keymat_transform(
        d=3,
        h=2,
        lam=0.1,
        init_seed=11,
        key_seed=12,
        inv_seed=13,
    )
    embed_obf = obfuscate_embedding_with_keymat(embed_weight, transform, alpha_e=0.0, seed=0)
    head_obf = obfuscate_head_with_keymat(head_weight, transform, alpha_h=0.0, seed=0)

    token_id = 2
    hidden_base = embed_weight[token_id]
    hidden_obf = embed_obf[token_id]
    expected_hidden_obf = hidden_base @ transform.key
    assert torch.allclose(hidden_obf, expected_hidden_obf, atol=1e-6)

    logits_base = hidden_base @ head_weight.T
    logits_obf = hidden_obf @ head_obf.T
    assert torch.allclose(logits_base, logits_obf, atol=1e-5)
