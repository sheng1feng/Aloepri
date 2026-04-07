import torch
from transformers import AutoTokenizer

from src.defaults import DEFAULT_MODEL_DIR
from src.key_manager import generate_vocab_permutation, invert_permutation, ordinary_token_ids, validate_permutation


def test_invert_permutation() -> None:
    perm = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long)
    inv = invert_permutation(perm)
    assert torch.equal(inv[perm], torch.arange(perm.numel()))
    assert torch.equal(perm[inv], torch.arange(perm.numel()))


def test_validate_permutation() -> None:
    assert validate_permutation(torch.tensor([1, 2, 0], dtype=torch.long))
    assert not validate_permutation(torch.tensor([1, 1, 0], dtype=torch.long))


def test_special_and_added_tokens_fixed() -> None:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR, trust_remote_code=True)
    model_vocab_size = len(tokenizer) + 16
    movable_ids = ordinary_token_ids(tokenizer)
    perm = generate_vocab_permutation(model_vocab_size, seed=1234, movable_ids=movable_ids)

    fixed_ids = set(tokenizer.get_added_vocab().values())
    fixed_ids.update(tokenizer.all_special_ids)
    fixed_ids.update(range(len(tokenizer), model_vocab_size))

    for token_id in fixed_ids:
        assert perm[token_id].item() == token_id

