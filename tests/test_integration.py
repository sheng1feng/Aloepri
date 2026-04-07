import torch

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS
from src.key_manager import generate_vocab_permutation, invert_permutation, ordinary_token_ids
from src.model_loader import load_model_and_tokenizer, set_global_seed, tokenize_prompt
from src.obfuscate_embed_head import build_vocab_permuted_model
from src.transforms import map_input_ids, restore_logits


def test_single_step_equivalence() -> None:
    set_global_seed(20260323)
    tokenizer, model = load_model_and_tokenizer(DEFAULT_MODEL_DIR, device="cpu", dtype="auto")
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    perm_vocab = generate_vocab_permutation(
        vocab_size=model_vocab_size,
        seed=20260323,
        movable_ids=ordinary_token_ids(tokenizer),
    )
    inv_perm_vocab = invert_permutation(perm_vocab)
    permuted_model = build_vocab_permuted_model(model, perm_vocab, inv_perm_vocab)

    encoded = tokenize_prompt(tokenizer, DEFAULT_PROMPTS[0], device="cpu")
    baseline_logits = model(**encoded).logits[0, -1].detach().cpu().to(torch.float32)

    encoded["input_ids"] = map_input_ids(encoded["input_ids"], perm_vocab)
    perm_logits = permuted_model(**encoded).logits[0, -1].detach().cpu().to(torch.float32)
    restored_logits = restore_logits(perm_logits, perm_vocab)

    assert torch.equal(baseline_logits, restored_logits)

