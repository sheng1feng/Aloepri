from types import SimpleNamespace

from transformers import LlamaConfig, LlamaForCausalLM

from src.stage_j_block0 import build_stage_j_square_model


class _TokenizerStub:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.all_special_ids = [0, 1]

    def get_added_vocab(self):
        return {}


def test_stage_j_square_model_builds_on_tiny_llama() -> None:
    tokenizer = _TokenizerStub(vocab_size=256)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config).eval()

    stage_model, perm_vocab, inv_perm_vocab, transform = build_stage_j_square_model(
        baseline_model=model,
        tokenizer=tokenizer,
        adapted_layers=[0, 1],
        seed=20260323,
        alpha_e=0.0,
        alpha_h=0.0,
    )

    assert stage_model.config.model_type == "llama"
    assert stage_model.model.embed_tokens.weight.shape == model.model.embed_tokens.weight.shape
    assert stage_model.lm_head.weight.shape == model.lm_head.weight.shape
    assert perm_vocab.shape[0] == config.vocab_size
    assert inv_perm_vocab.shape[0] == config.vocab_size
    assert transform.dim == config.hidden_size
