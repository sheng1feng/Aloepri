from types import SimpleNamespace

from src.aloepri import AloePriConfig, AloePriEngine
from src.aloepri.adapters import (
    LlamaArchitectureAdapter,
    build_architecture_config,
    build_llama_config,
    get_architecture_adapter,
    is_llama_compatible_model,
    is_qwen_compatible_model,
)


class _Linear:
    def __init__(self):
        self.weight = object()
        self.bias = None


class _Layer:
    def __init__(self):
        self.self_attn = SimpleNamespace(
            q_proj=_Linear(),
            k_proj=_Linear(),
            v_proj=_Linear(),
            o_proj=_Linear(),
        )
        self.input_layernorm = object()
        self.post_attention_layernorm = object()
        self.mlp = SimpleNamespace(
            gate_proj=_Linear(),
            up_proj=_Linear(),
            down_proj=_Linear(),
        )


class _Embed:
    def __init__(self, ptr: int):
        self._ptr = ptr

    @property
    def weight(self):
        ptr = self._ptr

        class _W:
            def data_ptr(self_nonlocal):
                return ptr

        return _W()


class _Head(_Embed):
    pass


class _LlamaModel:
    def __init__(self):
        self.config = SimpleNamespace(
            hidden_size=3072,
            num_hidden_layers=28,
            num_attention_heads=24,
            num_key_value_heads=8,
            head_dim=128,
            rope_theta=500000.0,
            model_type="llama",
        )
        self.model = SimpleNamespace(
            embed_tokens=_Embed(1),
            layers=[_Layer()],
            norm=object(),
        )
        self.lm_head = _Head(2)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class _Tokenizer:
    vocab_size = 128256

    def get_added_vocab(self):
        return {}

    all_special_ids = [128000, 128001]


def test_llama_adapter_accepts_structurally_compatible_model() -> None:
    model = _LlamaModel()
    adapter = LlamaArchitectureAdapter.from_model(model)
    assert adapter.hidden_size == 3072
    assert adapter.num_hidden_layers == 28
    assert adapter.num_attention_heads == 24
    assert adapter.num_key_value_heads == 8
    assert adapter.head_dim == 128
    assert adapter.rope_theta == 500000.0
    assert is_llama_compatible_model(model) is True
    assert is_qwen_compatible_model(model) is False


def test_build_llama_config_uses_model_structure() -> None:
    model = _LlamaModel()
    config = build_llama_config(model, adapted_layers=[0, 1], alpha_e=0.0, alpha_h=0.0)
    assert config.hidden_size == 3072
    assert config.num_hidden_layers == 28
    assert config.head_dim == 128
    assert config.adapted_layers == [0, 1]
    assert config.architecture_family == "llama_decoder"
    assert config.model_type == "llama"


def test_architecture_dispatch_selects_llama() -> None:
    model = _LlamaModel()
    adapter = get_architecture_adapter(model)
    config = build_architecture_config(model, alpha_e=0.0, alpha_h=0.0)
    assert isinstance(adapter, LlamaArchitectureAdapter)
    assert config.architecture_family == "llama_decoder"


def test_aloepri_config_and_engine_accept_llama() -> None:
    model = _LlamaModel()
    config = AloePriConfig.from_model(model, alpha_e=0.0, alpha_h=0.0)
    assert config.architecture_family == "llama_decoder"
    engine = AloePriEngine.from_model(model, _Tokenizer(), alpha_e=0.0, alpha_h=0.0)
    assert engine.config.architecture_family == "llama_decoder"
