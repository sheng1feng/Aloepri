from types import SimpleNamespace

from src.aloepri.adapters.qwen import QwenArchitectureAdapter, build_qwen_config, is_qwen_compatible_model


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
    def __init__(self):
        self._ptr = 1

    @property
    def weight(self):
        class _W:
            def data_ptr(self_nonlocal):
                return 1

        return _W()


class _Head:
    def __init__(self):
        self._ptr = 2

    @property
    def weight(self):
        class _W:
            def data_ptr(self_nonlocal):
                return 2

        return _W()


class _Model:
    def __init__(self):
        self.config = SimpleNamespace(
            hidden_size=896,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            rope_theta=10000.0,
            model_type="qwen2",
        )
        self.model = SimpleNamespace(
            embed_tokens=_Embed(),
            layers=[_Layer()],
            norm=object(),
        )
        self.lm_head = _Head()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_qwen_adapter_accepts_structurally_compatible_model() -> None:
    model = _Model()
    adapter = QwenArchitectureAdapter.from_model(model)
    assert adapter.hidden_size == 896
    assert adapter.num_hidden_layers == 24
    assert adapter.num_attention_heads == 14
    assert adapter.num_key_value_heads == 2
    assert adapter.head_dim == 64
    assert is_qwen_compatible_model(model) is True


def test_build_qwen_config_uses_model_structure() -> None:
    model = _Model()
    config = build_qwen_config(model, adapted_layers=[0, 1], alpha_e=0.0, alpha_h=0.0)
    assert config.hidden_size == 896
    assert config.num_hidden_layers == 24
    assert config.head_dim == 64
    assert config.adapted_layers == [0, 1]
    assert config.architecture_family == "qwen_decoder"
    assert config.model_type == "qwen2"


def test_qwen_adapter_rejects_non_qwen_model_type() -> None:
    model = _Model()
    model.config.model_type = "llama"
    assert is_qwen_compatible_model(model) is False
