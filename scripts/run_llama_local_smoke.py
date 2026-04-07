from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from src.aloepri import AloePriConfig, AloePriEngine
from src.llama_local_dev import build_mock_llama_from_local_metadata
from src.stage_b import prepare_stage_a_model
from src.stage_i_vllm import export_stage_i_vllm_checkpoint, load_stage_i_hf_bundle, summarize_token_partitions
from src.stage_j_block0 import build_stage_j_square_model


def main() -> None:
    model_dir = Path("model/Llama-3.2-3B")
    output_path = Path("outputs/llama_local_smoke.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = build_mock_llama_from_local_metadata(model_dir)
    aloepri_config = AloePriConfig.from_model(model, alpha_e=0.0, alpha_h=0.0, adapted_layers=[0, 1])
    engine = AloePriEngine.from_model(model, tokenizer, alpha_e=0.0, alpha_h=0.0, adapted_layers=[0, 1])

    stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(model, tokenizer, seed=20260323)
    metadata = summarize_token_partitions(
        tokenizer=tokenizer,
        model_vocab_size=stage_a_model.config.vocab_size,
        perm_vocab=perm_vocab,
    )

    with TemporaryDirectory() as export_dir:
        export_paths = export_stage_i_vllm_checkpoint(
            export_dir,
            tokenizer=tokenizer,
            stage_a_model=stage_a_model,
            perm_vocab=perm_vocab,
            inv_perm_vocab=inv_perm_vocab,
            metadata=metadata,
        )
        loaded = load_stage_i_hf_bundle(export_paths["server_dir"])

    stage_j_model, _, _, transform = build_stage_j_square_model(
        baseline_model=model,
        tokenizer=tokenizer,
        adapted_layers=[0, 1],
        seed=20260323,
        alpha_e=0.0,
        alpha_h=0.0,
    )

    payload = {
        "phase": "llama_local_smoke",
        "model_dir": str(model_dir),
        "mock_model": {
            "hidden_size": int(model.config.hidden_size),
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "num_attention_heads": int(model.config.num_attention_heads),
            "num_key_value_heads": int(model.config.num_key_value_heads),
            "vocab_size": int(model.config.vocab_size),
        },
        "adapter": {
            "architecture_family": aloepri_config.architecture_family,
            "model_type": aloepri_config.model_type,
            "engine_architecture_family": engine.config.architecture_family,
        },
        "stage_i_export": {
            "loaded_model_type": loaded["model"].config.model_type,
            "perm_vocab_loaded": loaded["perm_vocab"] is not None,
            "special_ids_fixed": metadata["special_ids_fixed"],
            "tail_rows_fixed": metadata["tail_rows_fixed"],
        },
        "stage_j_square": {
            "embed_shape": list(stage_j_model.model.embed_tokens.weight.shape),
            "lm_head_shape": list(stage_j_model.lm_head.weight.shape),
            "transform_hidden_size": int(transform.dim),
        },
        "status": "ok",
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
