import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_SEED
from src.key_manager import ordinary_token_ids
from src.llama_local_dev import build_mock_llama_from_local_metadata, mock_llama_summary
from src.stage_b import prepare_stage_a_model
from src.stage_i_vllm import export_stage_i_vllm_checkpoint, summarize_token_partitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a mock Llama Stage-I standard checkpoint from local metadata.")
    parser.add_argument("--model-dir", default="model/Llama-3.2-3B")
    parser.add_argument("--export-dir", default="artifacts/stage_i_llama_mock")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, baseline_model = build_mock_llama_from_local_metadata(args.model_dir, seed=args.seed)
    stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, seed=args.seed)
    metadata = {
        "stage": "I",
        "variant": "llama_mock_stage_a_checkpoint",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "movable_token_count": int(ordinary_token_ids(tokenizer).numel()),
        **mock_llama_summary(tokenizer, baseline_model),
        **summarize_token_partitions(
            tokenizer=tokenizer,
            model_vocab_size=stage_a_model.get_input_embeddings().weight.shape[0],
            perm_vocab=perm_vocab,
        ),
    }
    paths = export_stage_i_vllm_checkpoint(
        args.export_dir,
        tokenizer=tokenizer,
        stage_a_model=stage_a_model,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        metadata=metadata,
    )
    print(f"Saved mock Llama Stage-I checkpoint to {paths['server_dir']}")
    print(f"Saved client secret to {paths['client_secret_path']}")


if __name__ == "__main__":
    main()
