import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_SEED
from src.llama_local_dev import build_mock_llama_from_local_metadata, mock_llama_summary
from src.stage_j_block0 import build_stage_j_square_model
from src.stage_i_vllm import export_stage_i_vllm_checkpoint, summarize_token_partitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a mock Llama Stage-J full-layer standard-shape checkpoint.")
    parser.add_argument("--model-dir", default="model/Llama-3.2-3B")
    parser.add_argument("--export-dir", default="artifacts/stage_j_llama_mock_full_square")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--alpha-e", type=float, default=0.0)
    parser.add_argument("--alpha-h", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, baseline_model = build_mock_llama_from_local_metadata(args.model_dir, seed=args.seed)
    adapted_layers = list(range(baseline_model.config.num_hidden_layers))
    stage_model, perm_vocab, inv_perm_vocab, transform = build_stage_j_square_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        adapted_layers=adapted_layers,
        seed=args.seed,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
    )
    metadata = {
        "stage": "J",
        "variant": "llama_mock_standard_shape_full",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "alpha_e": args.alpha_e,
        "alpha_h": args.alpha_h,
        "transform_dim": int(transform.dim),
        **mock_llama_summary(tokenizer, baseline_model),
        **summarize_token_partitions(
            tokenizer=tokenizer,
            model_vocab_size=stage_model.get_input_embeddings().weight.shape[0],
            perm_vocab=perm_vocab,
        ),
    }
    paths = export_stage_i_vllm_checkpoint(
        args.export_dir,
        tokenizer=tokenizer,
        stage_a_model=stage_model,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        metadata=metadata,
    )
    print(f"Saved mock Llama Stage-J checkpoint to {paths['server_dir']}")
    print(f"Saved client secret to {paths['client_secret_path']}")


if __name__ == "__main__":
    main()
