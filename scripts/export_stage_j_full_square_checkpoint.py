import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_SEED
from src.key_manager import ordinary_token_ids
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_i_vllm import export_stage_i_vllm_checkpoint, summarize_token_partitions
from src.stage_j_block0 import build_stage_j_square_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage-J full-layer square-transform checkpoint.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--export-dir", default="artifacts/stage_j_full_square")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--alpha-e", type=float, default=0.0)
    parser.add_argument("--alpha-h", type=float, default=0.0)
    parser.add_argument("--global-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    adapted_layers = list(range(baseline_model.config.num_hidden_layers))
    stage_model, perm_vocab, inv_perm_vocab, transform = build_stage_j_square_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        adapted_layers=adapted_layers,
        seed=args.seed,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
        global_scale=args.global_scale,
        recorder=None,
    )
    metadata = {
        "stage": "J",
        "variant": "full_layer_square_transform",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "dtype": args.dtype,
        "alpha_e": args.alpha_e,
        "alpha_h": args.alpha_h,
        "global_scale": args.global_scale,
        "adapted_layers": adapted_layers,
        "hidden_size": int(baseline_model.config.hidden_size),
        "square_transform": {
            "perm": transform.perm.tolist(),
            "inv_perm": transform.inv_perm.tolist(),
            "signs": transform.signs.tolist(),
            "global_scale": transform.global_scale,
        },
        "movable_token_count": int(ordinary_token_ids(tokenizer).numel()),
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
    print(f"Saved Stage-J full-layer square checkpoint to {paths['server_dir']}")
    print(f"Saved client secret to {paths['client_secret_path']}")


if __name__ == "__main__":
    main()
