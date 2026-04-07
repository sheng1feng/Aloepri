import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_SEED
from src.key_manager import ordinary_token_ids
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import prepare_stage_a_model
from src.stage_i_vllm import export_stage_i_vllm_checkpoint, summarize_token_partitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a real Llama Stage-I standard checkpoint.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--export-dir", default="artifacts/stage_i_llama_real")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device=args.device, dtype=args.dtype)
    stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, seed=args.seed)
    metadata = {
        "stage": "I",
        "variant": "llama_real_stage_a_checkpoint",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
        "movable_token_count": int(ordinary_token_ids(tokenizer).numel()),
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
    print(f"Saved real Llama Stage-I checkpoint to {paths['server_dir']}")
    print(f"Saved client secret to {paths['client_secret_path']}")


if __name__ == "__main__":
    main()
