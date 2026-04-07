import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import TraceRecorder
from src.stage_f import build_default_stage_f_keymat, build_layer_stage_f_configs, calibrate_keymat_kappas
from src.stage_h import build_layer_stage_h_configs, build_stage_h_model
from src.stage_h_artifact import save_stage_h_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a full-layer stage-H obfuscated model artifact.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--artifact-dir", default="artifacts/stage_h_full_obfuscated")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.3)
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--attention-profile", default="rqk_hqk_block_taukv_taugroup")
    parser.add_argument("--alpha-e", type=float, default=0.1)
    parser.add_argument("--alpha-h", type=float, default=0.05)
    parser.add_argument("--beta", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=1e3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    adapted_layers = list(range(baseline_model.config.num_hidden_layers))
    keymat_transform = build_default_stage_f_keymat(baseline_model, lam=args.lam, h=args.h, seed=args.seed)
    kappas = calibrate_keymat_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS[:2],
        keymat_transform=keymat_transform,
        trace_layers=adapted_layers,
    )
    layer_configs_f = build_layer_stage_f_configs(
        baseline_model=baseline_model,
        keymat_transform=keymat_transform,
        trace_layers=adapted_layers,
        kappa_by_layer=kappas,
        attention_profile=args.attention_profile,
        seed=args.seed,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
    )
    layer_configs_h = build_layer_stage_h_configs(layer_configs_f)
    stage_model, perm_vocab, inv_perm_vocab = build_stage_h_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        keymat_transform=keymat_transform,
        seed=args.seed,
        recorder=TraceRecorder(),
        layer_configs=layer_configs_h,
        adapted_layers=adapted_layers,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
        use_keymat_head=True,
        beta=args.beta,
        gamma=args.gamma,
    )
    metadata = {
        "model_dir": args.model_dir,
        "dtype": args.dtype,
        "seed": args.seed,
        "lambda": args.lam,
        "h": args.h,
        "attention_profile": args.attention_profile,
        "alpha_e": args.alpha_e,
        "alpha_h": args.alpha_h,
        "beta": args.beta,
        "gamma": args.gamma,
        "adapted_layers": adapted_layers,
        "prompts_for_kappa": DEFAULT_PROMPTS[:2],
    }
    save_stage_h_artifact(
        args.artifact_dir,
        stage_model=stage_model,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        metadata=metadata,
    )
    print(f"Saved stage-H artifact to {args.artifact_dir}")


if __name__ == "__main__":
    main()
