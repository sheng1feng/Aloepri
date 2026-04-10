from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_SEED
from src.security_qwen import run_ima_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal Gate-2 IMA baseline.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--baseline-model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--candidate-pool-size", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--ridge-alphas", nargs="*", type=float, default=[1e-4, 1e-2, 1.0])
    parser.add_argument("--output-path", default="")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    payload = run_ima_baseline(
        target_name=args.target,
        baseline_model_dir=args.baseline_model_dir,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        candidate_pool_size=args.candidate_pool_size,
        topk=args.topk,
        ridge_alphas=tuple(float(item) for item in args.ridge_alphas),
    )
    output_path = Path(args.output_path or f"outputs/security_qwen/ima/{args.target}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved IMA result to {output_path}")
