from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_SEED
from src.security_qwen import run_vma_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal Gate-1 VMA baseline.")
    parser.add_argument("--target", required=True, help="Security target name, e.g. stage_j_stable_reference.")
    parser.add_argument("--baseline-model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--candidate-pool-size", type=int, default=4096)
    parser.add_argument("--feature-bins", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--layer-indices", nargs="*", type=int, default=None)
    parser.add_argument("--disable-projections", action="store_true")
    parser.add_argument("--output-path", default="")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    payload = run_vma_baseline(
        target_name=args.target,
        baseline_model_dir=args.baseline_model_dir,
        seed=args.seed,
        eval_size=args.eval_size,
        candidate_pool_size=args.candidate_pool_size,
        feature_bins=args.feature_bins,
        topk=args.topk,
        layer_indices=tuple(args.layer_indices) if args.layer_indices else None,
        use_projection_sources=not args.disable_projections,
    )
    output_path = args.output_path or f"outputs/security_qwen/vma/{args.target}.json"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved VMA result to {output}")
