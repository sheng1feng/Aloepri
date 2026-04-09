from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_SEED
from src.security_qwen import build_vma_comparison_payload, default_vma_gate1_targets, run_vma_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Gate-1 VMA baseline across a comparison target set.")
    parser.add_argument("--baseline-model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--candidate-pool-size", type=int, default=2048)
    parser.add_argument("--feature-bins", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--layer-indices", nargs="*", type=int, default=None)
    parser.add_argument("--disable-projections", action="store_true")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=default_vma_gate1_targets(),
        help="Security target names. Defaults to the Gate-1 baseline comparison set.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/security_qwen/summary/vma_comparison.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_payloads: list[dict] = []
    for target_name in args.targets:
        payload = run_vma_baseline(
            target_name=target_name,
            baseline_model_dir=args.baseline_model_dir,
            seed=args.seed,
            eval_size=args.eval_size,
            candidate_pool_size=args.candidate_pool_size,
            feature_bins=args.feature_bins,
            topk=args.topk,
            layer_indices=tuple(args.layer_indices) if args.layer_indices else None,
            use_projection_sources=not args.disable_projections,
        )
        result_payloads.append(payload)
        per_target_path = Path(f"outputs/security_qwen/vma/{target_name}.json")
        per_target_path.parent.mkdir(parents=True, exist_ok=True)
        per_target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved VMA result to {per_target_path}")

    comparison = build_vma_comparison_payload(result_payloads=result_payloads)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved VMA comparison to {output_path}")


if __name__ == "__main__":
    main()
