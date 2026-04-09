from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_SEED
from src.security_qwen import build_vma_source_attribution_payload, run_vma_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate-1 VMA source attribution on a single target.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--baseline-model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--candidate-pool-size", type=int, default=2048)
    parser.add_argument("--feature-bins", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output-path", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = [
        ("direct_only", dict(use_projection_sources=False, include_direct_sources=True)),
        ("projection_only_q", dict(use_projection_sources=True, include_direct_sources=False, projection_kinds=("q",))),
        ("projection_only_k", dict(use_projection_sources=True, include_direct_sources=False, projection_kinds=("k",))),
        ("projection_only_v", dict(use_projection_sources=True, include_direct_sources=False, projection_kinds=("v",))),
        ("projection_only_gate", dict(use_projection_sources=True, include_direct_sources=False, projection_kinds=("gate",))),
        ("projection_only_up", dict(use_projection_sources=True, include_direct_sources=False, projection_kinds=("up",))),
        ("projection_all", dict(use_projection_sources=True, include_direct_sources=False)),
        ("direct_plus_projection_all", dict(use_projection_sources=True, include_direct_sources=True)),
    ]

    payloads: list[dict] = []
    for label, kwargs in cases:
        payload = run_vma_baseline(
            target_name=args.target,
            baseline_model_dir=args.baseline_model_dir,
            seed=args.seed,
            eval_size=args.eval_size,
            candidate_pool_size=args.candidate_pool_size,
            feature_bins=args.feature_bins,
            topk=args.topk,
            **kwargs,
        )
        payload["config"]["attribution_label"] = label
        payloads.append(payload)

    summary = build_vma_source_attribution_payload(target_name=args.target, result_payloads=payloads)
    output_path = Path(args.output_path or f"outputs/security_qwen/vma/{args.target}.source_attribution.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved VMA source attribution to {output_path}")


if __name__ == "__main__":
    main()
