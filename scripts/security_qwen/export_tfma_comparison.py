from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_SEED
from src.security_qwen import build_tfma_comparison_payload, default_frequency_gate4_targets, run_tfma_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate-4 TFMA across all knowledge settings and targets.")
    parser.add_argument("--baseline-model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--targets", nargs="*", default=default_frequency_gate4_targets())
    parser.add_argument("--output-path", default="outputs/security_qwen/summary/tfma_comparison.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payloads: list[dict] = []
    for knowledge_setting in ["zero_knowledge", "domain_aware", "distribution_aware"]:
        for target_name in args.targets:
            payload = run_tfma_baseline(
                target_name=target_name,
                knowledge_setting=knowledge_setting,
                baseline_model_dir=args.baseline_model_dir,
                seed=args.seed,
                candidate_pool_size=args.candidate_pool_size,
                topk=args.topk,
            )
            payloads.append(payload)
            out = Path(f"outputs/security_qwen/tfma/{target_name}.{knowledge_setting}.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved TFMA result to {out}")
    comparison = build_tfma_comparison_payload(result_payloads=payloads)
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved TFMA comparison to {out}")


if __name__ == "__main__":
    main()
