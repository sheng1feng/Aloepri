from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_SEED
from src.security_qwen import run_sda_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal Gate-4 SDA baseline.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--knowledge-setting", default="zero_knowledge", choices=["zero_knowledge", "domain_aware", "distribution_aware"])
    parser.add_argument("--baseline-model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--candidate-pool-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--output-path", default="")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    payload = run_sda_baseline(
        target_name=args.target,
        knowledge_setting=args.knowledge_setting,
        baseline_model_dir=args.baseline_model_dir,
        seed=args.seed,
        candidate_pool_size=args.candidate_pool_size,
        topk=args.topk,
    )
    output_path = Path(args.output_path or f"outputs/security_qwen/sda/{args.target}.{args.knowledge_setting}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved SDA result to {output_path}")
