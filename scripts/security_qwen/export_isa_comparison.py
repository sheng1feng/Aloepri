from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_SEED
from src.security_qwen import build_isa_comparison_payload, default_isa_gate3_targets, run_isa_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Gate-3 ISA baseline across a comparison target set.")
    parser.add_argument("--observable-type", default="hidden_state", choices=["hidden_state", "attention_score"])
    parser.add_argument("--observable-layer", type=int, default=23)
    parser.add_argument("--baseline-model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--train-sequences", type=int, default=64)
    parser.add_argument("--val-sequences", type=int, default=16)
    parser.add_argument("--test-sequences", type=int, default=16)
    parser.add_argument("--candidate-pool-size", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--ridge-alphas", nargs="*", type=float, default=[1e-4, 1e-2, 1.0])
    parser.add_argument("--targets", nargs="*", default=default_isa_gate3_targets())
    parser.add_argument("--output-path", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payloads: list[dict] = []
    for target_name in args.targets:
        payload = run_isa_baseline(
            target_name=target_name,
            observable_type=args.observable_type,
            observable_layer=args.observable_layer,
            baseline_model_dir=args.baseline_model_dir,
            seed=args.seed,
            sequence_length=args.sequence_length,
            train_sequences=args.train_sequences,
            val_sequences=args.val_sequences,
            test_sequences=args.test_sequences,
            candidate_pool_size=args.candidate_pool_size,
            topk=args.topk,
            ridge_alphas=tuple(float(item) for item in args.ridge_alphas),
        )
        payloads.append(payload)
        per_target_path = Path(f"outputs/security_qwen/isa/{target_name}.{args.observable_type}.json")
        per_target_path.parent.mkdir(parents=True, exist_ok=True)
        per_target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved ISA result to {per_target_path}")

    comparison = build_isa_comparison_payload(result_payloads=payloads, observable_type=args.observable_type)
    default_output = f"outputs/security_qwen/summary/isa_{args.observable_type}_comparison.json"
    output_path = Path(args.output_path or default_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved ISA comparison to {output_path}")


if __name__ == "__main__":
    main()
