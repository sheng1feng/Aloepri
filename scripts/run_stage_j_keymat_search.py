from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import write_json
from src.stage_j_keymat_search import evaluate_keymat_candidate, rank_keymat_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for norm-friendlier KeyMat candidates.")
    parser.add_argument("--hidden-size", type=int, default=896)
    parser.add_argument("--expansion-size", type=int, default=128)
    parser.add_argument("--lam", type=float, default=0.3)
    parser.add_argument("--seed-start", type=int, default=20260323)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--output-path", default="outputs/stage_j/keymat_search.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        evaluate_keymat_candidate(
            hidden_size=args.hidden_size,
            expansion_size=args.expansion_size,
            lam=args.lam,
            seed=args.seed_start + offset,
        )
        for offset in range(args.num_candidates)
    ]
    ranked = rank_keymat_candidates(rows)
    payload = {
        "hidden_size": args.hidden_size,
        "expansion_size": args.expansion_size,
        "lam": args.lam,
        "seed_start": args.seed_start,
        "num_candidates": args.num_candidates,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
    }
    write_json(args.output_path, payload)
    print(f"Saved Stage-J keymat search to {args.output_path}")


if __name__ == "__main__":
    main()
