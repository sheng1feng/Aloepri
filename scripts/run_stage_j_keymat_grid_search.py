from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import write_json
from src.stage_j_keymat_grid import evaluate_keymat_grid, rank_keymat_grid_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search KeyMat norm-friendliness across h and λ.")
    parser.add_argument("--hidden-size", type=int, default=896)
    parser.add_argument("--expansion-sizes", nargs="*", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--lams", nargs="*", type=float, default=[0.0, 0.1, 0.3, 1.0])
    parser.add_argument("--seed-start", type=int, default=20260323)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--output-path", default="outputs/stage_j/keymat_grid_search.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate_keymat_grid(
        hidden_size=args.hidden_size,
        expansion_sizes=list(args.expansion_sizes),
        lams=list(args.lams),
        seed_start=args.seed_start,
        num_candidates=args.num_candidates,
    )
    ranked = rank_keymat_grid_rows(rows)
    payload = {
        "hidden_size": args.hidden_size,
        "expansion_sizes": list(args.expansion_sizes),
        "lams": list(args.lams),
        "seed_start": args.seed_start,
        "num_candidates": args.num_candidates,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
    }
    write_json(args.output_path, payload)
    print(f"Saved Stage-J keymat grid search to {args.output_path}")


if __name__ == "__main__":
    main()
