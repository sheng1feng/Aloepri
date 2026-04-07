import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_OUTPUT_DIR, DEFAULT_SEED
from src.evaluator import write_json
from src.keymat import build_keymat_transform, check_keymat_inverse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-F KeyMat unit checks.")
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_f/keymat_unit.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dims", default="64,128,896")
    parser.add_argument("--expansions", default="32,64,128")
    parser.add_argument("--lambdas", default="0.1,0.3,1.0")
    return parser.parse_args()


def parse_csv_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    dims = parse_csv_ints(args.dims)
    expansions = parse_csv_ints(args.expansions)
    lambdas = parse_csv_floats(args.lambdas)
    results = []
    for dim in dims:
        for expansion in expansions:
            if expansion * 2 >= dim and dim != 896:
                continue
            for lam in lambdas:
                transform = build_keymat_transform(
                    d=dim,
                    h=expansion,
                    lam=lam,
                    init_seed=args.seed + dim + expansion,
                    key_seed=args.seed + dim + expansion + 1,
                    inv_seed=args.seed + dim + expansion + 2,
                )
                metrics = check_keymat_inverse(transform.key, transform.inverse)
                results.append(
                    {
                        "hidden_size": dim,
                        "expansion_size": expansion,
                        "lambda": lam,
                        "expanded_size": transform.expanded_size,
                        **metrics,
                    }
                )
    write_json(
        args.output_path,
        {
            "mode": "stage_f_keymat_unit",
            "seed": args.seed,
            "results": results,
        },
    )
    print(f"Saved stage-F KeyMat unit report to {args.output_path}")


if __name__ == "__main__":
    main()
