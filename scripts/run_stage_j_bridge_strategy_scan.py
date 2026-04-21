from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import write_json
from src.stage_j_bridge_regression import run_stage_j_bridge_regression
from src.stage_j_bridge_scan import rank_bridge_strategies
from src.stage_j_standard_bridge import export_stage_j_redesign_standard_bridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Stage-J bridge norm strategies.")
    parser.add_argument("--buffered-server-dir", default="artifacts/stage_j_qwen_redesign/server")
    parser.add_argument("--bridge-export-dir", default="artifacts/stage_j_qwen_redesign_standard")
    parser.add_argument("--bridge-source-dir", default="artifacts/stage_j_qwen_redesign")
    parser.add_argument("--output-path", default="outputs/stage_j/bridge_strategy_scan.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategies = ["ones", "metric_diag_sqrt", "kappa_fused"]
    rows = []
    for strategy in strategies:
        export_stage_j_redesign_standard_bridge(
            args.bridge_export_dir,
            source_dir=args.bridge_source_dir,
            norm_strategy=strategy,
        )
        payload = run_stage_j_bridge_regression(
            buffered_server_dir=args.buffered_server_dir,
            bridge_server_dir=str(Path(args.bridge_export_dir) / "server"),
        )
        rows.append(
            {
                "norm_strategy": strategy,
                "summary": payload["summary"],
            }
        )
    ranked = rank_bridge_strategies(rows)
    output = {
        "strategies": rows,
        "ranked": ranked,
        "recommended_strategy": ranked[0]["norm_strategy"] if ranked else None,
    }
    write_json(args.output_path, output)
    print(f"Saved Stage-J bridge strategy scan to {args.output_path}")


if __name__ == "__main__":
    main()
