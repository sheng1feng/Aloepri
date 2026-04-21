from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import write_json
from src.stage_j_bridge_regression import run_stage_j_bridge_regression
from src.stage_j_bridge_scan import rank_bridge_strategies
from src.stage_j_standard_bridge import export_stage_j_redesign_standard_bridge


def main() -> None:
    strategies = ["ones", "metric_diag_sqrt", "kappa_fused"]
    rows = []
    for strategy in strategies:
        export_stage_j_redesign_standard_bridge(
            "artifacts/stage_j_qwen_redesign_standard",
            source_dir="artifacts/stage_j_qwen_redesign",
            norm_strategy=strategy,
        )
        payload = run_stage_j_bridge_regression(
            buffered_server_dir="artifacts/stage_j_qwen_redesign/server",
            bridge_server_dir="artifacts/stage_j_qwen_redesign_standard/server",
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
    write_json("outputs/stage_j/bridge_strategy_scan.json", output)
    print("Saved Stage-J bridge strategy scan to outputs/stage_j/bridge_strategy_scan.json")


if __name__ == "__main__":
    main()
