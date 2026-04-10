from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.security_qwen import (
    default_gate6_cases,
    ensure_gate6_artifact,
    run_ima_baseline,
    run_vma_baseline,
)
from src.security_qwen.gate6_enhancement import security_sensitive_plain_ids
from transformers import AutoTokenizer
from src.defaults import DEFAULT_MODEL_DIR


def _load_stage_j_accuracy() -> dict[str, dict]:
    path = Path("outputs/stage_j/noise_calibration.json")
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: item["summary"] for item in payload["cases"]}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR, trust_remote_code=True)
    sensitive_ids = security_sensitive_plain_ids(tokenizer)
    accuracy_cases = _load_stage_j_accuracy()

    rows: list[dict] = []

    baseline_rows = [
        ("stable_reference", "stage_j_stable_reference", "artifacts/stage_j_full_square"),
        ("tiny_a", "stage_j_tiny_a", "artifacts/stage_j_full_square_tiny_a"),
    ]
    for name, target_name, artifact_dir in baseline_rows:
        vma = run_vma_baseline(target_name=target_name, eval_size=128, candidate_pool_size=2048, feature_bins=64, topk=10, sensitive_plain_ids_override=sensitive_ids)
        ima = run_ima_baseline(target_name=target_name, train_size=1024, val_size=128, test_size=128, candidate_pool_size=2048, topk=10, sensitive_plain_ids_override=sensitive_ids)
        acc = accuracy_cases.get(name, {})
        rows.append(
            {
                "name": name,
                "target_name": target_name,
                "artifact_dir": artifact_dir,
                "alpha_e": vma["config"].get("alpha_e"),
                "alpha_h": vma["config"].get("alpha_h"),
                "accuracy_generated_ids_exact_match_rate": acc.get("generated_ids_exact_match_rate"),
                "accuracy_avg_final_logits_restored_max_abs_error": acc.get("avg_final_logits_restored_max_abs_error"),
                "vma_top1": vma["metrics"]["token_top1_recovery_rate"],
                "vma_sensitive_top1": vma["metrics"]["sensitive_token_recovery_rate"],
                "ima_top1": ima["metrics"]["token_top1_recovery_rate"],
                "ima_sensitive_top1": ima["metrics"]["sensitive_token_recovery_rate"],
            }
        )

    for case in default_gate6_cases():
        artifact = ensure_gate6_artifact(case)
        target_name = f"gate6_{case.name}"
        vma = run_vma_baseline(
            target_name=target_name,
            eval_size=128,
            candidate_pool_size=2048,
            feature_bins=64,
            topk=10,
            sensitive_plain_ids_override=sensitive_ids,
        )
        ima = run_ima_baseline(
            target_name=target_name,
            train_size=1024,
            val_size=128,
            test_size=128,
            candidate_pool_size=2048,
            topk=10,
            sensitive_plain_ids_override=sensitive_ids,
        )
        rows.append(
            {
                "name": case.name,
                "target_name": target_name,
                "artifact_dir": case.export_dir,
                "alpha_e": case.alpha_e,
                "alpha_h": case.alpha_h,
                "accuracy_generated_ids_exact_match_rate": artifact["accuracy"].get("generated_ids_exact_match_rate"),
                "accuracy_avg_final_logits_restored_max_abs_error": artifact["accuracy"].get("avg_final_logits_restored_max_abs_error"),
                "vma_top1": vma["metrics"]["token_top1_recovery_rate"],
                "vma_sensitive_top1": vma["metrics"]["sensitive_token_recovery_rate"],
                "ima_top1": ima["metrics"]["token_top1_recovery_rate"],
                "ima_sensitive_top1": ima["metrics"]["sensitive_token_recovery_rate"],
            }
        )

    output = {
        "format": "qwen_security_gate6_scan_v1",
        "rows": rows,
    }
    out = Path("outputs/security_qwen/summary/gate6_enhancement_scan.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Gate-6 enhancement scan to {out}")


if __name__ == "__main__":
    main()
