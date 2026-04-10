from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_SEED
from src.key_manager import ordinary_token_ids
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.security_qwen import run_ima_baseline, run_isa_baseline, run_sda_baseline, run_tfma_baseline, run_vma_baseline
from src.stage_i_vllm import export_stage_i_vllm_checkpoint, summarize_token_partitions
from src.stage_j_block0 import build_stage_j_square_model


@dataclass(frozen=True)
class Gate5Case:
    name: str
    target_name: str
    alpha_e: float
    alpha_h: float
    export_dir: str
    existing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_gate5_cases() -> list[Gate5Case]:
    return [
        Gate5Case("stable_reference", "stage_j_stable_reference", 0.0, 0.0, "artifacts/stage_j_full_square", existing=True),
        Gate5Case("tiny_a", "stage_j_tiny_a", 0.02, 0.01, "artifacts/stage_j_full_square_tiny_a", existing=True),
        Gate5Case("tiny_b", "stage_j_tiny_b_scan", 0.05, 0.02, "artifacts/stage_j_gate5_tiny_b"),
        Gate5Case("small_a", "stage_j_small_a_scan", 0.1, 0.05, "artifacts/stage_j_gate5_small_a"),
        Gate5Case("paper_like", "stage_j_paper_like_scan", 1.0, 0.2, "artifacts/stage_j_gate5_paper_like"),
    ]


def load_stage_j_noise_cases() -> dict[str, dict[str, Any]]:
    payload = json.loads(Path("outputs/stage_j/noise_calibration.json").read_text(encoding="utf-8"))
    return {item["name"]: item for item in payload["cases"]}


def ensure_gate5_artifact(case: Gate5Case) -> Path:
    export_dir = Path(case.export_dir)
    if case.existing and (export_dir / "server" / "model.safetensors").exists():
        return export_dir
    if (export_dir / "server" / "model.safetensors").exists() and (export_dir / "client" / "client_secret.pt").exists():
        return export_dir

    set_global_seed(DEFAULT_SEED)
    tokenizer, baseline_model = load_model_and_tokenizer(DEFAULT_MODEL_DIR, device="cpu", dtype="float32")
    adapted_layers = list(range(baseline_model.config.num_hidden_layers))
    stage_model, perm_vocab, inv_perm_vocab, transform = build_stage_j_square_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        adapted_layers=adapted_layers,
        seed=DEFAULT_SEED,
        alpha_e=case.alpha_e,
        alpha_h=case.alpha_h,
        global_scale=1.0,
        recorder=None,
    )
    metadata = {
        "stage": "J",
        "variant": "full_layer_square_transform",
        "model_dir": DEFAULT_MODEL_DIR,
        "seed": DEFAULT_SEED,
        "dtype": "float32",
        "alpha_e": case.alpha_e,
        "alpha_h": case.alpha_h,
        "global_scale": 1.0,
        "adapted_layers": adapted_layers,
        "hidden_size": int(baseline_model.config.hidden_size),
        "square_transform": {
            "perm": transform.perm.tolist(),
            "inv_perm": transform.inv_perm.tolist(),
            "signs": transform.signs.tolist(),
            "global_scale": transform.global_scale,
        },
        "movable_token_count": int(ordinary_token_ids(tokenizer).numel()),
        **summarize_token_partitions(
            tokenizer=tokenizer,
            model_vocab_size=stage_model.get_input_embeddings().weight.shape[0],
            perm_vocab=perm_vocab,
        ),
    }
    export_stage_i_vllm_checkpoint(
        export_dir,
        tokenizer=tokenizer,
        stage_a_model=stage_model,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        metadata=metadata,
    )
    return export_dir


def build_gate5_scan_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete_rows = [row for row in rows if row.get("generated_ids_exact_match_rate") == 1.0]
    recommended_nonzero = None
    if complete_rows:
        nonzero = [row for row in complete_rows if row["alpha_e"] > 0 or row["alpha_h"] > 0]
        ranked = sorted(nonzero or complete_rows, key=lambda item: (item["vma_projection_top1"], item["ima_top1"], item["isa_hidden_top1"], item["avg_final_logits_restored_max_abs_error"]))
        recommended_nonzero = ranked[0]
    return {
        "format": "qwen_security_gate5_scan_v1",
        "row_count": len(rows),
        "rows": rows,
        "recommended_nonzero_case": recommended_nonzero,
    }


def run_gate5_scan(cases: list[Gate5Case] | None = None) -> dict[str, Any]:
    case_list = cases or default_gate5_cases()
    noise_cases = load_stage_j_noise_cases()
    rows: list[dict[str, Any]] = []

    for case in case_list:
        ensure_gate5_artifact(case)
        accuracy = noise_cases[case.name]["summary"]
        vma = run_vma_baseline(target_name=case.target_name, eval_size=128, candidate_pool_size=2048, feature_bins=64, topk=10)
        ima = run_ima_baseline(target_name=case.target_name, train_size=1024, val_size=128, test_size=128, candidate_pool_size=2048, topk=10)
        isa = run_isa_baseline(target_name=case.target_name, observable_type="hidden_state", observable_layer=23, train_sequences=64, val_sequences=16, test_sequences=16, sequence_length=8, candidate_pool_size=2048, topk=10)
        tfma = run_tfma_baseline(target_name=case.target_name, knowledge_setting="domain_aware", candidate_pool_size=512, topk=100)
        sda = run_sda_baseline(target_name=case.target_name, knowledge_setting="distribution_aware", candidate_pool_size=256, topk=100)

        rows.append(
            {
                "name": case.name,
                "target_name": case.target_name,
                "artifact_dir": case.export_dir,
                "alpha_e": case.alpha_e,
                "alpha_h": case.alpha_h,
                "avg_final_logits_restored_max_abs_error": accuracy["avg_final_logits_restored_max_abs_error"],
                "generated_ids_exact_match_rate": accuracy["generated_ids_exact_match_rate"],
                "generated_text_exact_match_rate": accuracy["generated_text_exact_match_rate"],
                "vma_projection_top1": vma["metrics"]["token_top1_recovery_rate"],
                "ima_top1": ima["metrics"]["token_top1_recovery_rate"],
                "isa_hidden_top1": isa["metrics"]["intermediate_top1_recovery_rate"],
                "tfma_domain_top10": tfma["metrics"]["token_top10_recovery_rate"],
                "sda_distribution_bleu4": sda["metrics"]["bleu4"],
            }
        )
    rows.sort(key=lambda item: (item["alpha_e"], item["alpha_h"]))
    return build_gate5_scan_payload(rows)
