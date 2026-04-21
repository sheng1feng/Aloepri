from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import write_json
from src.stage_j_bridge_regression import run_stage_j_bridge_regression
from src.stage_j_norm_gap import build_stage_j_norm_gap_report
from src.stage_j_materialize import export_stage_j_redesign_checkpoint
from src.stage_j_standard_bridge import export_stage_j_redesign_standard_bridge
from src.stage_h_pretrained import export_stage_h_pretrained
from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import TraceRecorder
from src.stage_f import build_default_stage_f_keymat, build_layer_stage_f_configs, calibrate_keymat_kappas
from src.stage_h import build_layer_stage_h_configs, build_stage_h_model


def main() -> None:
    root = Path("/home/shengfeng/Privacy-inference")
    stage_h_dir = root / "artifacts/stage_h_pretrained_diagfriendly"
    stage_j_dir = root / "artifacts/stage_j_qwen_redesign_diagfriendly"
    stage_j_standard_dir = root / "artifacts/stage_j_qwen_redesign_standard_diagfriendly"

    set_global_seed(DEFAULT_SEED)
    tokenizer, baseline_model = load_model_and_tokenizer(DEFAULT_MODEL_DIR, device="cpu", dtype="float32")
    adapted_layers = list(range(baseline_model.config.num_hidden_layers))
    keymat_transform = build_default_stage_f_keymat(
        baseline_model,
        lam=0.3,
        h=128,
        seed=DEFAULT_SEED,
        family="diag_friendly",
    )
    kappas = calibrate_keymat_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS[:2],
        keymat_transform=keymat_transform,
        trace_layers=adapted_layers,
    )
    layer_configs_f = build_layer_stage_f_configs(
        baseline_model=baseline_model,
        keymat_transform=keymat_transform,
        trace_layers=adapted_layers,
        kappa_by_layer=kappas,
        attention_profile="rqk_hqk_block_taukv_taugroup",
        seed=DEFAULT_SEED,
        alpha_e=0.1,
        alpha_h=0.05,
    )
    layer_configs_h = build_layer_stage_h_configs(layer_configs_f)
    stage_model, perm_vocab, inv_perm_vocab = build_stage_h_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        keymat_transform=keymat_transform,
        seed=DEFAULT_SEED,
        recorder=TraceRecorder(),
        layer_configs=layer_configs_h,
        adapted_layers=adapted_layers,
        alpha_e=0.1,
        alpha_h=0.05,
        use_keymat_head=True,
        beta=8,
        gamma=1e3,
    )
    metadata = {
        "model_dir": DEFAULT_MODEL_DIR,
        "dtype": "float32",
        "seed": DEFAULT_SEED,
        "lambda": 0.3,
        "h": 128,
        "keymat_family": "diag_friendly",
        "attention_profile": "rqk_hqk_block_taukv_taugroup",
        "alpha_e": 0.1,
        "alpha_h": 0.05,
        "beta": 8,
        "gamma": 1e3,
        "adapted_layers": adapted_layers,
        "prompts_for_kappa": DEFAULT_PROMPTS[:2],
    }
    export_stage_h_pretrained(
        stage_h_dir,
        tokenizer=tokenizer,
        baseline_model=baseline_model,
        stage_model=stage_model,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        metadata=metadata,
    )
    export_stage_j_redesign_checkpoint(stage_j_dir, source_dir=stage_h_dir)
    export_stage_j_redesign_standard_bridge(stage_j_standard_dir, source_dir=stage_j_dir)

    norm_gap = build_stage_j_norm_gap_report(stage_j_dir / "server")
    bridge = run_stage_j_bridge_regression(
        buffered_server_dir=stage_j_dir / "server",
        bridge_server_dir=stage_j_standard_dir / "server",
    )
    summary = {
        "stage_h_pretrained_dir": str(stage_h_dir),
        "stage_j_redesign_dir": str(stage_j_dir),
        "stage_j_standard_dir": str(stage_j_standard_dir),
        "norm_gap_summary": norm_gap["summary"],
        "bridge_summary": bridge["summary"],
    }
    write_json(root / "outputs/stage_j/diagfriendly_candidate_summary.json", summary)
    print("Saved diagfriendly candidate summary to outputs/stage_j/diagfriendly_candidate_summary.json")


if __name__ == "__main__":
    main()
