import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.evaluator import write_json
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import TraceRecorder
from src.stage_d import attach_stage_d_hooks
from src.stage_f import (
    aggregate_stage_f_results,
    build_default_stage_f_keymat,
    build_layer_stage_f_configs,
    calibrate_keymat_kappas,
    run_stage_f_single_prompt,
)
from src.stage_g import build_layer_stage_g_configs, build_stage_g_model
from src.stage_h_noise import default_noise_cases, rank_noise_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-H noise calibration.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_h/noise_calibration.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--attention-profile", default="rqk_hqk_block_taukv_taugroup")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


def run_case(
    *,
    baseline_model,
    tokenizer,
    case,
    attention_profile: str,
    seed: int,
    max_new_tokens: int,
):
    trace_layers = list(range(baseline_model.config.num_hidden_layers))
    keymat_transform = build_default_stage_f_keymat(baseline_model, lam=case.lam, h=case.h, seed=seed)
    kappas = calibrate_keymat_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS[:2],
        keymat_transform=keymat_transform,
        trace_layers=trace_layers,
    )
    layer_configs_f = build_layer_stage_f_configs(
        baseline_model=baseline_model,
        keymat_transform=keymat_transform,
        trace_layers=trace_layers,
        kappa_by_layer=kappas,
        attention_profile=attention_profile,
        seed=seed,
        alpha_e=case.alpha_e,
        alpha_h=case.alpha_h,
    )
    layer_configs_g = build_layer_stage_g_configs(layer_configs_f)
    baseline_recorder = TraceRecorder()
    observed_recorder = TraceRecorder()
    cleanup = attach_stage_d_hooks(
        baseline_model,
        baseline_recorder,
        trace_layers=trace_layers,
        attention_mode="plain",
        capture_embed_output=True,
    )
    try:
        stage_model, perm_vocab, inv_perm_vocab = build_stage_g_model(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            keymat_transform=keymat_transform,
            seed=seed,
            recorder=observed_recorder,
            layer_configs=layer_configs_g,
            adapted_layers=trace_layers,
            mode="attention_fused",
            alpha_e=case.alpha_e,
            alpha_h=case.alpha_h,
            use_keymat_head=True,
            beta=case.beta,
            gamma=case.gamma,
        )
        results = [
            run_stage_f_single_prompt(
                baseline_model=baseline_model,
                tokenizer=tokenizer,
                prompt=prompt,
                stage_model=stage_model,
                perm_vocab=perm_vocab,
                inv_perm_vocab=inv_perm_vocab,
                baseline_recorder=baseline_recorder,
                observed_recorder=observed_recorder,
                trace_layers=trace_layers,
                max_new_tokens=max_new_tokens,
            )
            for prompt in DEFAULT_PROMPTS
        ]
    finally:
        cleanup()
    summary = aggregate_stage_f_results(results)
    return {
        "name": case.name,
        "alpha_e": case.alpha_e,
        "alpha_h": case.alpha_h,
        "lambda": case.lam,
        "h": case.h,
        "beta": case.beta,
        "gamma": case.gamma,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    cases = default_noise_cases()
    results = [
        run_case(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            case=case,
            attention_profile=args.attention_profile,
            seed=args.seed + idx * 100,
            max_new_tokens=args.max_new_tokens,
        )
        for idx, case in enumerate(cases)
    ]
    ranked = rank_noise_cases(results)
    nonzero_ranked = [item for item in ranked if item["alpha_e"] > 0 or item["alpha_h"] > 0]
    payload = {
        "mode": "stage_h_noise_calibration",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "dtype": args.dtype,
        "attention_profile": args.attention_profile,
        "cases": results,
        "paper_default_case": next(item for item in results if item["name"] == "paper_default"),
        "recommended_working_point": nonzero_ranked[0] if nonzero_ranked else ranked[0],
        "reference_point": next(item for item in results if item["name"] == "stable_reference"),
        "ranking": [item["name"] for item in ranked],
    }
    write_json(args.output_path, payload)
    print(f"Saved stage-H noise calibration report to {args.output_path}")


if __name__ == "__main__":
    main()
