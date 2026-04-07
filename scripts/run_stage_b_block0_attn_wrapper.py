import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPTS,
    DEFAULT_SEED,
    DEFAULT_STAGE_B_DTYPE,
    DEFAULT_STAGE_B_SCALE_RANGE,
)
from src.evaluator import write_json
from src.hidden_keys import (
    build_hidden_transform,
    generate_hidden_permutation,
    generate_hidden_scaling,
    invert_hidden_transform,
    validate_hidden_transform,
)
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import (
    StageBHiddenPermutationModel,
    TraceRecorder,
    aggregate_stage_b_results,
    attach_stage_b_hooks,
    prepare_stage_a_model,
    run_stage_b_single_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-B block0 attention wrapper experiment.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_b/block0_attn_wrapper.json")
    parser.add_argument("--dtype", default=DEFAULT_STAGE_B_DTYPE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scale-low", type=float, default=DEFAULT_STAGE_B_SCALE_RANGE[0])
    parser.add_argument("--scale-high", type=float, default=DEFAULT_STAGE_B_SCALE_RANGE[1])
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(
        model_name=args.model_dir,
        device="cpu",
        dtype=args.dtype,
    )
    stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, args.seed)

    hidden_perm = generate_hidden_permutation(baseline_model.config.hidden_size, seed=args.seed + 101)
    hidden_scale = generate_hidden_scaling(
        baseline_model.config.hidden_size,
        scale_range=(args.scale_low, args.scale_high),
        seed=args.seed + 202,
    )
    hidden_transform = build_hidden_transform(hidden_perm, hidden_scale)
    inv_hidden_transform = invert_hidden_transform(hidden_transform)

    baseline_recorder = TraceRecorder()
    observed_recorder = TraceRecorder()
    stage_b_model = StageBHiddenPermutationModel(
        stage_a_model=stage_a_model,
        hidden_transform=hidden_transform,
        recorder=observed_recorder,
    )

    baseline_cleanup = attach_stage_b_hooks(
        baseline_model,
        baseline_recorder,
        attention_mode="plain",
        capture_embed_output=True,
    )
    observed_cleanup = attach_stage_b_hooks(
        stage_b_model.stage_a_model,
        observed_recorder,
        attention_mode="wrapper",
        hidden_transform=hidden_transform,
        capture_embed_output=False,
    )

    try:
        results = [
            run_stage_b_single_prompt(
                baseline_model=baseline_model,
                tokenizer=tokenizer,
                prompt=prompt,
                stage_b_model=stage_b_model,
                perm_vocab=perm_vocab,
                baseline_recorder=baseline_recorder,
                observed_recorder=observed_recorder,
                hidden_transform=hidden_transform,
            )
            for prompt in DEFAULT_PROMPTS
        ]
    finally:
        baseline_cleanup()
        observed_cleanup()

    payload = {
        "mode": "block0_attn_wrapper",
        "model_dir": args.model_dir,
        "seed": args.seed,
        "dtype": args.dtype,
        "scale_range": [args.scale_low, args.scale_high],
        "hidden_transform": {
            "dim": hidden_transform.dim,
            "invertible": validate_hidden_transform(hidden_transform, inv_hidden_transform),
        },
        "stage_a": {
            "perm_vocab_size": int(perm_vocab.numel()),
            "inv_perm_vocab_size": int(inv_perm_vocab.numel()),
        },
        "summary": aggregate_stage_b_results(results),
        "prompts": [
            {
                "prompt": result.prompt,
                "mapped_input_ids": result.mapped_input_ids,
                **result.metrics,
            }
            for result in results
        ],
    }
    write_json(args.output_path, payload)
    print(f"Saved stage-B block0 wrapper report to {args.output_path}")


if __name__ == "__main__":
    main()

