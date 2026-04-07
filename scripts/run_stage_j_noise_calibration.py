import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.evaluator import max_abs_error, mean_abs_error, write_json
from src.model_loader import load_model_and_tokenizer, set_global_seed, tokenize_prompt
from src.stage_j_block0 import build_stage_j_square_model
from src.stage_j_noise import default_stage_j_noise_cases, rank_stage_j_noise_cases
from src.transforms import map_input_ids, restore_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-J full-layer noise calibration.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_j/noise_calibration.json")
    return parser.parse_args()


@torch.inference_mode()
def greedy_generate_plain(model, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    current_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(current_ids)
        logits = model(input_ids=current_ids, attention_mask=attention_mask).logits.detach()
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        current_ids = torch.cat([current_ids, next_token.to(current_ids.device)], dim=1)
    return current_ids[:, input_ids.shape[1] :]


@torch.inference_mode()
def greedy_generate_stage(model, input_ids: torch.Tensor, perm_vocab: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    current_plain_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        mapped_ids = map_input_ids(current_plain_ids, perm_vocab)
        attention_mask = torch.ones_like(mapped_ids)
        logits_perm = model(input_ids=mapped_ids, attention_mask=attention_mask).logits.detach().to(torch.float32)
        restored_logits = restore_logits(logits_perm[:, -1, :], perm_vocab)
        next_token = torch.argmax(restored_logits, dim=-1, keepdim=True)
        current_plain_ids = torch.cat([current_plain_ids, next_token.to(current_plain_ids.device)], dim=1)
    return current_plain_ids[:, input_ids.shape[1] :]


def aggregate(results: list[dict]) -> dict[str, float]:
    count = max(len(results), 1)
    numeric_keys = [key for key, value in results[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    summary = {f"avg_{key}": sum(float(result[key]) for result in results) / count for key in numeric_keys}
    for bool_key in ["greedy_first_token_match", "generated_ids_exact_match", "generated_text_exact_match"]:
        summary[f"{bool_key}_rate"] = sum(1.0 for result in results if result[bool_key]) / count
    return summary


def run_case(*, baseline_model, tokenizer, case, seed: int, max_new_tokens: int) -> dict:
    adapted_layers = list(range(baseline_model.config.num_hidden_layers))
    stage_model, perm_vocab, inv_perm_vocab, transform = build_stage_j_square_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        adapted_layers=adapted_layers,
        seed=seed,
        alpha_e=case.alpha_e,
        alpha_h=case.alpha_h,
        global_scale=case.global_scale,
        recorder=None,
    )
    inverse_matrix = transform.inverse(dtype=torch.float32)

    prompt_results: list[dict] = []
    for prompt in DEFAULT_PROMPTS:
        encoded = tokenize_prompt(tokenizer, prompt, device="cpu")
        baseline_out = baseline_model.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
        mapped_ids = map_input_ids(encoded["input_ids"], perm_vocab)
        stage_out = stage_model.model(
            input_ids=mapped_ids,
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
        baseline_logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)
        observed_logits_perm = stage_model(input_ids=mapped_ids, attention_mask=encoded["attention_mask"]).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(observed_logits_perm, perm_vocab)
        baseline_generated_ids = greedy_generate_plain(baseline_model, encoded["input_ids"], max_new_tokens)[0].cpu()
        stage_generated_ids = greedy_generate_stage(stage_model, encoded["input_ids"], perm_vocab, max_new_tokens)[0].cpu()

        final_hidden_base = baseline_out.hidden_states[-1].detach().cpu().to(torch.float32)
        final_hidden_obs = torch.matmul(stage_out.hidden_states[-1].detach().cpu().to(torch.float32), inverse_matrix)
        layer23_base = baseline_out.hidden_states[-2].detach().cpu().to(torch.float32)
        layer23_obs = torch.matmul(stage_out.hidden_states[-2].detach().cpu().to(torch.float32), inverse_matrix)

        prompt_results.append(
            {
                "prompt": prompt,
                "final_logits_restored_max_abs_error": max_abs_error(baseline_logits, restored_logits),
                "final_logits_restored_mean_abs_error": mean_abs_error(baseline_logits, restored_logits),
                "layer_23_block_out_max_abs_error": max_abs_error(layer23_base, layer23_obs),
                "layer_23_block_out_mean_abs_error": mean_abs_error(layer23_base, layer23_obs),
                "final_hidden_max_abs_error": max_abs_error(final_hidden_base, final_hidden_obs),
                "final_hidden_mean_abs_error": mean_abs_error(final_hidden_base, final_hidden_obs),
                "greedy_first_token_match": bool(baseline_generated_ids[0].item() == stage_generated_ids[0].item()),
                "generated_ids_exact_match": baseline_generated_ids.tolist() == stage_generated_ids.tolist(),
                "generated_text_exact_match": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True) == tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "baseline_generated_text": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True),
                "restored_generated_text": tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "has_nan_or_inf": not bool(torch.isfinite(observed_logits_perm).all().item()),
            }
        )

    return {
        "name": case.name,
        "alpha_e": case.alpha_e,
        "alpha_h": case.alpha_h,
        "global_scale": case.global_scale,
        "summary": aggregate(prompt_results),
        "prompts": prompt_results,
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    results = [
        run_case(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            case=case,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
        for case in default_stage_j_noise_cases()
    ]
    ranked = rank_stage_j_noise_cases(results)
    payload = {
        "stage": "J",
        "variant": "full_layer_square_noise_calibration",
        "model_dir": args.model_dir,
        "dtype": args.dtype,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "ranked_case_names": [item["name"] for item in ranked],
        "cases": results,
        "recommended_case": ranked[0],
    }
    write_json(args.output_path, payload)
    print(f"Saved Stage-J noise calibration report to {args.output_path}")


if __name__ == "__main__":
    main()
