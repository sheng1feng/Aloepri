import argparse
import gc
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.evaluator import max_abs_error, mean_abs_error, write_json
from src.llama_local_dev import tokenize_llama_prompt
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_j_block0 import build_stage_j_square_model
from src.stage_j_noise import default_stage_j_noise_cases, rank_stage_j_noise_cases
from src.transforms import map_input_ids, restore_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-J full-layer noise calibration on real Llama-3.2-3B.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_j_llama/real_noise_calibration.json")
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
        mapped_ids = map_input_ids(current_plain_ids, perm_vocab.to(current_plain_ids.device))
        attention_mask = torch.ones_like(mapped_ids)
        logits_perm = model(input_ids=mapped_ids, attention_mask=attention_mask).logits.detach().to(torch.float32)
        restored_logits = restore_logits(logits_perm[:, -1, :].cpu(), perm_vocab.cpu())
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
    stage_model, perm_vocab, _, transform = build_stage_j_square_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        adapted_layers=adapted_layers,
        seed=seed,
        alpha_e=case.alpha_e,
        alpha_h=case.alpha_h,
        global_scale=case.global_scale,
        recorder=None,
    )
    inverse_matrix = transform.inverse(dtype=torch.float32).to(baseline_model.device)
    last_layer_idx = baseline_model.config.num_hidden_layers - 1
    prompt_results: list[dict] = []

    for prompt in DEFAULT_PROMPTS:
        encoded = tokenize_llama_prompt(tokenizer, prompt, device=baseline_model.device)
        baseline_model_out = baseline_model.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
        mapped_ids = map_input_ids(encoded["input_ids"].cpu(), perm_vocab.cpu()).to(encoded["input_ids"].device)
        stage_model_out = stage_model.model(
            input_ids=mapped_ids,
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
        baseline_logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)
        observed_logits_perm = stage_model(input_ids=mapped_ids, attention_mask=encoded["attention_mask"]).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(observed_logits_perm, perm_vocab.cpu())

        baseline_hidden = baseline_model_out.hidden_states[last_layer_idx + 1].detach().to(torch.float32)
        observed_hidden = stage_model_out.hidden_states[last_layer_idx + 1].detach().to(torch.float32)
        restored_hidden = torch.matmul(observed_hidden, inverse_matrix.to(observed_hidden.device))

        baseline_generated_ids = greedy_generate_plain(baseline_model, encoded["input_ids"], max_new_tokens)[0].cpu()
        stage_generated_ids = greedy_generate_stage(stage_model, encoded["input_ids"], perm_vocab, max_new_tokens)[0].cpu()

        prompt_results.append(
            {
                "prompt": prompt,
                "final_logits_restored_max_abs_error": max_abs_error(baseline_logits, restored_logits),
                "final_logits_restored_mean_abs_error": mean_abs_error(baseline_logits, restored_logits),
                f"layer_{last_layer_idx}_block_out_max_abs_error": max_abs_error(baseline_hidden.cpu(), restored_hidden.cpu()),
                f"layer_{last_layer_idx}_block_out_mean_abs_error": mean_abs_error(baseline_hidden.cpu(), restored_hidden.cpu()),
                "greedy_first_token_match": bool(baseline_generated_ids[0].item() == stage_generated_ids[0].item()),
                "generated_ids_exact_match": baseline_generated_ids.tolist() == stage_generated_ids.tolist(),
                "generated_text_exact_match": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True)
                == tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "baseline_generated_text": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True),
                "restored_generated_text": tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
            }
        )

    summary = aggregate(prompt_results)
    result = {
        "case": case.name,
        "alpha_e": case.alpha_e,
        "alpha_h": case.alpha_h,
        "global_scale": case.global_scale,
        "summary": summary,
        "prompts": prompt_results,
    }

    del stage_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device=args.device, dtype=args.dtype)
    cases = default_stage_j_noise_cases()
    results = [
        run_case(
            baseline_model=baseline_model,
            tokenizer=tokenizer,
            case=case,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
        for case in cases
    ]
    ranked = rank_stage_j_noise_cases(results)
    payload = {
        "stage": "J",
        "variant": "llama_real_noise_calibration",
        "model_dir": args.model_dir,
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "ranked_cases": [item["case"] for item in ranked],
        "results": results,
    }
    write_json(args.output_path, payload)
    print(f"Saved Llama Stage-J real noise calibration report to {args.output_path}")


if __name__ == "__main__":
    main()
