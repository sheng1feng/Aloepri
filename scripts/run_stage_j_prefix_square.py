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
from src.transforms import map_input_ids, restore_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-J square-transform prefix regression using hidden_states.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--alpha-e", type=float, default=0.0)
    parser.add_argument("--alpha-h", type=float, default=0.0)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-path", default=None)
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


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    total_layers = baseline_model.config.num_hidden_layers
    if args.layer_count < 1 or args.layer_count > total_layers:
        raise ValueError(f"layer-count must be in [1, {total_layers}]")
    if args.output_path is None:
        if args.layer_count == total_layers:
            args.output_path = f"{DEFAULT_OUTPUT_DIR}/stage_j/full_layers_square.json"
        else:
            args.output_path = f"{DEFAULT_OUTPUT_DIR}/stage_j/prefix_{args.layer_count}_square.json"

    adapted_layers = list(range(args.layer_count))
    stage_model, perm_vocab, inv_perm_vocab, transform = build_stage_j_square_model(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        adapted_layers=adapted_layers,
        seed=args.seed,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
        global_scale=args.global_scale,
        recorder=None,
    )
    inverse_matrix = transform.inverse(dtype=torch.float32)

    prompt_results: list[dict] = []
    for prompt in DEFAULT_PROMPTS:
        encoded = tokenize_prompt(tokenizer, prompt, device="cpu")
        baseline_model_out = baseline_model.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
        mapped_ids = map_input_ids(encoded["input_ids"], perm_vocab)
        stage_model_out = stage_model.model(
            input_ids=mapped_ids,
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
        baseline_logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)
        observed_logits_perm = stage_model(input_ids=mapped_ids, attention_mask=encoded["attention_mask"]).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(observed_logits_perm, perm_vocab)

        item = {
            "prompt": prompt,
            "final_logits_restored_max_abs_error": max_abs_error(baseline_logits, restored_logits),
            "final_logits_restored_mean_abs_error": mean_abs_error(baseline_logits, restored_logits),
        }
        for layer_idx in adapted_layers:
            baseline_hidden = baseline_model_out.hidden_states[layer_idx + 1].detach().cpu().to(torch.float32)
            observed_hidden = stage_model_out.hidden_states[layer_idx + 1].detach().cpu().to(torch.float32)
            restored_hidden = torch.matmul(observed_hidden, inverse_matrix)
            item[f"layer_{layer_idx}_block_out_max_abs_error"] = max_abs_error(baseline_hidden, restored_hidden)
            item[f"layer_{layer_idx}_block_out_mean_abs_error"] = mean_abs_error(baseline_hidden, restored_hidden)

        baseline_generated_ids = greedy_generate_plain(baseline_model, encoded["input_ids"], args.max_new_tokens)[0].cpu()
        stage_generated_ids = greedy_generate_stage(stage_model, encoded["input_ids"], perm_vocab, args.max_new_tokens)[0].cpu()
        item["greedy_first_token_match"] = bool(baseline_generated_ids[0].item() == stage_generated_ids[0].item())
        item["generated_ids_exact_match"] = baseline_generated_ids.tolist() == stage_generated_ids.tolist()
        item["generated_text_exact_match"] = tokenizer.decode(baseline_generated_ids, skip_special_tokens=True) == tokenizer.decode(stage_generated_ids, skip_special_tokens=True)
        item["baseline_generated_text"] = tokenizer.decode(baseline_generated_ids, skip_special_tokens=True)
        item["restored_generated_text"] = tokenizer.decode(stage_generated_ids, skip_special_tokens=True)
        prompt_results.append(item)

    payload = {
        "stage": "J",
        "variant": "square_full_hidden_states_regression" if args.layer_count == total_layers else "square_prefix_hidden_states_regression",
        "layer_count": args.layer_count,
        "adapted_layers": adapted_layers,
        "dtype": args.dtype,
        "seed": args.seed,
        "alpha_e": args.alpha_e,
        "alpha_h": args.alpha_h,
        "global_scale": args.global_scale,
        "summary": aggregate(prompt_results),
        "prompts": prompt_results,
    }
    write_json(args.output_path, payload)
    print(f"Saved Stage-J prefix regression report to {args.output_path}")


if __name__ == "__main__":
    main()
