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
from src.stage_i_vllm import load_stage_i_hf_bundle
from src.transforms import map_input_ids, restore_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-I HF regression on the exported checkpoint.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--server-dir", default="artifacts/stage_i_vllm/server")
    parser.add_argument("--client-secret", default="artifacts/stage_i_vllm/client/client_secret.pt")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_i/hf_regression.json")
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
def greedy_generate_stage_a(model, input_ids: torch.Tensor, perm_vocab: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    current_plain_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        mapped_ids = map_input_ids(current_plain_ids, perm_vocab)
        attention_mask = torch.ones_like(mapped_ids)
        logits_perm = model(input_ids=mapped_ids, attention_mask=attention_mask).logits.detach().to(torch.float32)
        restored_logits = restore_logits(logits_perm[:, -1, :], perm_vocab)
        next_token = torch.argmax(restored_logits, dim=-1, keepdim=True)
        current_plain_ids = torch.cat([current_plain_ids, next_token.to(current_plain_ids.device)], dim=1)
    return current_plain_ids[:, input_ids.shape[1] :]


def summarize_items(items: list[dict]) -> dict[str, float]:
    count = max(len(items), 1)
    return {
        "prompt_count": len(items),
        "avg_full_logits_max_abs_error": sum(float(item["full_logits_max_abs_error"]) for item in items) / count,
        "avg_full_logits_mean_abs_error": sum(float(item["full_logits_mean_abs_error"]) for item in items) / count,
        "avg_last_token_logits_max_abs_error": sum(float(item["last_token_logits_max_abs_error"]) for item in items) / count,
        "avg_last_token_logits_mean_abs_error": sum(float(item["last_token_logits_mean_abs_error"]) for item in items) / count,
        "greedy_first_token_match_rate": sum(1.0 for item in items if item["greedy_first_token_match"]) / count,
        "generated_ids_exact_match_rate": sum(1.0 for item in items if item["generated_ids_exact_match"]) / count,
        "generated_text_exact_match_rate": sum(1.0 for item in items if item["generated_text_exact_match"]) / count,
        "baseline_has_nan_or_inf": any(item["baseline_has_nan_or_inf"] for item in items),
        "stage_a_has_nan_or_inf": any(item["stage_a_has_nan_or_inf"] for item in items),
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    bundle = load_stage_i_hf_bundle(
        args.server_dir,
        client_secret_path=args.client_secret,
        device="cpu",
        dtype=args.dtype,
    )
    exported_tokenizer = bundle["tokenizer"]
    stage_model = bundle["model"]
    perm_vocab = bundle["perm_vocab"]

    if perm_vocab is None:
        raise ValueError("client secret is required for Stage-I HF regression")

    prompt_results: list[dict] = []
    for index, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        encoded = tokenize_prompt(tokenizer, prompt, device="cpu")
        baseline_logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)

        mapped_ids = map_input_ids(encoded["input_ids"], perm_vocab)
        stage_logits_perm = stage_model(
            input_ids=mapped_ids,
            attention_mask=encoded["attention_mask"],
        ).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(stage_logits_perm, perm_vocab)

        baseline_generated_ids = greedy_generate_plain(
            baseline_model,
            encoded["input_ids"],
            max_new_tokens=args.max_new_tokens,
        )[0].cpu()
        stage_generated_ids = greedy_generate_stage_a(
            stage_model,
            encoded["input_ids"],
            perm_vocab=perm_vocab,
            max_new_tokens=args.max_new_tokens,
        )[0].cpu()

        prompt_results.append(
            {
                "prompt_id": index,
                "prompt": prompt,
                "mapped_input_ids": mapped_ids[0].tolist(),
                "full_logits_max_abs_error": max_abs_error(baseline_logits, restored_logits),
                "full_logits_mean_abs_error": mean_abs_error(baseline_logits, restored_logits),
                "last_token_logits_max_abs_error": max_abs_error(baseline_logits[0, -1], restored_logits[0, -1]),
                "last_token_logits_mean_abs_error": mean_abs_error(baseline_logits[0, -1], restored_logits[0, -1]),
                "greedy_first_token_match": int(torch.argmax(baseline_logits[0, -1]).item()) == int(torch.argmax(restored_logits[0, -1]).item()),
                "generated_ids_exact_match": baseline_generated_ids.tolist() == stage_generated_ids.tolist(),
                "generated_text_exact_match": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True)
                == exported_tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "baseline_generated_ids": baseline_generated_ids.tolist(),
                "stage_a_generated_ids": stage_generated_ids.tolist(),
                "baseline_generated_text": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True),
                "stage_a_generated_text": exported_tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "baseline_has_nan_or_inf": not bool(torch.isfinite(baseline_logits).all().item()),
                "stage_a_has_nan_or_inf": not bool(torch.isfinite(stage_logits_perm).all().item()),
            }
        )

    payload = {
        "stage": "I",
        "phase": "I-A",
        "backend": "transformers",
        "model_dir": args.model_dir,
        "server_dir": args.server_dir,
        "client_secret": args.client_secret,
        "dtype": args.dtype,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "summary": summarize_items(prompt_results),
        "prompts": prompt_results,
    }
    write_json(args.output_path, payload)
    print(f"Saved Stage-I HF regression report to {args.output_path}")


if __name__ == "__main__":
    main()
