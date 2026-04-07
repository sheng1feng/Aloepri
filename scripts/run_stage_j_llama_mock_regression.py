import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.evaluator import max_abs_error, mean_abs_error, write_json
from src.llama_local_dev import build_mock_llama_from_local_metadata, mock_llama_summary, tokenize_llama_prompt
from src.model_loader import set_global_seed
from src.stage_i_vllm import load_stage_i_hf_bundle
from src.transforms import map_input_ids, restore_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-J regression on a mock Llama full-layer standard-shape checkpoint.")
    parser.add_argument("--model-dir", default="model/Llama-3.2-3B")
    parser.add_argument("--server-dir", default="artifacts/stage_j_llama_mock_full_square/server")
    parser.add_argument("--client-secret", default="artifacts/stage_j_llama_mock_full_square/client/client_secret.pt")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_j_llama/mock_full_regression.json")
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
    tokenizer, baseline_model = build_mock_llama_from_local_metadata(args.model_dir, seed=args.seed)
    bundle = load_stage_i_hf_bundle(args.server_dir, client_secret_path=args.client_secret, device="cpu", dtype="float32")
    stage_model = bundle["model"]
    perm_vocab = bundle["perm_vocab"]

    prompt_results: list[dict] = []
    for prompt in DEFAULT_PROMPTS:
        encoded = tokenize_llama_prompt(tokenizer, prompt, device="cpu")
        baseline_logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)
        mapped_ids = map_input_ids(encoded["input_ids"], perm_vocab)
        stage_logits_perm = stage_model(input_ids=mapped_ids, attention_mask=encoded["attention_mask"]).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(stage_logits_perm, perm_vocab)

        baseline_generated_ids = greedy_generate_plain(baseline_model, encoded["input_ids"], args.max_new_tokens)[0].cpu()
        stage_generated_ids = greedy_generate_stage(stage_model, encoded["input_ids"], perm_vocab, args.max_new_tokens)[0].cpu()

        prompt_results.append(
            {
                "prompt": prompt,
                "full_logits_max_abs_error": max_abs_error(baseline_logits, restored_logits),
                "full_logits_mean_abs_error": mean_abs_error(baseline_logits, restored_logits),
                "last_token_logits_max_abs_error": max_abs_error(baseline_logits[:, -1, :], restored_logits[:, -1, :]),
                "last_token_logits_mean_abs_error": mean_abs_error(baseline_logits[:, -1, :], restored_logits[:, -1, :]),
                "greedy_first_token_match": bool(baseline_generated_ids[0].item() == stage_generated_ids[0].item()),
                "generated_ids_exact_match": baseline_generated_ids.tolist() == stage_generated_ids.tolist(),
                "generated_text_exact_match": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True)
                == tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "baseline_generated_text": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True),
                "restored_generated_text": tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
            }
        )

    payload = {
        "stage": "J",
        "variant": "llama_mock_standard_shape_full_regression",
        "seed": args.seed,
        "model_dir": args.model_dir,
        "server_dir": args.server_dir,
        "client_secret": args.client_secret,
        "mock_model": mock_llama_summary(tokenizer, baseline_model),
        "summary": aggregate(prompt_results),
        "prompts": prompt_results,
    }
    write_json(args.output_path, payload)
    print(f"Saved mock Llama Stage-J regression report to {args.output_path}")


if __name__ == "__main__":
    main()
