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
    parser = argparse.ArgumentParser(description="Run HF regression for Stage-I Phase-2 square-monomial embed/head-only export.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--server-dir", default="artifacts/stage_i_phase2_square/server")
    parser.add_argument("--client-secret", default="artifacts/stage_i_phase2_square/client/client_secret.pt")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_i/phase2_square_regression.json")
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
def greedy_generate_candidate(model, input_ids: torch.Tensor, perm_vocab: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    current_plain_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        mapped_ids = map_input_ids(current_plain_ids, perm_vocab)
        attention_mask = torch.ones_like(mapped_ids)
        logits_perm = model(input_ids=mapped_ids, attention_mask=attention_mask).logits.detach().to(torch.float32)
        restored_logits = restore_logits(logits_perm[:, -1, :], perm_vocab)
        next_token = torch.argmax(restored_logits, dim=-1, keepdim=True)
        current_plain_ids = torch.cat([current_plain_ids, next_token.to(current_plain_ids.device)], dim=1)
    return current_plain_ids[:, input_ids.shape[1] :]


def summarize(items: list[dict]) -> dict[str, float | bool]:
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
        "candidate_has_nan_or_inf": any(item["candidate_has_nan_or_inf"] for item in items),
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
    candidate_model = bundle["model"]
    perm_vocab = bundle["perm_vocab"]
    metadata = bundle["metadata"]
    if perm_vocab is None:
        raise ValueError("client secret is required for Stage-I Phase-2 square regression")

    candidate_embed = candidate_model.get_input_embeddings().weight.detach().cpu().to(torch.float32)
    baseline_embed = baseline_model.get_input_embeddings().weight.detach().cpu().to(torch.float32)
    candidate_head = candidate_model.get_output_embeddings().weight.detach().cpu().to(torch.float32)
    baseline_head = baseline_model.get_output_embeddings().weight.detach().cpu().to(torch.float32)

    prompt_results: list[dict] = []
    for index, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        encoded = tokenize_prompt(tokenizer, prompt, device="cpu")
        baseline_logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)

        mapped_ids = map_input_ids(encoded["input_ids"], perm_vocab)
        candidate_logits_perm = candidate_model(
            input_ids=mapped_ids,
            attention_mask=encoded["attention_mask"],
        ).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(candidate_logits_perm, perm_vocab)

        baseline_generated_ids = greedy_generate_plain(baseline_model, encoded["input_ids"], args.max_new_tokens)[0].cpu()
        candidate_generated_ids = greedy_generate_candidate(candidate_model, encoded["input_ids"], perm_vocab, args.max_new_tokens)[0].cpu()

        prompt_results.append(
            {
                "prompt_id": index,
                "prompt": prompt,
                "full_logits_max_abs_error": max_abs_error(baseline_logits, restored_logits),
                "full_logits_mean_abs_error": mean_abs_error(baseline_logits, restored_logits),
                "last_token_logits_max_abs_error": max_abs_error(baseline_logits[0, -1], restored_logits[0, -1]),
                "last_token_logits_mean_abs_error": mean_abs_error(baseline_logits[0, -1], restored_logits[0, -1]),
                "greedy_first_token_match": int(torch.argmax(baseline_logits[0, -1]).item()) == int(torch.argmax(restored_logits[0, -1]).item()),
                "generated_ids_exact_match": baseline_generated_ids.tolist() == candidate_generated_ids.tolist(),
                "generated_text_exact_match": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True)
                == tokenizer.decode(candidate_generated_ids, skip_special_tokens=True),
                "baseline_generated_text": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True),
                "candidate_generated_text": tokenizer.decode(candidate_generated_ids, skip_special_tokens=True),
                "baseline_has_nan_or_inf": not bool(torch.isfinite(baseline_logits).all().item()),
                "candidate_has_nan_or_inf": not bool(torch.isfinite(candidate_logits_perm).all().item()),
            }
        )

    payload = {
        "stage": "I",
        "phase": "I-B",
        "variant": "square_monomial_embed_head_only",
        "model_dir": args.model_dir,
        "server_dir": args.server_dir,
        "client_secret": args.client_secret,
        "dtype": args.dtype,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "shape_summary": {
            "server_load_success": True,
            "embed_shape_match_baseline": list(candidate_embed.shape) == list(baseline_embed.shape),
            "lm_head_shape_match_baseline": list(candidate_head.shape) == list(baseline_head.shape),
            "candidate_embed_shape": list(candidate_embed.shape),
            "baseline_embed_shape": list(baseline_embed.shape),
            "candidate_lm_head_shape": list(candidate_head.shape),
            "baseline_lm_head_shape": list(baseline_head.shape),
        },
        "square_transform": metadata.get("square_transform", {}),
        "summary": summarize(prompt_results),
        "prompts": prompt_results,
    }
    write_json(args.output_path, payload)
    print(f"Saved Stage-I Phase-2 square regression report to {args.output_path}")


if __name__ == "__main__":
    main()
