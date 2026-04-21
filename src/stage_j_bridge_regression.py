from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_PROMPTS
from src.evaluator import max_abs_error, mean_abs_error
from src.model_loader import tokenize_prompt
from src.stage_h_pretrained import load_stage_h_pretrained
from src.transforms import map_input_ids, restore_logits


def summarize_bridge_items(items: list[dict[str, Any]]) -> dict[str, float]:
    count = max(len(items), 1)
    return {
        "prompt_count": len(items),
        "avg_restored_full_logits_max_abs_error": sum(float(item["restored_full_logits_max_abs_error"]) for item in items) / count,
        "avg_restored_full_logits_mean_abs_error": sum(float(item["restored_full_logits_mean_abs_error"]) for item in items) / count,
        "avg_restored_last_token_max_abs_error": sum(float(item["restored_last_token_max_abs_error"]) for item in items) / count,
        "avg_restored_last_token_mean_abs_error": sum(float(item["restored_last_token_mean_abs_error"]) for item in items) / count,
        "generated_ids_exact_match_rate": sum(1.0 for item in items if item["generated_ids_exact_match"]) / count,
        "generated_text_exact_match_rate": sum(1.0 for item in items if item["generated_text_exact_match"]) / count,
    }


@torch.inference_mode()
def _greedy_generate_with_perm(model, input_ids: torch.Tensor, perm_vocab: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    current_plain_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        mapped_ids = map_input_ids(current_plain_ids, perm_vocab)
        attention_mask = torch.ones_like(mapped_ids)
        logits_perm = model(input_ids=mapped_ids, attention_mask=attention_mask).logits.detach().to(torch.float32)
        restored = restore_logits(logits_perm[:, -1, :], perm_vocab)
        next_token = torch.argmax(restored, dim=-1, keepdim=True)
        current_plain_ids = torch.cat([current_plain_ids, next_token.to(current_plain_ids.device)], dim=1)
    return current_plain_ids[:, input_ids.shape[1] :]


def run_stage_j_bridge_regression(
    *,
    buffered_server_dir: str = "artifacts/stage_j_qwen_redesign/server",
    bridge_server_dir: str = "artifacts/stage_j_qwen_redesign_standard/server",
    prompts: list[str] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    buffered_bundle = load_stage_h_pretrained(buffered_server_dir)
    buffered_model = buffered_bundle["stage_model"].eval()
    tokenizer = buffered_bundle["tokenizer"]
    perm_vocab = torch.as_tensor(buffered_bundle["perm_vocab"], dtype=torch.long)

    bridge_model = AutoModelForCausalLM.from_pretrained(bridge_server_dir, trust_remote_code=True).eval()
    bridge_tokenizer = AutoTokenizer.from_pretrained(bridge_server_dir, trust_remote_code=True)

    active_prompts = prompts or list(DEFAULT_PROMPTS)
    prompt_results: list[dict[str, Any]] = []

    for index, prompt in enumerate(active_prompts, start=1):
        encoded = tokenize_prompt(tokenizer, prompt, device="cpu")
        mapped_ids = map_input_ids(encoded["input_ids"], perm_vocab)
        attention_mask = encoded["attention_mask"]

        buffered_logits_perm = buffered_model(input_ids=mapped_ids, attention_mask=attention_mask).logits.detach().cpu().to(torch.float32)
        bridge_logits_perm = bridge_model(input_ids=mapped_ids, attention_mask=attention_mask).logits.detach().cpu().to(torch.float32)
        buffered_restored = restore_logits(buffered_logits_perm, perm_vocab)
        bridge_restored = restore_logits(bridge_logits_perm, perm_vocab)

        buffered_generated = _greedy_generate_with_perm(buffered_model, encoded["input_ids"], perm_vocab, max_new_tokens)[0].cpu()
        bridge_generated = _greedy_generate_with_perm(bridge_model, encoded["input_ids"], perm_vocab, max_new_tokens)[0].cpu()

        prompt_results.append(
            {
                "prompt_id": index,
                "prompt": prompt,
                "restored_full_logits_max_abs_error": max_abs_error(buffered_restored, bridge_restored),
                "restored_full_logits_mean_abs_error": mean_abs_error(buffered_restored, bridge_restored),
                "restored_last_token_max_abs_error": max_abs_error(buffered_restored[0, -1], bridge_restored[0, -1]),
                "restored_last_token_mean_abs_error": mean_abs_error(buffered_restored[0, -1], bridge_restored[0, -1]),
                "generated_ids_exact_match": buffered_generated.tolist() == bridge_generated.tolist(),
                "generated_text_exact_match": tokenizer.decode(buffered_generated, skip_special_tokens=True)
                == bridge_tokenizer.decode(bridge_generated, skip_special_tokens=True),
                "buffered_generated_ids": buffered_generated.tolist(),
                "bridge_generated_ids": bridge_generated.tolist(),
                "buffered_generated_text": tokenizer.decode(buffered_generated, skip_special_tokens=True),
                "bridge_generated_text": bridge_tokenizer.decode(bridge_generated, skip_special_tokens=True),
            }
        )

    return {
        "stage": "J",
        "buffered_server_dir": buffered_server_dir,
        "bridge_server_dir": bridge_server_dir,
        "summary": summarize_bridge_items(prompt_results),
        "prompts": prompt_results,
    }
