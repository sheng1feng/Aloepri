from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.defaults import DEFAULT_SYSTEM_PROMPT
from src.model_loader import tokenize_prompt


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def mean_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def top1_equal(logits_a: torch.Tensor, logits_b: torch.Tensor) -> bool:
    return bool(torch.argmax(logits_a, dim=-1).item() == torch.argmax(logits_b, dim=-1).item())


def topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int = 10) -> int:
    topk_a = torch.topk(logits_a, k=k).indices.tolist()
    topk_b = torch.topk(logits_b, k=k).indices.tolist()
    return len(set(topk_a) & set(topk_b))


def safe_token_to_string(tokenizer: Any, token_id: int) -> str:
    try:
        if 0 <= token_id < len(tokenizer):
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token is not None:
                return token
        return f"<unmapped:{token_id}>"
    except Exception:
        return f"<unmapped:{token_id}>"


@torch.inference_mode()
def collect_prompt_outputs(
    model,
    tokenizer,
    prompt: str,
    device: str = "cpu",
    max_new_tokens: int = 8,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    encoded = tokenize_prompt(
        tokenizer,
        prompt,
        system_prompt=system_prompt,
        device=device,
    )
    outputs = model(**encoded)
    logits = outputs.logits.detach().cpu().to(torch.float32)
    last_token_logits = logits[0, -1]
    topk = torch.topk(last_token_logits, k=10)
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    prompt_length = encoded["input_ids"].shape[1]
    generated_ids = generated[0, prompt_length:].detach().cpu()
    greedy_next_token_id = int(torch.argmax(last_token_logits).item())
    input_ids = encoded["input_ids"][0].detach().cpu()
    return {
        "prompt": prompt,
        "input_ids": input_ids.tolist(),
        "decoded_input_text": tokenizer.decode(input_ids, skip_special_tokens=False),
        "last_token_logits": last_token_logits.tolist(),
        "top10_token_ids": topk.indices.tolist(),
        "top10_token_strings": [safe_token_to_string(tokenizer, token_id) for token_id in topk.indices.tolist()],
        "greedy_next_token_id": greedy_next_token_id,
        "greedy_next_token": safe_token_to_string(tokenizer, greedy_next_token_id),
        "generated_token_ids": generated_ids.tolist(),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

