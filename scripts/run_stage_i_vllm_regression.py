import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.evaluator import write_json
from src.model_loader import format_chat_prompt, load_model_and_tokenizer, set_global_seed
from src.stage_i_vllm import load_stage_i_hf_bundle
from src.transforms import map_input_ids, unmap_output_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-I vLLM regression if vLLM is available.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--server-dir", default="artifacts/stage_i_vllm/server")
    parser.add_argument("--client-secret", default="artifacts/stage_i_vllm/client/client_secret.pt")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_i/vllm_regression.json")
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


def _run_vllm_generate(llm, mapped_ids: list[int], max_new_tokens: int):
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )
    attempts = [
        lambda: llm.generate(prompt_token_ids=[mapped_ids], sampling_params=sampling_params, use_tqdm=False),
        lambda: llm.generate(prompts=None, prompt_token_ids=[mapped_ids], sampling_params=sampling_params, use_tqdm=False),
        lambda: llm.generate([{"prompt_token_ids": mapped_ids}], sampling_params=sampling_params, use_tqdm=False),
    ]
    last_error = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("No compatible vLLM generate signature worked.")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    os.environ.setdefault("VLLM_TARGET_DEVICE", args.device)

    try:
        from vllm import LLM
    except Exception as exc:
        payload = {
            "stage": "I",
            "phase": "I-A",
            "backend": "vllm",
            "available": False,
            "skipped": True,
            "reason": f"vllm is unavailable in the current environment: {exc}",
            "server_dir": args.server_dir,
            "client_secret": args.client_secret,
        }
        write_json(args.output_path, payload)
        print(f"Saved skipped Stage-I vLLM regression report to {args.output_path}")
        return

    baseline_tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    bundle = load_stage_i_hf_bundle(
        args.server_dir,
        client_secret_path=args.client_secret,
        device="cpu",
        dtype=args.dtype,
    )
    tokenizer = bundle["tokenizer"]
    perm_vocab = bundle["perm_vocab"]
    inv_perm_vocab = bundle["inv_perm_vocab"]
    if perm_vocab is None or inv_perm_vocab is None:
        raise ValueError("client secret is required for Stage-I vLLM regression")

    try:
        llm = LLM(
            model=args.server_dir,
            tokenizer=args.server_dir,
            trust_remote_code=True,
            dtype=args.dtype,
            enforce_eager=True,
        )
    except Exception as exc:
        payload = {
            "stage": "I",
            "phase": "I-A",
            "backend": "vllm",
            "available": True,
            "skipped": True,
            "reason": f"vllm import succeeded but runtime initialization failed: {type(exc).__name__}: {exc}",
            "server_dir": args.server_dir,
            "client_secret": args.client_secret,
            "dtype": args.dtype,
            "device": args.device,
            "seed": args.seed,
        }
        write_json(args.output_path, payload)
        print(f"Saved skipped Stage-I vLLM regression report to {args.output_path}")
        return

    prompt_results: list[dict] = []
    for index, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        text = format_chat_prompt(tokenizer, prompt)
        encoded = tokenizer(text, return_tensors="pt")
        mapped_ids = map_input_ids(encoded["input_ids"], perm_vocab)[0].tolist()
        outputs = _run_vllm_generate(llm, mapped_ids=mapped_ids, max_new_tokens=args.max_new_tokens)
        output_token_ids = outputs[0].outputs[0].token_ids
        restored_ids = unmap_output_ids(torch.tensor(output_token_ids, dtype=torch.long), inv_perm_vocab).tolist()

        baseline_generated_ids = greedy_generate_plain(
            baseline_model,
            encoded["input_ids"],
            max_new_tokens=args.max_new_tokens,
        )[0].cpu().tolist()

        prompt_results.append(
            {
                "prompt_id": index,
                "prompt": prompt,
                "mapped_input_ids": mapped_ids,
                "vllm_generated_ids_obfuscated": output_token_ids,
                "vllm_generated_ids_restored": restored_ids,
                "baseline_generated_ids": baseline_generated_ids,
                "generated_ids_exact_match": baseline_generated_ids == restored_ids,
                "generated_text_exact_match": baseline_tokenizer.decode(baseline_generated_ids, skip_special_tokens=True)
                == tokenizer.decode(torch.tensor(restored_ids), skip_special_tokens=True),
            }
        )

    count = max(len(prompt_results), 1)
    payload = {
        "stage": "I",
        "phase": "I-A",
        "backend": "vllm",
        "available": True,
        "skipped": False,
        "server_dir": args.server_dir,
        "client_secret": args.client_secret,
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "summary": {
            "prompt_count": len(prompt_results),
            "generated_ids_exact_match_rate": sum(1.0 for item in prompt_results if item["generated_ids_exact_match"]) / count,
            "generated_text_exact_match_rate": sum(1.0 for item in prompt_results if item["generated_text_exact_match"]) / count,
        },
        "prompts": prompt_results,
    }
    write_json(args.output_path, payload)
    print(f"Saved Stage-I vLLM regression report to {args.output_path}")


if __name__ == "__main__":
    main()
