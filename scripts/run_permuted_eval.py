import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPTS,
    DEFAULT_SEED,
)
from src.evaluator import (
    max_abs_error,
    mean_abs_error,
    safe_token_to_string,
    top1_equal,
    topk_overlap,
    write_json,
)
from src.key_manager import (
    generate_vocab_permutation,
    invert_permutation,
    ordinary_token_ids,
    save_permutation,
)
from src.model_loader import load_model_and_tokenizer, set_global_seed, tokenize_prompt
from src.obfuscate_embed_head import build_vocab_permuted_model
from src.transforms import map_input_ids, restore_logits, unmap_output_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stage-A vocab permutation evaluation.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--baseline-dir", default=f"{DEFAULT_OUTPUT_DIR}/baseline")
    parser.add_argument("--output-dir", default=f"{DEFAULT_OUTPUT_DIR}/permuted_eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_baseline_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, model = load_model_and_tokenizer(
        model_name=args.model_dir,
        device=args.device,
        dtype=args.dtype,
    )
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    movable_ids = ordinary_token_ids(tokenizer)
    perm_vocab = generate_vocab_permutation(
        vocab_size=model_vocab_size,
        seed=args.seed,
        movable_ids=movable_ids,
    )
    inv_perm_vocab = invert_permutation(perm_vocab)
    permuted_model = build_vocab_permuted_model(model, perm_vocab, inv_perm_vocab)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    permutation_path = output_dir / "vocab_permutation.pt"
    save_permutation(
        perm=perm_vocab,
        inv_perm=inv_perm_vocab,
        path=permutation_path,
        metadata={
            "seed": args.seed,
            "model_vocab_size": model_vocab_size,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "tokenizer_length": len(tokenizer),
            "movable_count": int(movable_ids.numel()),
        },
    )

    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    for prompt_id, prompt in enumerate(DEFAULT_PROMPTS, start=1):
        baseline_path = baseline_dir / f"{prompt_id:02d}.json"
        baseline = load_baseline_result(baseline_path)
        encoded = tokenize_prompt(tokenizer, prompt, device=args.device)
        mapped_input_ids = map_input_ids(encoded["input_ids"], perm_vocab)
        encoded["input_ids"] = mapped_input_ids
        outputs = permuted_model(**encoded)
        logits_perm = outputs.logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(logits_perm[0, -1], perm_vocab).cpu()
        baseline_logits = torch.tensor(baseline["last_token_logits"], dtype=torch.float32)

        generated = permuted_model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        prompt_length = encoded["input_ids"].shape[1]
        generated_perm_ids = generated[0, prompt_length:].detach().cpu()
        generated_restored_ids = unmap_output_ids(generated_perm_ids, inv_perm_vocab)
        restored_greedy_token_id = int(torch.argmax(restored_logits).item())

        top10_ids = torch.topk(restored_logits, k=10).indices.tolist()
        result = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "baseline_input_ids": baseline["input_ids"],
            "mapped_input_ids": mapped_input_ids[0].detach().cpu().tolist(),
            "last_token_logits_restored": restored_logits.tolist(),
            "max_abs_error": max_abs_error(restored_logits, baseline_logits),
            "mean_abs_error": mean_abs_error(restored_logits, baseline_logits),
            "top10_overlap_count": topk_overlap(restored_logits, baseline_logits, k=10),
            "baseline_top10_token_ids": baseline["top10_token_ids"],
            "restored_top10_token_ids": top10_ids,
            "restored_top10_token_strings": [safe_token_to_string(tokenizer, token_id) for token_id in top10_ids],
            "baseline_greedy_next_token_id": baseline["greedy_next_token_id"],
            "restored_greedy_next_token_id": restored_greedy_token_id,
            "greedy_match": bool(restored_greedy_token_id == baseline["greedy_next_token_id"]),
            "top1_equal": top1_equal(restored_logits, baseline_logits),
            "generated_token_ids_perm_space": generated_perm_ids.tolist(),
            "generated_token_ids_restored": generated_restored_ids.tolist(),
            "generated_text_restored": tokenizer.decode(generated_restored_ids, skip_special_tokens=True),
            "baseline_generated_text": baseline["generated_text"],
            "generated_text_match": bool(tokenizer.decode(generated_restored_ids, skip_special_tokens=True) == baseline["generated_text"]),
            "permutation_path": str(permutation_path),
            "seed": args.seed,
        }
        write_json(output_dir / f"{prompt_id:02d}.json", result)
        print(f"Saved permuted evaluation prompt {prompt_id} to {output_dir / f'{prompt_id:02d}.json'}")


if __name__ == "__main__":
    main()
