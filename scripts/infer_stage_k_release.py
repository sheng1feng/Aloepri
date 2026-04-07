import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.llama_local_dev import tokenize_llama_prompt
from src.stage_i_vllm import load_stage_i_hf_bundle
from src.transforms import map_input_ids, restore_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer from a Stage-K release profile.")
    parser.add_argument("--release-dir", default="artifacts/stage_k_release")
    parser.add_argument("--profile", default="tiny_a")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="float32")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    catalog = json.loads((release_dir / "catalog.json").read_text(encoding="utf-8"))
    profile_map = {item["name"]: item for item in catalog["profiles"]}
    if args.profile not in profile_map:
        raise ValueError(f"Unknown Stage-K profile: {args.profile}")
    profile_dir = release_dir / profile_map[args.profile]["server_dir"]
    client_secret = release_dir / profile_map[args.profile]["client_secret"]

    bundle = load_stage_i_hf_bundle(
        profile_dir,
        client_secret_path=client_secret,
        device="cpu",
        dtype=args.dtype,
    )
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    perm_vocab = bundle["perm_vocab"]

    encoded = tokenize_llama_prompt(tokenizer, args.prompt, device="cpu")
    current_ids = encoded["input_ids"].clone()
    generated_ids: list[int] = []

    for _ in range(args.max_new_tokens):
        mapped_ids = map_input_ids(current_ids, perm_vocab)
        logits_perm = model(input_ids=mapped_ids, attention_mask=torch.ones_like(mapped_ids)).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(logits_perm[0, -1], perm_vocab)
        next_token = int(torch.argmax(restored_logits).item())
        generated_ids.append(next_token)
        current_ids = torch.cat([current_ids, torch.tensor([[next_token]], dtype=current_ids.dtype)], dim=1)

    print(tokenizer.decode(torch.tensor(generated_ids), skip_special_tokens=True))


if __name__ == "__main__":
    main()
