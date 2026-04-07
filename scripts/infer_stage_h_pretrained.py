import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.model_loader import format_chat_prompt
from src.stage_h_pretrained import load_stage_h_pretrained
from src.transforms import map_input_ids, restore_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer with a pretrained-like stage-H export.")
    parser.add_argument("--server-dir", default="artifacts/stage_h_pretrained/server")
    parser.add_argument("--client-secret", default="artifacts/stage_h_pretrained/client/client_secret.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    bundle = load_stage_h_pretrained(args.server_dir, client_secret_path=args.client_secret)
    tokenizer = bundle["tokenizer"]
    stage_model = bundle["stage_model"]
    perm_vocab = bundle["perm_vocab"]

    text = format_chat_prompt(tokenizer, args.prompt)
    encoded = tokenizer(text, return_tensors="pt")
    current_ids = encoded["input_ids"].clone()
    generated_ids: list[int] = []

    for _ in range(args.max_new_tokens):
        mapped_ids = map_input_ids(current_ids, perm_vocab)
        logits_perm = stage_model(input_ids=mapped_ids, attention_mask=encoded.get("attention_mask")).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(logits_perm[0, -1], perm_vocab)
        next_token = int(torch.argmax(restored_logits).item())
        generated_ids.append(next_token)
        current_ids = torch.cat([current_ids, torch.tensor([[next_token]], dtype=current_ids.dtype)], dim=1)
        encoded["attention_mask"] = torch.ones_like(current_ids)

    print(tokenizer.decode(torch.tensor(generated_ids), skip_special_tokens=True))


if __name__ == "__main__":
    main()
