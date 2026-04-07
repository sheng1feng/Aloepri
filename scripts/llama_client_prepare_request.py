import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from src.llama_local_dev import tokenize_llama_prompt
from src.transforms import map_input_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare client-side mapped token ids for a Llama obfuscated HF checkpoint.")
    parser.add_argument("--server-dir", required=True, help="Path to the obfuscated server HF directory.")
    parser.add_argument("--client-secret", required=True, help="Path to client_secret.pt.")
    parser.add_argument("--prompt", required=True, help="Plaintext prompt on the client side.")
    parser.add_argument("--output-path", default="outputs/llama_client_request.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.server_dir, trust_remote_code=True)
    secret = torch.load(args.client_secret, map_location="cpu")
    perm_vocab = torch.as_tensor(secret["perm_vocab"], dtype=torch.long)

    encoded = tokenize_llama_prompt(tokenizer, args.prompt, device="cpu")
    mapped_ids = map_input_ids(encoded["input_ids"], perm_vocab)

    payload = {
        "server_dir": args.server_dir,
        "client_secret": args.client_secret,
        "prompt": args.prompt,
        "input_ids": encoded["input_ids"][0].tolist(),
        "mapped_input_ids": mapped_ids[0].tolist(),
        "attention_mask": encoded["attention_mask"][0].tolist(),
        "note": "Send mapped_input_ids + attention_mask to the server model. Do not send plaintext input_ids.",
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved client request payload to {output_path}")


if __name__ == "__main__":
    main()
