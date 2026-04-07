import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer

from src.transforms import unmap_output_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore obfuscated generated token ids back to plaintext token ids on the client side.")
    parser.add_argument("--client-secret", required=True, help="Path to client_secret.pt.")
    parser.add_argument("--server-dir", required=True, help="Path to the obfuscated server HF directory, used for tokenizer decode.")
    parser.add_argument("--mapped-token-ids", default=None, help="Comma-separated obfuscated token ids from the server.")
    parser.add_argument("--input-json", default=None, help="Optional JSON file containing `generated_token_ids` or `mapped_token_ids`.")
    parser.add_argument("--output-path", default="outputs/llama_client_restored.json")
    return parser.parse_args()


def _load_ids(args: argparse.Namespace) -> list[int]:
    if args.input_json is not None:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        if "generated_token_ids" in payload:
            return [int(x) for x in payload["generated_token_ids"]]
        if "mapped_token_ids" in payload:
            return [int(x) for x in payload["mapped_token_ids"]]
        raise ValueError("input-json must contain `generated_token_ids` or `mapped_token_ids`.")
    if args.mapped_token_ids is None:
        raise ValueError("Provide either --mapped-token-ids or --input-json.")
    return [int(part.strip()) for part in args.mapped_token_ids.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    mapped_ids = _load_ids(args)
    secret = torch.load(args.client_secret, map_location="cpu")
    inv_perm_vocab = torch.as_tensor(secret["inv_perm_vocab"], dtype=torch.long)
    tokenizer = AutoTokenizer.from_pretrained(args.server_dir, trust_remote_code=True)

    mapped_tensor = torch.tensor(mapped_ids, dtype=torch.long)
    restored = unmap_output_ids(mapped_tensor, inv_perm_vocab)
    decoded = tokenizer.decode(restored, skip_special_tokens=True)

    payload = {
        "mapped_token_ids": mapped_ids,
        "restored_token_ids": restored.tolist(),
        "decoded_text": decoded,
        "note": "The server should only see mapped_token_ids. Plaintext text is recovered only on the client.",
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved restored client output to {output_path}")
    print(decoded)


if __name__ == "__main__":
    main()
