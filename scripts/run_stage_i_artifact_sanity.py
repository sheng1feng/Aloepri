import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.defaults import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED
from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.stage_b import prepare_stage_a_model
from src.stage_i_vllm import load_stage_i_hf_bundle, summarize_token_partitions
from src.evaluator import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-I export artifact sanity checks.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--server-dir", default="artifacts/stage_i_vllm/server")
    parser.add_argument("--client-secret", default="artifacts/stage_i_vllm/client/client_secret.pt")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output-path", default=f"{DEFAULT_OUTPUT_DIR}/stage_i/artifact_sanity.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    tokenizer, baseline_model = load_model_and_tokenizer(args.model_dir, device="cpu", dtype=args.dtype)
    stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, seed=args.seed)
    bundle = load_stage_i_hf_bundle(
        args.server_dir,
        client_secret_path=args.client_secret,
        device="cpu",
        dtype=args.dtype,
    )
    exported_model = bundle["model"]
    exported_perm = bundle["perm_vocab"]
    exported_inv_perm = bundle["inv_perm_vocab"]

    embed_weight = stage_a_model.get_input_embeddings().weight.detach().cpu().to(torch.float32)
    exported_embed_weight = exported_model.get_input_embeddings().weight.detach().cpu().to(torch.float32)
    head_weight = stage_a_model.get_output_embeddings().weight.detach().cpu().to(torch.float32)
    exported_head_weight = exported_model.get_output_embeddings().weight.detach().cpu().to(torch.float32)

    parameter_max_diffs: dict[str, float] = {}
    exported_param_map = dict(exported_model.named_parameters())
    for name, tensor in stage_a_model.named_parameters():
        if name not in exported_param_map:
            continue
        diff = (tensor.detach().cpu().to(torch.float32) - exported_param_map[name].detach().cpu().to(torch.float32)).abs().max().item()
        parameter_max_diffs[name] = float(diff)

    model_vocab_size = int(embed_weight.shape[0])
    payload = {
        "stage": "I",
        "phase": "I-A",
        "model_dir": args.model_dir,
        "server_dir": args.server_dir,
        "client_secret": args.client_secret,
        "dtype": args.dtype,
        "seed": args.seed,
        "server_load_success": True,
        "perm_vocab_match_export": bool(torch.equal(perm_vocab.cpu(), exported_perm.cpu())),
        "inv_perm_vocab_match_export": bool(torch.equal(inv_perm_vocab.cpu(), exported_inv_perm.cpu())),
        "embed_weight_max_abs_diff": float((embed_weight - exported_embed_weight).abs().max().item()),
        "lm_head_weight_max_abs_diff": float((head_weight - exported_head_weight).abs().max().item()),
        "max_parameter_abs_diff": max(parameter_max_diffs.values()) if parameter_max_diffs else 0.0,
        "parameter_count_checked": len(parameter_max_diffs),
        "largest_parameter_diffs": dict(sorted(parameter_max_diffs.items(), key=lambda item: item[1], reverse=True)[:10]),
        **summarize_token_partitions(
            tokenizer=tokenizer,
            model_vocab_size=model_vocab_size,
            perm_vocab=perm_vocab,
        ),
    }
    write_json(args.output_path, payload)
    print(f"Saved Stage-I artifact sanity report to {args.output_path}")


if __name__ == "__main__":
    main()
