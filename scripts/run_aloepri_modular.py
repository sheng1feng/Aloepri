import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aloepri.config import AloePriConfig
from src.aloepri.engine import AloePriEngine
from src.model_loader import load_model_and_tokenizer
from src.evaluator import max_abs_error
from src.stage_b import TraceRecorder
from src.defaults import DEFAULT_PROMPTS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--layer-count", type=int, default=2)
    parser.add_argument("--output-path", default="outputs/aloepri_modular_test.json")
    parser.add_argument("--seed", type=int, default=20260323)
    args = parser.parse_args()

    # 1. Load baseline
    tokenizer, model = load_model_and_tokenizer(args.model_path, dtype="float32")
    
    # 2. Setup Config
    config = AloePriConfig(
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        num_attention_heads=model.config.num_attention_heads,
        num_key_value_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        rope_theta=getattr(model.config, "rope_theta", 10000.0),
        adapted_layers=list(range(args.layer_count)),
        seed=args.seed,
        alpha_e=0.0,
        alpha_h=0.0,
    )

    # 3. Apply AloePri Obfuscation
    recorder = TraceRecorder()
    engine = AloePriEngine(config, tokenizer)
    obf_model = engine.obfuscate_model(model, recorder=recorder)
    
    # 4. Evaluate on prompts
    results = []
    for prompt in DEFAULT_PROMPTS[:2]: # Test first two prompts
        recorder.clear()
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        # Obfuscate input
        permuted_input_ids = torch.empty_like(input_ids)
        permuted_input_ids[0] = obf_model.perm_vocab[input_ids[0]]
        
        # Run inference
        with torch.no_grad():
            obf_logits = obf_model(permuted_input_ids).logits
            
        # Restore output
        restored_logits = torch.empty_like(obf_logits)
        restored_logits[0, :, obf_model.inv_perm_vocab] = obf_logits[0]
        
        # Get baseline logits
        with torch.no_grad():
            baseline_logits = model(input_ids).logits
            
        error = max_abs_error(baseline_logits, restored_logits)
        results.append({
            "prompt": prompt,
            "max_abs_error": error,
        })
        print(f"Prompt: {prompt[:20]}... Error: {error:.6f}")

    # 5. Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
