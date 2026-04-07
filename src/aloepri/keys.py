from __future__ import annotations

from dataclasses import dataclass
import torch
from typing import Tuple, Dict, Any, List

from src.keymat import KeyMatBases, KeyMatTransform, init_keymat_bases, generate_keymat, generate_inv_keymat
from src.attention_keys import build_attention_complex_config, AttentionComplexConfig
from src.obfuscate_ffn import FFNTransform, build_ffn_transform, generate_ffn_permutation, generate_ffn_scaling
from src.aloepri.config import AloePriConfig

@dataclass(frozen=True)
class AloePriKeys:
    config: AloePriConfig
    token_perm: torch.Tensor
    inv_token_perm: torch.Tensor
    keymat_transform: KeyMatTransform
    # Layer-specific keys (per-layer)
    attention_configs: Dict[int, AttentionComplexConfig]
    ffn_transforms: Dict[int, FFNTransform]
    # Normalization kappa values
    kappas: Dict[int, Dict[str, float]] # e.g., {layer_idx: {"input_norm": 1.0, "post_attn_norm": 1.0}}
    final_norm_kappa: float

def build_aloepri_keys(config: AloePriConfig, tokenizer_vocab_size: int) -> AloePriKeys:
    # 1. Token permutation
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    token_perm = torch.randperm(tokenizer_vocab_size, generator=generator)
    inv_token_perm = torch.empty_like(token_perm)
    inv_token_perm[token_perm] = torch.arange(tokenizer_vocab_size)

    # 2. KeyMat transform
    keymat_transform = build_keymat_transform_from_config(config)

    # 3. Layer-specific keys
    attention_configs = {}
    ffn_transforms = {}
    kappas = {}
    
    num_groups = config.num_attention_heads // config.num_key_value_heads

    for layer_idx in config.adapted_layers:
        # Attention keys
        attn_cfg = build_attention_complex_config(
            profile=config.attention_profile,
            head_dim=config.head_dim,
            num_kv_heads=config.num_key_value_heads,
            num_groups=num_groups,
            seed=config.seed + 1000 + layer_idx,
            qk_scale_range=config.qk_scale_range,
            beta=config.beta,
            gamma=config.gamma,
            rope_base=config.rope_theta,
        )
        attention_configs[layer_idx] = attn_cfg

        # FFN keys
        # Qwen-specific: gate_proj and up_proj share intermediate_size
        # We need intermediate_size, which we can get from the model later, 
        # but for now we'll assume we know it or pass it. 
        # Actually, let's keep it simple and assume a standard size or handle it in the engine.
        # For now, let's defer FFN and kappa generation to when we have the model instance.
        pass

    return AloePriKeys(
        config=config,
        token_perm=token_perm,
        inv_token_perm=inv_token_perm,
        keymat_transform=keymat_transform,
        attention_configs=attention_configs,
        ffn_transforms={}, # Filled later
        kappas={},         # Filled later
        final_norm_kappa=1.0, # Filled later
    )

def build_keymat_transform_from_config(config: AloePriConfig) -> KeyMatTransform:
    from src.keymat import build_keymat_transform
    return build_keymat_transform(
        d=config.hidden_size,
        h=config.expansion_size,
        lam=config.lam,
        init_seed=config.seed + 100,
        dtype=getattr(torch, config.dtype)
    )
