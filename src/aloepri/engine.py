from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Iterable, Dict, Any

from src.aloepri.adapters.qwen import QwenArchitectureAdapter
from src.aloepri.config import AloePriConfig
from src.aloepri.keys import build_aloepri_keys, AloePriKeys
from src.aloepri.layers.norm import wrap_norm
from src.aloepri.layers.ffn import wrap_mlp
from src.aloepri.layers.attention import wrap_attention
from src.aloepri.layers.embeddings import wrap_embedding, wrap_head
from src.key_manager import ordinary_token_ids
from src.stage_b import TraceRecorder, prepare_stage_a_model
from src.stage_f import _clear_copied_hooks
from src.stage_g import KeyMatDecoderLayerHandoff
from src.keymat_norm import estimate_kappa_for_keymat

class AloePriEngine:
    """
    Main engine to apply AloePri obfuscation to a model.
    """
    def __init__(self, config: AloePriConfig, tokenizer: Any):
        self.config = config
        self.tokenizer = tokenizer
        self.keys = build_aloepri_keys(config, tokenizer.vocab_size)

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        tokenizer: Any,
        **config_overrides,
    ) -> "AloePriEngine":
        QwenArchitectureAdapter.from_model(model)
        config = AloePriConfig.from_model(model, **config_overrides)
        return cls(config=config, tokenizer=tokenizer)

    def obfuscate_model(self, model: nn.Module, recorder: Optional[TraceRecorder] = None) -> nn.Module:
        """
        Applies full AloePri obfuscation to the model.
        """
        QwenArchitectureAdapter.from_model(model)
        # 1. Prepare Stage A model (token permutation)
        stage_a_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(
            model, self.tokenizer, self.config.seed
        )
        _clear_copied_hooks(stage_a_model)
        
        target_dtype = stage_a_model.dtype
        device = stage_a_model.device
        
        # 2. Add recording hooks if recorder is provided
        if recorder is not None:
            self._add_recording_hooks(stage_a_model, recorder)

        # 3. Obfuscate Embedding
        movable_ids = ordinary_token_ids(self.tokenizer)
        stage_a_model.model.embed_tokens = wrap_embedding(
            embed_module=stage_a_model.model.embed_tokens,
            keymat_transform=self.keys.keymat_transform,
            alpha_e=self.config.alpha_e,
            seed=self.config.seed,
            movable_ids=movable_ids,
            recorder=recorder,
            record_name="embed_tokens_out"
        ).to(device=device, dtype=target_dtype)

        # 4. Obfuscate Layers
        total_layers = stage_a_model.config.num_hidden_layers
        adapted_layers = sorted(self.config.adapted_layers)
        
        # Handoff layer logic
        handoff_layer = None
        if adapted_layers and adapted_layers[-1] < total_layers - 1:
            handoff_layer = adapted_layers[-1] + 1

        for layer_idx in adapted_layers:
            layer = stage_a_model.model.layers[layer_idx]
            
            # Get input/post-attn weights for fusion
            input_norm_weight = layer.input_layernorm.weight.detach().to(torch.float32)
            post_norm_weight = layer.post_attention_layernorm.weight.detach().to(torch.float32)
            
            # Estimate kappas
            input_kappa = estimate_kappa_for_keymat(
                self.keys.keymat_transform,
                hidden_size=self.config.hidden_size,
                num_samples=1024,
                seed=self.config.seed + 2000 + layer_idx
            )
            post_attn_kappa = estimate_kappa_for_keymat(
                self.keys.keymat_transform,
                hidden_size=self.config.hidden_size,
                num_samples=1024,
                seed=self.config.seed + 3000 + layer_idx
            )
            
            # Apply obfuscation
            layer.input_layernorm = wrap_norm(
                layer.input_layernorm,
                keymat_transform=self.keys.keymat_transform,
                kappa=input_kappa,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_input_norm_out"
            ).to(device=device, dtype=target_dtype)
            
            layer.self_attn = wrap_attention(
                attention_module=layer.self_attn,
                keymat_transform=self.keys.keymat_transform,
                input_norm_weight=input_norm_weight,
                attention_profile=self.config.attention_profile,
                seed=self.config.seed + 1000 + layer_idx,
                qk_scale_range=self.config.qk_scale_range,
                beta=self.config.beta,
                gamma=self.config.gamma,
                rope_base=self.config.rope_theta,
                layer_idx=layer_idx,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_attn_out"
            ).to(device=device, dtype=target_dtype)
            
            layer.post_attention_layernorm = wrap_norm(
                layer.post_attention_layernorm,
                keymat_transform=self.keys.keymat_transform,
                kappa=post_attn_kappa,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_post_attn_norm_out"
            ).to(device=device, dtype=target_dtype)
            
            # FFN Transform - build it if needed
            from src.obfuscate_ffn import build_ffn_transform, generate_ffn_permutation, generate_ffn_scaling
            ffn_intermediate_size = layer.mlp.gate_proj.out_features
            ffn_perm = generate_ffn_permutation(ffn_intermediate_size, seed=self.config.seed + 5000 + layer_idx)
            ffn_scale = generate_ffn_scaling(ffn_intermediate_size, scale_range=self.config.ffn_scale_range, seed=self.config.seed + 6000 + layer_idx)
            ffn_transform = build_ffn_transform(ffn_perm, ffn_scale)
            
            layer.mlp = wrap_mlp(
                mlp_module=layer.mlp,
                keymat_transform=self.keys.keymat_transform,
                ffn_transform=ffn_transform,
                input_norm_weight=post_norm_weight,
                recorder=recorder,
                record_name=f"layer_{layer_idx}_mlp_out"
            ).to(device=device, dtype=target_dtype)

        # 5. Handle Handoff
        if handoff_layer is not None:
            original_handoff = stage_a_model.model.layers[handoff_layer]
            stage_a_model.model.layers[handoff_layer] = KeyMatDecoderLayerHandoff(
                layer_module=original_handoff,
                keymat_transform=self.keys.keymat_transform,
                layer_idx=handoff_layer,
                recorder=recorder,
            )

        # 6. Obfuscate Head and Final Norm
        if handoff_layer is None:
            # Full model obfuscated - fuse final norm into head
            final_norm = stage_a_model.model.norm
            final_norm_weight = final_norm.weight.detach().to(torch.float32)
            
            final_kappa = estimate_kappa_for_keymat(
                self.keys.keymat_transform,
                hidden_size=self.config.hidden_size,
                num_samples=1024,
                seed=self.config.seed + 12345
            )
            
            stage_a_model.model.norm = wrap_norm(
                final_norm,
                keymat_transform=self.keys.keymat_transform,
                kappa=final_kappa,
                recorder=recorder,
                record_name="final_norm_out"
            ).to(device=device, dtype=target_dtype)
            
            stage_a_model.lm_head = wrap_head(
                head_module=stage_a_model.lm_head,
                keymat_transform=self.keys.keymat_transform,
                alpha_h=self.config.alpha_h,
                seed=self.config.seed + 202,
                movable_ids=movable_ids,
                final_norm_weight=final_norm_weight,
                expects_obfuscated_input=True,
                recorder=recorder,
                record_name="head_out"
            ).to(device=device, dtype=target_dtype)
        else:
            # Handoff - head is not fused with final norm because final norm input is already restored
            stage_a_model.lm_head = wrap_head(
                head_module=stage_a_model.lm_head,
                keymat_transform=self.keys.keymat_transform,
                alpha_h=self.config.alpha_h,
                seed=self.config.seed + 202,
                movable_ids=movable_ids,
                expects_obfuscated_input=False,
                recorder=recorder,
                record_name="head_out"
            ).to(device=device, dtype=target_dtype)

        # Attach permutation for easy access
        stage_a_model.perm_vocab = perm_vocab
        stage_a_model.inv_perm_vocab = inv_perm_vocab
        
        return stage_a_model

    def _add_recording_hooks(self, model: nn.Module, recorder: TraceRecorder):
        # Add basic recording hooks for layers if needed
        pass
