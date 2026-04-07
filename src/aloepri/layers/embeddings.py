from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from src.keymat import KeyMatTransform
from src.keymat_embed_head import build_keymat_embed_head_artifacts, KeyMatEmbeddingWrapper, KeyMatHeadWrapper, add_head_noise
from src.aloepri.layers.base import ObfuscatedLayer

class AloePriEmbedding(ObfuscatedLayer):
    """
    Standardized AloePri Embedding that handles KeyMat transform and noise.
    """
    def __init__(
        self,
        embed_module: nn.Module,
        keymat_transform: KeyMatTransform,
        alpha_e: float,
        seed: int,
        movable_ids: torch.Tensor,
        recorder: Optional[object] = None,
        record_name: Optional[str] = None,
    ) -> None:
        super().__init__(recorder=recorder, record_name=record_name)
        
        # This function builds the artifacts (permuted and noised weight)
        from src.keymat_embed_head import build_keymat_embed_head_artifacts
        
        # Mocking Stage A model just to get the artifacts
        class MockModel:
            def __init__(self, embed):
                self.model = type('model', (), {'embed_tokens': embed})
                self.get_input_embeddings = lambda: embed
                self.get_output_embeddings = lambda: embed
                
        artifacts = build_keymat_embed_head_artifacts(
            stage_a_model=MockModel(embed_module),
            keymat_transform=keymat_transform,
            alpha_e=alpha_e,
            alpha_h=0.0, # Handled separately in Head
            seed=seed,
            movable_ids=movable_ids,
        )
        
        self.wrapper = KeyMatEmbeddingWrapper(
            obfuscated_weight=artifacts.embed_weight_obf,
            base_weight_for_recording=embed_module.weight.detach().cpu().to(torch.float32),
            recorder=recorder,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.wrapper(input_ids)
        self.record(output)
        return output

class AloePriHead(ObfuscatedLayer):
    """
    Standardized AloePri Head that handles KeyMat transform and noise.
    """
    def __init__(
        self,
        head_module: nn.Module,
        keymat_transform: KeyMatTransform,
        alpha_h: float,
        seed: int,
        movable_ids: torch.Tensor,
        final_norm_weight: Optional[torch.Tensor] = None,
        expects_obfuscated_input: bool = True,
        recorder: Optional[object] = None,
        record_name: Optional[str] = None,
    ) -> None:
        super().__init__(recorder=recorder, record_name=record_name)
        
        head_weight = head_module.weight.detach().cpu().to(torch.float32)
        noisy_head = add_head_noise(head_weight, alpha_h=alpha_h, seed=seed, movable_ids=movable_ids)
        
        if final_norm_weight is not None:
            # Fused version
            fused_head = (noisy_head * final_norm_weight.unsqueeze(0)) @ keymat_transform.inverse.to(torch.float32).T
        else:
            # Standard version
            fused_head = noisy_head @ keymat_transform.inverse.to(torch.float32).T
            
        self.wrapper = KeyMatHeadWrapper(
            obfuscated_weight=fused_head,
            keymat_transform=keymat_transform,
            expects_obfuscated_input=expects_obfuscated_input,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.wrapper(hidden_states)
        self.record(output)
        return output

def wrap_embedding(
    embed_module: nn.Module,
    keymat_transform: KeyMatTransform,
    alpha_e: float,
    seed: int,
    movable_ids: torch.Tensor,
    recorder: Optional[object] = None,
    record_name: Optional[str] = None,
) -> AloePriEmbedding:
    return AloePriEmbedding(
        embed_module=embed_module,
        keymat_transform=keymat_transform,
        alpha_e=alpha_e,
        seed=seed,
        movable_ids=movable_ids,
        recorder=recorder,
        record_name=record_name,
    )

def wrap_head(
    head_module: nn.Module,
    keymat_transform: KeyMatTransform,
    alpha_h: float,
    seed: int,
    movable_ids: torch.Tensor,
    final_norm_weight: Optional[torch.Tensor] = None,
    expects_obfuscated_input: bool = True,
    recorder: Optional[object] = None,
    record_name: Optional[str] = None,
) -> AloePriHead:
    return AloePriHead(
        head_module=head_module,
        keymat_transform=keymat_transform,
        alpha_h=alpha_h,
        seed=seed,
        movable_ids=movable_ids,
        final_norm_weight=final_norm_weight,
        expects_obfuscated_input=expects_obfuscated_input,
        recorder=recorder,
        record_name=record_name,
    )
