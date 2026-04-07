from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.keymat import KeyMatTransform, apply_keymat_transform
from src.transforms import restore_logits


def _build_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _resolve_rows(weight: torch.Tensor, movable_ids: torch.Tensor | None) -> torch.Tensor:
    if movable_ids is None:
        return torch.arange(weight.shape[0], dtype=torch.long)
    return torch.as_tensor(movable_ids, dtype=torch.long)


def add_embed_noise(
    weight: torch.Tensor,
    alpha_e: float,
    seed: int,
    movable_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    noisy = weight.detach().clone().to(torch.float32)
    if alpha_e == 0:
        return noisy
    rows = _resolve_rows(noisy, movable_ids)
    sigma = float(noisy.index_select(0, rows).std(unbiased=False).item())
    generator = _build_generator(seed)
    noise = torch.randn((rows.numel(), noisy.shape[1]), generator=generator, dtype=torch.float32) * sigma
    noisy.index_copy_(0, rows, noisy.index_select(0, rows) + alpha_e * noise)
    return noisy


def add_head_noise(
    weight: torch.Tensor,
    alpha_h: float,
    seed: int,
    movable_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    noisy = weight.detach().clone().to(torch.float32)
    if alpha_h == 0:
        return noisy
    rows = _resolve_rows(noisy, movable_ids)
    sigma = float(noisy.index_select(0, rows).std(unbiased=False).item())
    generator = _build_generator(seed)
    noise = torch.randn((rows.numel(), noisy.shape[1]), generator=generator, dtype=torch.float32) * sigma
    noisy.index_copy_(0, rows, noisy.index_select(0, rows) + alpha_h * noise)
    return noisy


def obfuscate_embedding_with_keymat(
    embed_weight: torch.Tensor,
    keymat_transform: KeyMatTransform,
    alpha_e: float = 0.0,
    seed: int = 0,
    movable_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    noisy = add_embed_noise(embed_weight, alpha_e=alpha_e, seed=seed, movable_ids=movable_ids)
    key = keymat_transform.key.to(dtype=noisy.dtype)
    return noisy @ key


def obfuscate_head_with_keymat(
    head_weight: torch.Tensor,
    keymat_transform: KeyMatTransform,
    alpha_h: float = 0.0,
    seed: int = 0,
    movable_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    noisy = add_head_noise(head_weight, alpha_h=alpha_h, seed=seed, movable_ids=movable_ids)
    inverse = keymat_transform.inverse.to(dtype=noisy.dtype)
    return noisy @ inverse.T


def restore_logits_from_keymat_head(logits: torch.Tensor, perm_vocab: torch.Tensor) -> torch.Tensor:
    return restore_logits(logits, perm_vocab)


@dataclass(frozen=True)
class KeyMatEmbedHeadArtifacts:
    embed_weight_obf: torch.Tensor
    head_weight_obf: torch.Tensor


def build_keymat_embed_head_artifacts(
    stage_a_model,
    keymat_transform: KeyMatTransform,
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
    seed: int = 0,
    movable_ids: torch.Tensor | None = None,
) -> KeyMatEmbedHeadArtifacts:
    embed_weight = stage_a_model.get_input_embeddings().weight.detach().cpu().to(torch.float32)
    head_weight = stage_a_model.get_output_embeddings().weight.detach().cpu().to(torch.float32)
    return KeyMatEmbedHeadArtifacts(
        embed_weight_obf=obfuscate_embedding_with_keymat(
            embed_weight,
            keymat_transform=keymat_transform,
            alpha_e=alpha_e,
            seed=seed + 101,
            movable_ids=movable_ids,
        ),
        head_weight_obf=obfuscate_head_with_keymat(
            head_weight,
            keymat_transform=keymat_transform,
            alpha_h=alpha_h,
            seed=seed + 202,
            movable_ids=movable_ids,
        ),
    )


class KeyMatEmbeddingWrapper(nn.Module):
    def __init__(
        self,
        obfuscated_weight: torch.Tensor,
        base_weight_for_recording: torch.Tensor | None = None,
        recorder=None,
    ) -> None:
        super().__init__()
        self.recorder = recorder
        self.register_buffer("weight", obfuscated_weight.detach().to(torch.float32), persistent=False)
        if base_weight_for_recording is not None:
            self.register_buffer("base_weight_for_recording", base_weight_for_recording.detach().to(torch.float32), persistent=False)
        else:
            self.base_weight_for_recording = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.recorder is not None and self.base_weight_for_recording is not None:
            base_output = F.embedding(input_ids, self.base_weight_for_recording.to(device=input_ids.device))
            self.recorder.record("embed_out", base_output)
        output = F.embedding(input_ids, self.weight.to(device=input_ids.device))
        if self.recorder is not None:
            self.recorder.record("embed_out_obf", output)
        return output


class KeyMatHeadWrapper(nn.Module):
    def __init__(
        self,
        obfuscated_weight: torch.Tensor,
        keymat_transform: KeyMatTransform,
        expects_obfuscated_input: bool,
    ) -> None:
        super().__init__()
        self.expects_obfuscated_input = expects_obfuscated_input
        self.keymat_transform = keymat_transform
        self.register_buffer("weight", obfuscated_weight.detach().to(torch.float32), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_for_head = hidden_states
        if not self.expects_obfuscated_input:
            hidden_for_head = apply_keymat_transform(hidden_states, self.keymat_transform)
        weight = self.weight.to(device=hidden_for_head.device, dtype=hidden_for_head.dtype)
        return torch.matmul(hidden_for_head, weight.T)
