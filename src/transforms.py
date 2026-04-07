from __future__ import annotations

import torch

from src.hidden_keys import HiddenTransform


def map_input_ids(input_ids: torch.Tensor, perm_vocab: torch.Tensor) -> torch.Tensor:
    perm_vocab = perm_vocab.to(input_ids.device)
    return perm_vocab[input_ids]


def unmap_output_ids(output_ids: torch.Tensor, inv_perm_vocab: torch.Tensor) -> torch.Tensor:
    inv_perm_vocab = inv_perm_vocab.to(output_ids.device)
    return inv_perm_vocab[output_ids]


def restore_logits(logits: torch.Tensor, perm_vocab: torch.Tensor) -> torch.Tensor:
    perm_vocab = perm_vocab.to(logits.device)
    return logits.index_select(dim=-1, index=perm_vocab)


def permute_logits(logits: torch.Tensor, perm_vocab: torch.Tensor) -> torch.Tensor:
    return restore_logits(logits, perm_vocab)


def apply_hidden_transform(hidden: torch.Tensor, transform: HiddenTransform) -> torch.Tensor:
    perm = transform.perm.to(hidden.device)
    scale = transform.scale.to(device=hidden.device, dtype=hidden.dtype)
    transformed = torch.empty_like(hidden)
    transformed[..., perm] = hidden
    return transformed * scale


def apply_inverse_hidden_transform(hidden: torch.Tensor, transform: HiddenTransform) -> torch.Tensor:
    inv_scale = transform.inv_scale.to(device=hidden.device, dtype=hidden.dtype)
    scaled = hidden * inv_scale
    perm = transform.perm.to(hidden.device)
    return scaled.index_select(dim=-1, index=perm)
