from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def ordinary_token_ids(tokenizer: Any) -> torch.Tensor:
    fixed_special = {token_id for token_id in tokenizer.all_special_ids if token_id < tokenizer.vocab_size}
    movable = [token_id for token_id in range(tokenizer.vocab_size) if token_id not in fixed_special]
    return torch.tensor(movable, dtype=torch.long)


def generate_vocab_permutation(
    vocab_size: int,
    seed: int,
    movable_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    perm = torch.arange(vocab_size, dtype=torch.long)
    if movable_ids is None:
        movable_ids = torch.arange(vocab_size, dtype=torch.long)
    movable_ids = torch.as_tensor(movable_ids, dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shuffled = movable_ids[torch.randperm(movable_ids.numel(), generator=generator)]
    perm[movable_ids] = shuffled
    return perm


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    perm = torch.as_tensor(perm, dtype=torch.long)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), dtype=torch.long)
    return inv_perm


def validate_permutation(perm: torch.Tensor) -> bool:
    perm = torch.as_tensor(perm, dtype=torch.long)
    if perm.ndim != 1:
        return False
    expected = torch.arange(perm.numel(), dtype=torch.long)
    actual = torch.sort(perm).values
    return bool(torch.equal(actual, expected))


def save_permutation(
    perm: torch.Tensor,
    inv_perm: torch.Tensor,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "perm_vocab": torch.as_tensor(perm, dtype=torch.long),
        "inv_perm_vocab": torch.as_tensor(inv_perm, dtype=torch.long),
        "metadata": metadata or {},
    }
    torch.save(payload, Path(path))


def load_permutation(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu")
    perm = torch.as_tensor(payload["perm_vocab"], dtype=torch.long)
    inv_perm = torch.as_tensor(payload["inv_perm_vocab"], dtype=torch.long)
    metadata = dict(payload.get("metadata", {}))
    return perm, inv_perm, metadata

