from __future__ import annotations

from dataclasses import dataclass

import torch

from src.keymat import KeyMatBases, KeyMatTransform


def build_diag_friendly_keymat_transform(
    *,
    hidden_size: int,
    expansion_size: int,
    seed: int,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
    dtype: torch.dtype = torch.float32,
) -> KeyMatTransform:
    if expansion_size <= 0 or expansion_size % 2 != 0:
        raise ValueError(f"expansion_size must be a positive even integer, got {expansion_size}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    scales = torch.empty(hidden_size, dtype=torch.float64).uniform_(scale_min, scale_max, generator=generator)
    b = torch.diag(scales)
    b_inv = torch.diag(1.0 / scales)
    e = torch.zeros((hidden_size, expansion_size), dtype=torch.float64)
    f = torch.zeros((expansion_size, hidden_size), dtype=torch.float64)
    z = torch.eye(hidden_size + 2 * expansion_size, dtype=torch.float64)

    bases = KeyMatBases(
        hidden_size=hidden_size,
        expansion_size=expansion_size,
        lam=0.0,
        b=b,
        b_inv=b_inv,
        e=e,
        f=f,
        z=z,
    )

    zero_c = torch.zeros((hidden_size, expansion_size), dtype=torch.float64)
    key = torch.cat([b, zero_c, e], dim=1)
    zero_d = torch.zeros((expansion_size, hidden_size), dtype=torch.float64)
    inverse = torch.cat([b_inv, f, zero_d], dim=0)

    return KeyMatTransform(
        hidden_size=hidden_size,
        expansion_size=expansion_size,
        lam=0.0,
        key=key.to(dtype=dtype),
        inverse=inverse.to(dtype=dtype),
        bases=bases,
    )
