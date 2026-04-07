from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HiddenTransform:
    perm: torch.Tensor
    inv_perm: torch.Tensor
    scale: torch.Tensor
    inv_scale: torch.Tensor

    @property
    def dim(self) -> int:
        return int(self.perm.numel())

    def matrix(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        device = device or self.perm.device
        matrix = torch.zeros((self.dim, self.dim), device=device, dtype=dtype)
        row_index = torch.arange(self.dim, device=device)
        perm = self.perm.to(device)
        scale = self.scale.to(device=device, dtype=dtype)
        matrix[row_index, perm] = scale.index_select(0, perm)
        return matrix


def generate_hidden_permutation(hidden_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(hidden_size, generator=generator, dtype=torch.long)


def generate_hidden_scaling(
    hidden_size: int,
    scale_range: tuple[float, float],
    seed: int,
) -> torch.Tensor:
    low, high = scale_range
    if low <= 0 or high <= 0 or low > high:
        raise ValueError(f"Invalid scale range: {scale_range}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.empty(hidden_size, dtype=torch.float32).uniform_(low, high, generator=generator)


def build_hidden_transform(
    perm: torch.Tensor,
    scale: torch.Tensor,
) -> HiddenTransform:
    perm = torch.as_tensor(perm, dtype=torch.long).clone()
    scale = torch.as_tensor(scale, dtype=torch.float32).clone()
    if perm.ndim != 1 or scale.ndim != 1 or perm.numel() != scale.numel():
        raise ValueError("Permutation and scale must be 1D tensors with the same length.")
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), dtype=torch.long)
    inv_scale = 1.0 / scale
    return HiddenTransform(
        perm=perm,
        inv_perm=inv_perm,
        scale=scale,
        inv_scale=inv_scale,
    )


def invert_hidden_transform(transform: HiddenTransform) -> HiddenTransform:
    inverse_scale = transform.inv_scale.index_select(0, transform.perm)
    return build_hidden_transform(transform.inv_perm, inverse_scale)


def validate_hidden_transform(
    transform: HiddenTransform,
    inv_transform: HiddenTransform,
    atol: float = 1e-6,
) -> bool:
    matrix = transform.matrix(dtype=torch.float64)
    inv_matrix = inv_transform.matrix(dtype=torch.float64)
    identity = torch.eye(transform.dim, dtype=torch.float64)
    return bool(
        torch.allclose(matrix @ inv_matrix, identity, atol=atol)
        and torch.allclose(inv_matrix @ matrix, identity, atol=atol)
    )


def build_identity_hidden_transform(hidden_size: int) -> HiddenTransform:
    perm = torch.arange(hidden_size, dtype=torch.long)
    scale = torch.ones(hidden_size, dtype=torch.float32)
    return build_hidden_transform(perm=perm, scale=scale)
