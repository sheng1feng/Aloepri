from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KeyMatBases:
    hidden_size: int
    expansion_size: int
    lam: float
    b: torch.Tensor
    b_inv: torch.Tensor
    e: torch.Tensor
    f: torch.Tensor
    z: torch.Tensor

    @property
    def expanded_size(self) -> int:
        return self.hidden_size + 2 * self.expansion_size


@dataclass(frozen=True)
class KeyMatTransform:
    hidden_size: int
    expansion_size: int
    lam: float
    key: torch.Tensor
    inverse: torch.Tensor
    bases: KeyMatBases

    @property
    def expanded_size(self) -> int:
        return int(self.key.shape[-1])


def _build_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _sample_gaussian(shape: tuple[int, ...], seed: int, scale: float = 1.0) -> torch.Tensor:
    generator = _build_generator(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float64) * scale


def _sample_orthogonal(dim: int, seed: int) -> torch.Tensor:
    gaussian = _sample_gaussian((dim, dim), seed=seed)
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    sign = torch.sign(torch.diagonal(r))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return q * sign.unsqueeze(0)


def _nullspace_basis(matrix: torch.Tensor, atol: float = 1e-10) -> torch.Tensor:
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=True)
    if singular_values.numel() == 0:
        return torch.eye(matrix.shape[1], dtype=torch.float64)
    cutoff = max(atol, atol * float(singular_values.max().item()))
    rank = int((singular_values > cutoff).sum().item())
    basis = vh[rank:].T.contiguous()
    if basis.numel() == 0:
        raise ValueError("Null space is empty; cannot construct Algorithm-1 key matrix.")
    return basis


def sample_null_columns(f_t: torch.Tensor, out_rows: int, seed: int) -> torch.Tensor:
    basis = _nullspace_basis(f_t)
    coeffs = _sample_gaussian((out_rows, basis.shape[1]), seed=seed, scale=1.0)
    return coeffs @ basis.T


def sample_null_rows(e: torch.Tensor, out_cols: int, seed: int) -> torch.Tensor:
    basis = _nullspace_basis(e)
    coeffs = _sample_gaussian((basis.shape[1], out_cols), seed=seed, scale=1.0)
    return basis @ coeffs


def init_keymat_bases(d: int, h: int, lam: float, seed: int) -> KeyMatBases:
    if h <= 0 or h % 2 != 0:
        raise ValueError(f"expansion size h must be a positive even integer, got {h}")
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")

    half_h = h // 2
    u = _sample_orthogonal(d, seed=seed + 1)
    v = _sample_gaussian((d, d), seed=seed + 2, scale=d ** -0.5)
    b = u + lam * v
    b_inv = torch.linalg.inv(b)

    e1 = _sample_gaussian((d, half_h), seed=seed + 3, scale=d ** -0.5)
    e2 = _sample_gaussian((half_h, h), seed=seed + 4, scale=d ** -0.5)
    e = e1 @ e2

    f1 = _sample_gaussian((h, half_h), seed=seed + 5, scale=d ** -0.5)
    f2 = _sample_gaussian((half_h, d), seed=seed + 6, scale=d ** -0.5)
    f = f1 @ f2

    z = _sample_orthogonal(d + 2 * h, seed=seed + 7)
    return KeyMatBases(
        hidden_size=d,
        expansion_size=h,
        lam=float(lam),
        b=b,
        b_inv=b_inv,
        e=e,
        f=f,
        z=z,
    )


def generate_keymat(bases: KeyMatBases, seed: int) -> torch.Tensor:
    c = sample_null_columns(bases.f.T, out_rows=bases.hidden_size, seed=seed + 11)
    left = torch.cat([bases.b, c, bases.e], dim=1)
    return left @ bases.z


def generate_inv_keymat(bases: KeyMatBases, seed: int) -> torch.Tensor:
    d = sample_null_rows(bases.e, out_cols=bases.hidden_size, seed=seed + 29)
    right = torch.cat([bases.b_inv, bases.f, d], dim=0)
    return bases.z.T @ right


def build_keymat_transform(
    d: int,
    h: int,
    lam: float,
    init_seed: int,
    key_seed: int | None = None,
    inv_seed: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> KeyMatTransform:
    bases = init_keymat_bases(d=d, h=h, lam=lam, seed=init_seed)
    key = generate_keymat(bases, seed=key_seed if key_seed is not None else init_seed + 1000)
    inverse = generate_inv_keymat(bases, seed=inv_seed if inv_seed is not None else init_seed + 2000)
    return KeyMatTransform(
        hidden_size=d,
        expansion_size=h,
        lam=float(lam),
        key=key.to(dtype=dtype),
        inverse=inverse.to(dtype=dtype),
        bases=bases,
    )


def check_keymat_inverse(
    key: torch.Tensor,
    inverse: torch.Tensor,
    tol: float = 1e-5,
) -> dict[str, float | bool]:
    key = torch.as_tensor(key, dtype=torch.float64)
    inverse = torch.as_tensor(inverse, dtype=torch.float64)
    identity = torch.eye(key.shape[0], dtype=torch.float64)
    product = key @ inverse
    singular_values = torch.linalg.svdvals(key)
    return {
        "max_abs_error": float((product - identity).abs().max().item()),
        "mean_abs_error": float((product - identity).abs().mean().item()),
        "spectral_norm": float(singular_values.max().item()),
        "min_singular_value": float(singular_values.min().item()),
        "condition_number": float((singular_values.max() / singular_values.min()).item()),
        "passes_tolerance": bool(torch.allclose(product, identity, atol=tol)),
    }


def apply_keymat_transform(hidden: torch.Tensor, transform: KeyMatTransform) -> torch.Tensor:
    key = transform.key.to(device=hidden.device, dtype=hidden.dtype)
    return torch.matmul(hidden, key)


def apply_inverse_keymat_transform(hidden: torch.Tensor, transform: KeyMatTransform) -> torch.Tensor:
    inverse = transform.inverse.to(device=hidden.device, dtype=hidden.dtype)
    return torch.matmul(hidden, inverse)
