from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from src.key_manager import ordinary_token_ids
from src.keymat_embed_head import add_embed_noise, add_head_noise


@dataclass(frozen=True)
class SquareMonomialTransform:
    perm: torch.Tensor
    inv_perm: torch.Tensor
    signs: torch.Tensor
    global_scale: float

    @property
    def dim(self) -> int:
        return int(self.perm.numel())

    def key(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        matrix = torch.zeros((self.dim, self.dim), dtype=dtype)
        for source_idx in range(self.dim):
            target_idx = int(self.perm[source_idx].item())
            matrix[source_idx, target_idx] = float(self.global_scale) * float(self.signs[target_idx].item())
        return matrix

    def inverse(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        matrix = torch.zeros((self.dim, self.dim), dtype=dtype)
        inv_scale = 1.0 / float(self.global_scale)
        for target_idx in range(self.dim):
            source_idx = int(self.inv_perm[target_idx].item())
            matrix[target_idx, source_idx] = inv_scale * float(self.signs[target_idx].item())
        return matrix


def build_square_monomial_transform(
    hidden_size: int,
    seed: int,
    *,
    global_scale: float = 1.0,
    enforce_nontrivial: bool = True,
) -> SquareMonomialTransform:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if global_scale == 0:
        raise ValueError("global_scale must be non-zero")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(hidden_size, generator=generator, dtype=torch.long)
    if enforce_nontrivial and hidden_size > 1 and torch.equal(perm, torch.arange(hidden_size, dtype=torch.long)):
        perm = perm.clone()
        perm[0], perm[1] = perm[1], perm[0]

    sign_bits = torch.randint(0, 2, (hidden_size,), generator=generator, dtype=torch.long)
    signs = torch.where(sign_bits == 0, torch.full((hidden_size,), -1.0), torch.ones(hidden_size)).to(torch.float32)
    if enforce_nontrivial and bool(torch.all(signs == 1).item()):
        signs = signs.clone()
        signs[0] = -1.0

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(hidden_size, dtype=torch.long)
    return SquareMonomialTransform(
        perm=perm,
        inv_perm=inv_perm,
        signs=signs,
        global_scale=float(global_scale),
    )


def check_square_monomial_inverse(
    transform: SquareMonomialTransform,
    tol: float = 1e-6,
) -> dict[str, float | bool]:
    key = transform.key(dtype=torch.float64)
    inverse = transform.inverse(dtype=torch.float64)
    identity = torch.eye(transform.dim, dtype=torch.float64)
    product = key @ inverse
    return {
        "max_abs_error": float((product - identity).abs().max().item()),
        "mean_abs_error": float((product - identity).abs().mean().item()),
        "passes_tolerance": bool(torch.allclose(product, identity, atol=tol)),
    }


def obfuscate_embedding_with_square_transform(
    embed_weight: torch.Tensor,
    transform: SquareMonomialTransform,
    *,
    alpha_e: float,
    seed: int,
    movable_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    noisy = add_embed_noise(embed_weight, alpha_e=alpha_e, seed=seed, movable_ids=movable_ids)
    return noisy @ transform.key(dtype=noisy.dtype)


def obfuscate_head_with_square_transform(
    head_weight: torch.Tensor,
    transform: SquareMonomialTransform,
    *,
    alpha_h: float,
    seed: int,
    movable_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    noisy = add_head_noise(head_weight, alpha_h=alpha_h, seed=seed, movable_ids=movable_ids)
    return noisy @ transform.inverse(dtype=noisy.dtype).T


def build_stage_i_square_embed_head_model(
    baseline_model,
    tokenizer,
    perm_vocab: torch.Tensor,
    inv_perm_vocab: torch.Tensor,
    transform: SquareMonomialTransform,
    *,
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
    seed: int = 0,
):
    del inv_perm_vocab  # kept for call-site symmetry; not needed here
    model = deepcopy(baseline_model)
    movable_ids = ordinary_token_ids(tokenizer)
    embed_weight = model.get_input_embeddings().weight.detach().cpu().to(torch.float32)
    head_weight = model.get_output_embeddings().weight.detach().cpu().to(torch.float32)
    embed_obf = obfuscate_embedding_with_square_transform(
        embed_weight,
        transform,
        alpha_e=alpha_e,
        seed=seed + 101,
        movable_ids=movable_ids,
    )
    head_obf = obfuscate_head_with_square_transform(
        head_weight,
        transform,
        alpha_h=alpha_h,
        seed=seed + 202,
        movable_ids=movable_ids,
    )
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(embed_obf.to(model.get_input_embeddings().weight.dtype))
        model.get_output_embeddings().weight.copy_(head_obf.to(model.get_output_embeddings().weight.dtype))
    model.eval()
    return model
