from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IntraHeadKeys:
    q_matrix: torch.Tensor
    q_inverse: torch.Tensor
    k_matrix: torch.Tensor
    k_inverse: torch.Tensor
    r_qk: torch.Tensor
    h_qk: torch.Tensor
    h_qk_inverse: torch.Tensor
    z_block: torch.Tensor
    z_block_inverse: torch.Tensor
    block_perm: torch.Tensor
    profile: str


@dataclass(frozen=True)
class InterHeadPermutations:
    tau_kv: torch.Tensor | None
    inv_tau_kv: torch.Tensor | None
    tau_group: torch.Tensor | None
    inv_tau_group: torch.Tensor | None


@dataclass(frozen=True)
class AttentionComplexConfig:
    profile: str
    intra_head: IntraHeadKeys
    inter_head: InterHeadPermutations
    beta: int
    gamma: float
    rope_base: float


def _eye(size: int) -> torch.Tensor:
    return torch.eye(size, dtype=torch.float32)


def generate_r_qk(head_dim: int, seed: int) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE-compatible 2x2 blocks.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_blocks = head_dim // 2
    angles = torch.empty(num_blocks, dtype=torch.float32).uniform_(0.0, 2.0 * torch.pi, generator=generator)
    matrix = torch.zeros((head_dim, head_dim), dtype=torch.float32)
    half = head_dim // 2
    for block_idx, angle in enumerate(angles.tolist()):
        cos = torch.cos(torch.tensor(angle))
        sin = torch.sin(torch.tensor(angle))
        block = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float32)
        i = block_idx
        j = block_idx + half
        matrix[i, i] = block[0, 0]
        matrix[i, j] = block[0, 1]
        matrix[j, i] = block[1, 0]
        matrix[j, j] = block[1, 1]
    return matrix


def generate_h_qk(
    head_dim: int,
    scale_range: tuple[float, float],
    seed: int,
) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for paired scaling.")
    low, high = scale_range
    if low <= 0 or high <= 0 or low > high:
        raise ValueError(f"Invalid Q/K scale range: {scale_range}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    num_blocks = head_dim // 2
    block_scales = torch.empty(num_blocks, dtype=torch.float32).uniform_(low, high, generator=generator)
    diag = torch.cat([block_scales, block_scales], dim=0)
    return torch.diag(diag)


def generate_block_perm(
    num_blocks: int,
    beta: int,
    gamma: float,
    rope_base: float,
    seed: int,
    mode: str = "dynamic_window",
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode not in {"simplified_window", "dynamic_window"}:
        raise ValueError(f"Unsupported block permutation mode: {mode}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm_blocks = torch.arange(num_blocks, dtype=torch.long)
    beta = max(1, min(beta, num_blocks))
    if mode == "simplified_window":
        for start in range(0, num_blocks, beta):
            end = min(start + beta, num_blocks)
            window = perm_blocks[start:end]
            perm_blocks[start:end] = window[torch.randperm(window.numel(), generator=generator)]
    else:
        zeta_log = torch.tensor(
            [(-2.0 * idx / max(1, num_blocks)) * torch.log(torch.tensor(float(rope_base))) for idx in range(num_blocks)],
            dtype=torch.float32,
        )
        blocks = []
        start = 0
        while start < num_blocks:
            c = min(beta, num_blocks - start)
            if c == 1:
                blocks.append(torch.tensor([start], dtype=torch.long))
                start += 1
                continue
            local_scores = gamma * (zeta_log[start : start + c] - zeta_log[start])
            probs = torch.softmax(local_scores, dim=0)
            window_size = int(torch.multinomial(probs, num_samples=1, generator=generator).item()) + 1
            window = torch.arange(start, start + window_size, dtype=torch.long)
            permuted = window[torch.randperm(window.numel(), generator=generator)]
            blocks.append(permuted)
            start += window_size
        perm_blocks = torch.cat(blocks, dim=0)

    head_dim = num_blocks * 2
    half = num_blocks
    block_matrix = torch.zeros((head_dim, head_dim), dtype=torch.float32)
    for original_block_idx in range(num_blocks):
        target_block_idx = int(perm_blocks[original_block_idx].item())
        block_matrix[target_block_idx, original_block_idx] = 1.0
        block_matrix[target_block_idx + half, original_block_idx + half] = 1.0
    return perm_blocks, block_matrix


def generate_tau_kv(num_kv_heads: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tau = torch.randperm(num_kv_heads, generator=generator, dtype=torch.long)
    if num_kv_heads > 1:
        identity = torch.arange(num_kv_heads, dtype=torch.long)
        attempts = 0
        while torch.equal(tau, identity) and attempts < 8:
            tau = torch.randperm(num_kv_heads, generator=generator, dtype=torch.long)
            attempts += 1
        if torch.equal(tau, identity):
            tau = torch.roll(identity, shifts=1)
    inv = torch.empty_like(tau)
    inv[tau] = torch.arange(num_kv_heads, dtype=torch.long)
    return tau, inv


def generate_tau_group(num_groups: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tau = torch.randperm(num_groups, generator=generator, dtype=torch.long)
    if num_groups > 1:
        identity = torch.arange(num_groups, dtype=torch.long)
        attempts = 0
        while torch.equal(tau, identity) and attempts < 8:
            tau = torch.randperm(num_groups, generator=generator, dtype=torch.long)
            attempts += 1
        if torch.equal(tau, identity):
            tau = torch.roll(identity, shifts=1)
    inv = torch.empty_like(tau)
    inv[tau] = torch.arange(num_groups, dtype=torch.long)
    return tau, inv


def build_attention_complex_config(
    *,
    profile: str,
    head_dim: int,
    num_kv_heads: int,
    num_groups: int,
    seed: int,
    qk_scale_range: tuple[float, float] = (0.95, 1.05),
    beta: int = 4,
    gamma: float = 1e3,
    rope_base: float = 10000.0,
) -> AttentionComplexConfig:
    supported_profiles = {
        "simplified",
        "rqk",
        "rqk_hqk",
        "rqk_hqk_block",
        "rqk_hqk_block_taukv",
        "rqk_hqk_block_taukv_taugroup",
    }
    if profile not in supported_profiles:
        raise ValueError(f"Unsupported attention profile: {profile}")

    enable_r = profile in {"rqk", "rqk_hqk", "rqk_hqk_block", "rqk_hqk_block_taukv", "rqk_hqk_block_taukv_taugroup"}
    enable_h = profile in {"rqk_hqk", "rqk_hqk_block", "rqk_hqk_block_taukv", "rqk_hqk_block_taukv_taugroup"}
    enable_block = profile in {"rqk_hqk_block", "rqk_hqk_block_taukv", "rqk_hqk_block_taukv_taugroup"}
    enable_tau_kv = profile in {"rqk_hqk_block_taukv", "rqk_hqk_block_taukv_taugroup"}
    enable_tau_group = profile in {"rqk_hqk_block_taukv_taugroup"}

    r_qk = generate_r_qk(head_dim, seed=seed + 1) if enable_r else _eye(head_dim)
    h_qk = generate_h_qk(head_dim, scale_range=qk_scale_range, seed=seed + 2) if enable_h else _eye(head_dim)
    h_qk_inverse = torch.inverse(h_qk)
    num_blocks = head_dim // 2
    block_perm, z_block = generate_block_perm(
        num_blocks=num_blocks,
        beta=beta,
        gamma=gamma,
        rope_base=rope_base,
        seed=seed + 3,
    ) if enable_block else (torch.arange(num_blocks, dtype=torch.long), _eye(head_dim))
    z_block_inverse = z_block.T

    q_matrix = r_qk @ h_qk @ z_block
    q_inverse = z_block_inverse @ h_qk_inverse @ r_qk.T
    k_matrix = r_qk @ h_qk_inverse @ z_block_inverse
    k_inverse = z_block @ h_qk @ r_qk.T

    tau_kv, inv_tau_kv = generate_tau_kv(num_kv_heads, seed=seed + 4) if enable_tau_kv else (None, None)
    tau_group, inv_tau_group = generate_tau_group(num_groups, seed=seed + 5) if enable_tau_group else (None, None)

    return AttentionComplexConfig(
        profile=profile,
        intra_head=IntraHeadKeys(
            q_matrix=q_matrix,
            q_inverse=q_inverse,
            k_matrix=k_matrix,
            k_inverse=k_inverse,
            r_qk=r_qk,
            h_qk=h_qk,
            h_qk_inverse=h_qk_inverse,
            z_block=z_block,
            z_block_inverse=z_block_inverse,
            block_perm=block_perm,
            profile=profile,
        ),
        inter_head=InterHeadPermutations(
            tau_kv=tau_kv,
            inv_tau_kv=inv_tau_kv,
            tau_group=tau_group,
            inv_tau_group=inv_tau_group,
        ),
        beta=beta,
        gamma=gamma,
        rope_base=rope_base,
    )
