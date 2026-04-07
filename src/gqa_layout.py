from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GQALayout:
    num_query_heads: int
    num_kv_heads: int

    @property
    def num_groups(self) -> int:
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads for GQA.")
        return self.num_query_heads // self.num_kv_heads

    def reshape_query_groups(self, query_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        if num_heads != self.num_query_heads:
            raise ValueError("Unexpected query head count.")
        return query_states.view(batch_size, self.num_kv_heads, self.num_groups, seq_len, head_dim)

    def merge_query_groups(self, query_grouped: torch.Tensor) -> torch.Tensor:
        batch_size, num_kv_heads, num_groups, seq_len, head_dim = query_grouped.shape
        if num_kv_heads != self.num_kv_heads or num_groups != self.num_groups:
            raise ValueError("Unexpected grouped query layout.")
        return query_grouped.reshape(batch_size, self.num_query_heads, seq_len, head_dim)

    def permute_kv_heads(self, tensor: torch.Tensor, tau_kv: torch.Tensor) -> torch.Tensor:
        return tensor.index_select(dim=1, index=tau_kv.to(tensor.device))

    def permute_query_groups(self, query_grouped: torch.Tensor, tau_kv: torch.Tensor | None, tau_group: torch.Tensor | None) -> torch.Tensor:
        result = query_grouped
        if tau_kv is not None:
            result = result.index_select(dim=1, index=tau_kv.to(result.device))
        if tau_group is not None:
            result = result.index_select(dim=2, index=tau_group.to(result.device))
        return result

    def invert_query_groups(self, query_grouped: torch.Tensor, inv_tau_kv: torch.Tensor | None, inv_tau_group: torch.Tensor | None) -> torch.Tensor:
        result = query_grouped
        if inv_tau_group is not None:
            result = result.index_select(dim=2, index=inv_tau_group.to(result.device))
        if inv_tau_kv is not None:
            result = result.index_select(dim=1, index=inv_tau_kv.to(result.device))
        return result
