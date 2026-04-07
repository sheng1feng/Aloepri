from __future__ import annotations

from copy import deepcopy

import torch


def permute_embedding_weight(
    embed_weight: torch.Tensor,
    perm_vocab: torch.Tensor,
) -> torch.Tensor:
    perm_vocab = perm_vocab.to(embed_weight.device)
    permuted = embed_weight.clone()
    permuted[perm_vocab] = embed_weight
    return permuted


def permute_lm_head_weight(
    head_weight: torch.Tensor,
    perm_vocab: torch.Tensor,
) -> torch.Tensor:
    perm_vocab = perm_vocab.to(head_weight.device)
    permuted = head_weight.clone()
    permuted[perm_vocab] = head_weight
    return permuted


def permute_output_bias(
    bias: torch.Tensor,
    perm_vocab: torch.Tensor,
) -> torch.Tensor:
    perm_vocab = perm_vocab.to(bias.device)
    permuted = bias.clone()
    permuted[perm_vocab] = bias
    return permuted


def build_vocab_permuted_model(
    model,
    perm_vocab: torch.Tensor,
    inv_perm_vocab: torch.Tensor | None = None,
):
    del inv_perm_vocab
    permuted_model = deepcopy(model)
    input_embeddings = permuted_model.get_input_embeddings()
    output_embeddings = permuted_model.get_output_embeddings()

    permuted_embed_weight = permute_embedding_weight(input_embeddings.weight.detach(), perm_vocab)
    permuted_head_weight = permute_lm_head_weight(output_embeddings.weight.detach(), perm_vocab)

    with torch.no_grad():
        input_embeddings.weight.copy_(permuted_embed_weight)
        output_embeddings.weight.copy_(permuted_head_weight)
        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.copy_(permute_output_bias(output_embeddings.bias.detach(), perm_vocab))

    return permuted_model

