from __future__ import annotations

from torch import nn

from src.hidden_keys import build_identity_hidden_transform
from src.keymat import KeyMatTransform, apply_inverse_keymat_transform, apply_keymat_transform
from src.obfuscate_ffn import FFNTransform, obfuscate_ffn_block


class KeyMatFFNBridge(nn.Module):
    def __init__(
        self,
        mlp_module: nn.Module,
        keymat_transform: KeyMatTransform,
        ffn_transform: FFNTransform,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.keymat_transform = keymat_transform
        self.inner = obfuscate_ffn_block(
            mlp_module=mlp_module,
            hidden_transform=build_identity_hidden_transform(keymat_transform.hidden_size),
            ffn_transform=ffn_transform,
            recorder=recorder,
            record_name=record_name,
        )

    def forward(self, hidden_states):
        base_hidden = apply_inverse_keymat_transform(hidden_states, self.keymat_transform)
        mlp_base = self.inner(base_hidden)
        return apply_keymat_transform(mlp_base, self.keymat_transform)


def build_keymat_ffn_wrapper(
    mlp_module: nn.Module,
    keymat_transform: KeyMatTransform,
    ffn_transform: FFNTransform,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatFFNBridge:
    return KeyMatFFNBridge(
        mlp_module=mlp_module,
        keymat_transform=keymat_transform,
        ffn_transform=ffn_transform,
        recorder=recorder,
        record_name=record_name,
    )


def obfuscate_ffn_with_keymat(
    mlp_module: nn.Module,
    keymat_transform: KeyMatTransform,
    ffn_transform: FFNTransform,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatFFNBridge:
    return build_keymat_ffn_wrapper(
        mlp_module=mlp_module,
        keymat_transform=keymat_transform,
        ffn_transform=ffn_transform,
        recorder=recorder,
        record_name=record_name,
    )
