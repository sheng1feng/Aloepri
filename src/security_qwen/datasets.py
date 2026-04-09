from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    description: str
    phase: str
    split_requirements: list[str]
    status: str = "planned"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_phase0_datasets() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="phase0_smoke_prompts",
            kind="smoke",
            description="Repository prompts plus a small local text sample for schema and runner smoke tests.",
            phase="phase0",
            split_requirements=["single_eval_split"],
        ),
        DatasetSpec(
            name="phase0_inversion_public_corpus",
            kind="inversion",
            description="Public text corpus reserved for NN/IMA training and validation.",
            phase="phase0",
            split_requirements=["train", "val", "test"],
        ),
        DatasetSpec(
            name="phase0_frequency_attack_corpus",
            kind="frequency",
            description="Corpus family reserved for TFMA/SDA zero-knowledge, domain-aware, and distribution-aware settings.",
            phase="phase0",
            split_requirements=["public_prior", "private_eval"],
        ),
    ]


def phase0_dataset_payload() -> dict[str, Any]:
    datasets = [item.to_dict() for item in default_phase0_datasets()]
    return {
        "format": "qwen_security_datasets_v1",
        "phase": "phase0",
        "datasets": datasets,
    }
