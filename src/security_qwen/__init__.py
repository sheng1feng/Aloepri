from __future__ import annotations

from src.security_qwen.artifacts import (
    SecurityResolvedTarget,
    SecurityTargetSpec,
    default_security_targets,
    get_security_target,
    resolve_security_target,
    security_targets_payload,
)
from src.security_qwen.datasets import DatasetSpec, default_phase0_datasets, phase0_dataset_payload
from src.security_qwen.matrix import security_matrix_payload
from src.security_qwen.schema import (
    SecurityEvalTarget,
    build_security_eval_payload,
    validate_security_eval_payload,
)
from src.security_qwen.summary import security_summary_payload
from src.security_qwen.vma import (
    build_vma_layer_ablation_payload,
    build_vma_source_attribution_payload,
    build_vma_comparison_payload,
    build_vma_template,
    default_vma_gate1_targets,
    infer_vma_default_projection_layers,
    run_vma_baseline,
)
from src.security_qwen.tfma import (
    FrequencyAttackConfig,
    build_frequency_corpora,
    build_tfma_comparison_payload,
    build_tfma_template,
    default_frequency_gate4_targets,
    run_tfma_baseline,
)
from src.security_qwen.sda import (
    SDABaselineConfig,
    build_sda_comparison_payload,
    build_sda_template,
    run_sda_baseline,
)
from src.security_qwen.gate6_enhancement import (
    GATE6_SENSITIVE_TERMS,
    Gate6Case,
    default_gate6_cases,
    ensure_gate6_artifact,
    security_sensitive_plain_ids,
)
from src.security_qwen.ia import build_ia_template
from src.security_qwen.ima import (
    IMABaselineConfig,
    build_ima_comparison_payload,
    build_ima_template,
    default_ima_gate2_targets,
    load_ima_embedding_sources,
    run_ima_baseline,
)
from src.security_qwen.isa import (
    ISABaselineConfig,
    build_isa_comparison_payload,
    build_isa_template,
    default_isa_gate3_targets,
    load_isa_model_bundle,
    run_isa_baseline,
)
from src.security_qwen.tfma import build_tfma_template
from src.security_qwen.sda import build_sda_template

__all__ = [
    "SecurityResolvedTarget",
    "SecurityTargetSpec",
    "default_security_targets",
    "get_security_target",
    "resolve_security_target",
    "security_targets_payload",
    "DatasetSpec",
    "default_phase0_datasets",
    "phase0_dataset_payload",
    "security_matrix_payload",
    "security_summary_payload",
    "SecurityEvalTarget",
    "build_security_eval_payload",
    "validate_security_eval_payload",
    "build_vma_template",
    "build_vma_layer_ablation_payload",
    "build_vma_source_attribution_payload",
    "build_vma_comparison_payload",
    "default_vma_gate1_targets",
    "infer_vma_default_projection_layers",
    "run_vma_baseline",
    "build_ia_template",
    "IMABaselineConfig",
    "build_ima_template",
    "build_ima_comparison_payload",
    "default_ima_gate2_targets",
    "load_ima_embedding_sources",
    "run_ima_baseline",
    "ISABaselineConfig",
    "build_isa_comparison_payload",
    "build_isa_template",
    "default_isa_gate3_targets",
    "load_isa_model_bundle",
    "run_isa_baseline",
    "FrequencyAttackConfig",
    "build_frequency_corpora",
    "build_tfma_comparison_payload",
    "build_tfma_template",
    "default_frequency_gate4_targets",
    "run_tfma_baseline",
    "SDABaselineConfig",
    "build_sda_comparison_payload",
    "build_sda_template",
    "run_sda_baseline",
    "GATE6_SENSITIVE_TERMS",
    "Gate6Case",
    "default_gate6_cases",
    "ensure_gate6_artifact",
    "security_sensitive_plain_ids",
    "build_tfma_template",
    "build_sda_template",
]
