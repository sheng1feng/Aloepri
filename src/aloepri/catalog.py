from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StageCatalogEntry:
    stage: str
    title: str
    status: str
    objective: str
    legacy_scripts: list[str]
    core_modules: list[str]
    modular_entrypoints: list[str]
    artifacts: list[str]
    docs: list[str]


def default_stage_catalog() -> list[StageCatalogEntry]:
    return [
        StageCatalogEntry(
            stage="A",
            title="词表空间闭环",
            status="completed",
            objective="完成 token permutation + embedding/head 置换，并验证严格闭环。",
            legacy_scripts=["scripts/run_baseline.py", "scripts/run_permuted_eval.py"],
            core_modules=["src/key_manager.py", "src/transforms.py", "src/obfuscate_embed_head.py"],
            modular_entrypoints=["src/aloepri/token_ops.py", "src/aloepri/pipelines/stage_a.py"],
            artifacts=["artifacts/stage_i_vllm"],
            docs=["docs/阶段A_B严格复现报告.md"],
        ),
        StageCatalogEntry(
            stage="B",
            title="hidden-space 入口与 block0 attention",
            status="completed",
            objective="引入 hidden transform 并验证 block0 attention 子层恢复。",
            legacy_scripts=["scripts/run_stage_b_hidden_only.py", "scripts/run_stage_b_block0_attn_wrapper.py", "scripts/run_stage_b_block0_attn_fused.py"],
            core_modules=["src/hidden_keys.py", "src/stage_b.py"],
            modular_entrypoints=["src/aloepri/transforms (planned)"],
            artifacts=[],
            docs=["docs/阶段A_B严格复现报告.md"],
        ),
        StageCatalogEntry(
            stage="C",
            title="block0 完整恢复",
            status="completed",
            objective="完成 block0 norm + attention + ffn 协变恢复。",
            legacy_scripts=["scripts/run_stage_c_block0_full.py"],
            core_modules=["src/obfuscate_rmsnorm.py", "src/obfuscate_ffn.py", "src/stage_c.py"],
            modular_entrypoints=["src/aloepri/layers/norm.py", "src/aloepri/layers/ffn.py"],
            artifacts=[],
            docs=["docs/阶段C_block0完整恢复报告.md"],
        ),
        StageCatalogEntry(
            stage="D",
            title="多层 block 传播",
            status="completed",
            objective="验证 block 级恢复能否复制到多层并观察系统级回归。",
            legacy_scripts=["scripts/run_stage_d_layers.py", "scripts/run_stage_d_layers_2.py", "scripts/run_stage_d_layers_4.py", "scripts/run_stage_d_layers_8.py", "scripts/run_stage_d_layers_full.py"],
            core_modules=["src/stage_d.py"],
            modular_entrypoints=["src/aloepri/engine.py (legacy reference only)"],
            artifacts=[],
            docs=["docs/完整复现总报告_阶段A-K.md"],
        ),
        StageCatalogEntry(
            stage="E",
            title="复杂 attention 结构",
            status="completed",
            objective="接入并修复 R̂_qk/Ĥ_qk/Ẑ_block/τ_kv/τ_group。",
            legacy_scripts=["scripts/run_stage_e_block0_attention_complex.py", "scripts/run_stage_e_prefix_layers.py", "scripts/run_stage_e_ablation.py", "scripts/run_stage_e_head_trace_check.py"],
            core_modules=["src/attention_keys.py", "src/gqa_layout.py", "src/obfuscate_attention_complex.py", "src/stage_e.py"],
            modular_entrypoints=["src/aloepri/layers/attention.py"],
            artifacts=[],
            docs=["docs/阶段E复杂Attention复现报告.md", "docs/阶段E排错与修正报告.md"],
        ),
        StageCatalogEntry(
            stage="F",
            title="KeyMat 原生体系接入",
            status="completed",
            objective="用 Algorithm 1 / KeyMat 完成首次系统功能接入。",
            legacy_scripts=["scripts/run_stage_f_keymat_unit.py", "scripts/run_stage_f_embed_head.py", "scripts/run_stage_f_block0.py", "scripts/run_stage_f_prefix_layers.py", "scripts/run_stage_f_full_layers.py"],
            core_modules=["src/keymat.py", "src/keymat_embed_head.py", "src/keymat_norm.py", "src/keymat_ffn.py", "src/keymat_attention_bridge.py", "src/stage_f.py"],
            modular_entrypoints=["src/aloepri/keys.py", "src/aloepri/layers/embeddings.py", "src/aloepri/layers/norm.py", "src/aloepri/layers/ffn.py", "src/aloepri/layers/attention.py"],
            artifacts=[],
            docs=["docs/阶段F-KeyMat复现计划与结果.md"],
        ),
        StageCatalogEntry(
            stage="G",
            title="KeyMat 融合化 / 去 bridge 化",
            status="completed",
            objective="将内部层 runtime Q/P bridge 融入参数表达。",
            legacy_scripts=["scripts/run_stage_g_regression.py", "scripts/export_stage_g_model.py", "scripts/infer_stage_g_model.py"],
            core_modules=["src/stage_g_norm.py", "src/stage_g_ffn.py", "src/stage_g_attention.py", "src/stage_g.py", "src/stage_g_artifact.py"],
            modular_entrypoints=["src/aloepri/layers/norm.py", "src/aloepri/layers/ffn.py", "src/aloepri/layers/attention.py"],
            artifacts=["artifacts/stage_g_full_obfuscated"],
            docs=["docs/阶段G-KeyMat融合化报告.md"],
        ),
        StageCatalogEntry(
            stage="H",
            title="attention 静态化与噪声定标",
            status="completed",
            objective="进一步静态化 attention，并确定 KeyMat 路线下的噪声工作点。",
            legacy_scripts=["scripts/run_stage_h_noise_calibration.py", "scripts/run_stage_h_attention_static.py", "scripts/run_stage_h_joint_regression.py", "scripts/export_stage_h_model.py", "scripts/export_stage_h_pretrained.py"],
            core_modules=["src/stage_h_attention_static.py", "src/stage_h_noise.py", "src/stage_h.py", "src/stage_h_artifact.py", "src/stage_h_pretrained.py"],
            modular_entrypoints=["src/aloepri/layers/attention.py", "src/aloepri/pipelines/release.py (future packaging)"],
            artifacts=["artifacts/stage_h_full_obfuscated", "artifacts/stage_h_pretrained"],
            docs=["docs/阶段H-Attention静态化与噪声定标报告.md", "docs/阶段H_混淆模型部署说明.md"],
        ),
        StageCatalogEntry(
            stage="I",
            title="标准 HF/vLLM 入口与阻塞定位",
            status="completed_primary_path",
            objective="打通标准 HF/vLLM Phase-1 入口，并定位继续 materialize 的真实阻塞点。",
            legacy_scripts=["scripts/export_stage_i_vllm_checkpoint.py", "scripts/run_stage_i_hf_regression.py", "scripts/run_stage_i_vllm_regression.py", "scripts/run_stage_i_phase2_probe.py"],
            core_modules=["src/stage_i_vllm.py", "src/stage_i_square.py"],
            modular_entrypoints=["src/aloepri/pipelines/stage_a.py", "src/aloepri/token_ops.py"],
            artifacts=["artifacts/stage_i_vllm", "artifacts/stage_i_phase2_square"],
            docs=["docs/阶段I_vLLM复现报告.md", "docs/阶段I_Phase2_非扩维可逆变换设计.md", "docs/阶段I_Phase2_最小原型报告.md"],
        ),
        StageCatalogEntry(
            stage="J",
            title="standard-shape full-layer 恢复",
            status="completed",
            objective="在不改变标准 checkpoint shape 的前提下恢复 full-layer 功能，并给出噪声工作点。",
            legacy_scripts=["scripts/run_stage_j_block0_square.py", "scripts/run_stage_j_prefix_square.py", "scripts/run_stage_j_noise_calibration.py", "scripts/export_stage_j_full_square_checkpoint.py"],
            core_modules=["src/stage_j_block0.py", "src/stage_j_noise.py"],
            modular_entrypoints=["src/aloepri/pipelines/standard_shape.py"],
            artifacts=["artifacts/stage_j_full_square", "artifacts/stage_j_full_square_tiny_a"],
            docs=["docs/阶段J_标准形状协变恢复计划.md", "docs/阶段J_标准形状前缀恢复报告.md", "docs/阶段J_标准形状噪声定标报告.md"],
        ),
        StageCatalogEntry(
            stage="K",
            title="standard-shape 工件统一交付包装",
            status="completed_first_round",
            objective="统一 standard-shape 工件的 profile、catalog、contract 和推理入口。",
            legacy_scripts=["scripts/export_stage_k_release.py", "scripts/infer_stage_k_release.py"],
            core_modules=["src/stage_k_release.py"],
            modular_entrypoints=["src/aloepri/pipelines/release.py"],
            artifacts=["artifacts/stage_k_release"],
            docs=["docs/阶段K_标准形状交付包装报告.md"],
        ),
    ]


def stage_catalog_payload() -> dict[str, Any]:
    entries = default_stage_catalog()
    return {
        "format": "aloepri_stage_catalog_v1",
        "stages": [asdict(entry) for entry in entries],
    }
