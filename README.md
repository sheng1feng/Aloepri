# AloePri 复现仓库

本仓库是对技术报告  
`Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report)`  
的工程复现、部署线整理与验证。

## 活跃文档入口

- 仓库总入口：[`docs/复现主线总览.md`](docs/复现主线总览.md)
- Qwen 主线：[`docs/论文一致最终部署主线.md`](docs/论文一致最终部署主线.md)
- Llama 主线：[`docs/Llama-3.2-3B最终部署主线.md`](docs/Llama-3.2-3B最终部署主线.md)

论文与基础参考源保留在 `docs/` 根目录：

- [`docs/AloePri 论文中的部署适配机制整理.md`](docs/AloePri%20论文中的部署适配机制整理.md)
- [`docs/AloePri_技术报告梳理与复现方案.md`](docs/AloePri_%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A%E6%A2%B3%E7%90%86%E4%B8%8E%E5%A4%8D%E7%8E%B0%E6%96%B9%E6%A1%88.md)
- `docs/Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report).pdf`
- `docs/Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report).txt`

---

## Qwen 论文一致最终部署主线

Qwen 当前只保留一条活跃主线：

- [`docs/论文一致最终部署主线.md`](docs/论文一致最终部署主线.md)

当前关键工件：

- `artifacts/stage_j_qwen_paper_consistent/`
- `artifacts/stage_k_release/`

当前状态：

- `Stage J` 已达到 export-visible completion
- `Stage K` 已切到唯一 `paper_consistent` release surface
- 最终 release 面上的 correctness 与 `VMA / IMA / ISA` 仍需同口径复跑

Qwen 活跃阶段支撑文档：

- [`docs/阶段H_Qwen可部署混淆表达重构报告.md`](docs/阶段H_Qwen%E5%8F%AF%E9%83%A8%E7%BD%B2%E6%B7%B7%E6%B7%86%E8%A1%A8%E8%BE%BE%E9%87%8D%E6%9E%84%E6%8A%A5%E5%91%8A.md)
- [`docs/阶段I_部署约束验证报告.md`](docs/%E9%98%B6%E6%AE%B5I_%E9%83%A8%E7%BD%B2%E7%BA%A6%E6%9D%9F%E9%AA%8C%E8%AF%81%E6%8A%A5%E5%91%8A.md)
- [`docs/阶段J_论文一致部署路线说明.md`](docs/%E9%98%B6%E6%AE%B5J_%E8%AE%BA%E6%96%87%E4%B8%80%E8%87%B4%E9%83%A8%E7%BD%B2%E8%B7%AF%E7%BA%BF%E8%AF%B4%E6%98%8E.md)
- [`docs/阶段K_Qwen交付包装报告.md`](docs/%E9%98%B6%E6%AE%B5K_Qwen%E4%BA%A4%E4%BB%98%E5%8C%85%E8%A3%85%E6%8A%A5%E5%91%8A.md)

---

## Llama-3.2-3B 最终部署主线

Llama 当前只保留一条活跃主线：

- [`docs/Llama-3.2-3B最终部署主线.md`](docs/Llama-3.2-3B%E6%9C%80%E7%BB%88%E9%83%A8%E7%BD%B2%E4%B8%BB%E7%BA%BF.md)

当前关键工件：

- `artifacts/stage_j_llama_real_full_square/`
- `artifacts/stage_j_llama_real_full_square_tiny_a/`
- `artifacts/stage_k_llama_release/`

当前状态：

- adapter、本机 smoke、Stage I、Stage J 已成立
- 真实 `RTX 4090` correctness 验证已完成
- 噪声定标与 Llama 专属 `Stage K release` 已完成

Llama 活跃支撑文档：

- [`docs/Llama-3.2-3B标准形状恢复报告.md`](docs/Llama-3.2-3B%E6%A0%87%E5%87%86%E5%BD%A2%E7%8A%B6%E6%81%A2%E5%A4%8D%E6%8A%A5%E5%91%8A.md)
- [`docs/Llama-3.2-3B噪声定标与StageK推进说明.md`](docs/Llama-3.2-3B%E5%99%AA%E5%A3%B0%E5%AE%9A%E6%A0%87%E4%B8%8EStageK%E6%8E%A8%E8%BF%9B%E8%AF%B4%E6%98%8E.md)
- [`docs/Llama-3.2-3B客户端与Server使用说明.md`](docs/Llama-3.2-3B%E5%AE%A2%E6%88%B7%E7%AB%AF%E4%B8%8EServer%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E.md)

---

## 最常用命令

环境：

- [`environment.qwen-transformers.yml`](environment.qwen-transformers.yml)

全量测试：

```bash
conda run --no-capture-output -n qwen-transformers pytest -q
```

Qwen release 推理：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/infer_stage_k_release.py \
  --release-dir artifacts/stage_k_release \
  --profile default \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

Llama release 推理：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/infer_stage_k_release.py \
  --release-dir artifacts/stage_k_llama_release \
  --profile tiny_a \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

---

## 历史文档

非主线文档将统一迁入：

- `docs/history/qwen/`
- `docs/history/llama/`
- `docs/history/security/qwen_security/`
- `docs/history/shared/`

这些路径只保留历史证据、旧路线和旧总览，不再承担活跃主线入口职责。

---

默认 `.gitignore` 已忽略：

- `artifacts/`
- `outputs/`
- `model/`
