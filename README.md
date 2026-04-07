# Privacy-inference（AloePri 复现仓库）

本仓库面向技术报告  
`Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report)`  
的工程复现与验证，围绕 **AloePri / 协变混淆推理链路**，从最小词表闭环一路推进到：

- `A–H`：主方法链复现
- `I`：标准 HF / vLLM 入口打通与阻塞定位
- `J`：standard-shape full-layer 恢复
- `K`：standard-shape 工件统一交付包装

当前默认验证模型为：

- `Qwen2.5-0.5B-Instruct`

---

## 当前仓库做到哪里了

截至当前版本，仓库已经完成：

- 词表空间闭环（Stage A）
- hidden-space / block / 多层传播恢复（Stage B–D）
- 复杂 attention 结构接入与修复（Stage E）
- Algorithm 1 / KeyMat 原生体系接入（Stage F）
- KeyMat 融合化 / 去 bridge 化（Stage G）
- attention 静态化与噪声定标（Stage H）
- 标准 HF checkpoint 导出与 Phase 2 阻塞定位（Stage I）
- non-expanding / standard-shape full-layer 功能恢复（Stage J）
- standard-shape 统一 release catalog 与 profile 推理入口（Stage K）

当前最重要的成果是：

> **standard-shape 路线已经实现 full-layer 恢复，并能导出标准 HF checkpoint；同时已给出推荐非零工作点 `tiny_a = (alpha_e=0.02, alpha_h=0.01)`，并整理成统一交付目录。**

---

## 文档入口

### 总览

- 技术总报告（阶段 A–K）：[`docs/完整复现总报告_阶段A-K.md`](docs/完整复现总报告_阶段A-K.md)
- 接手/开发总入口：[`docs/仓库技术文档.md`](docs/仓库技术文档.md)
- 模块化整理总览：[`docs/阶段A-K模块化整理报告.md`](docs/阶段A-K模块化整理报告.md)

### 阶段性文档

- 阶段 I：[`docs/阶段I_vLLM复现报告.md`](docs/阶段I_vLLM复现报告.md)
- 阶段 J：[`docs/阶段J_标准形状前缀恢复报告.md`](docs/阶段J_标准形状前缀恢复报告.md)
- 阶段 J 噪声：[`docs/阶段J_标准形状噪声定标报告.md`](docs/阶段J_标准形状噪声定标报告.md)
- 阶段 K：[`docs/阶段K_标准形状交付包装报告.md`](docs/阶段K_标准形状交付包装报告.md)
- 阶段 H 部署说明：[`docs/阶段H_混淆模型部署说明.md`](docs/阶段H_混淆模型部署说明.md)

---

## 快速开始

### 环境

- 环境文件：[`environment.qwen-transformers.yml`](environment.qwen-transformers.yml)
- 默认模型目录：`model/Qwen2.5-0.5B-Instruct`

### 一键回归

```bash
conda run --no-capture-output -n qwen-transformers pytest -q
```

### 常用入口

#### 1. 查看阶段目录（机器可读）

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_aloepri_stage_catalog.py
```

产物：

- `outputs/aloepri_stage_catalog.json`

#### 2. 跑 standard-shape full-layer 回归（Stage J）

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_j_prefix_square.py --layer-count 24
```

#### 3. 导出 standard-shape full-layer 工件

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_j_full_square_checkpoint.py
```

#### 4. 导出统一 release catalog（Stage K）

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_k_release.py
```

#### 5. 直接按 profile 推理

```bash
conda run --no-capture-output -n qwen-transformers python scripts/infer_stage_k_release.py --profile tiny_a --prompt "请用一句话介绍你自己。" --max-new-tokens 8
```

---

## 推荐工件

### 零噪声参考工件

- `artifacts/stage_j_full_square/`

特点：

- 近乎精确恢复
- 适合 regression / correctness baseline

### 推荐非零噪声工件

- `artifacts/stage_j_full_square_tiny_a/`

配置：

- `alpha_e = 0.02`
- `alpha_h = 0.01`

特点：

- 当前推荐的 non-zero standard-shape full-layer 工作点
- 导出后 HF 回归仍保持：
  - `generated_ids_exact_match_rate = 1.0`
  - `generated_text_exact_match_rate = 1.0`

### 统一交付目录

- `artifacts/stage_k_release/`

当前 profile：

- `stable_reference`
- `tiny_a`

---

## 仓库结构

- `src/`
  - 历史阶段实现与核心能力
- `src/aloepri/`
  - 模块化封装层
  - 当前重点包括：
    - `adapters/qwen.py`
    - `token_ops.py`
    - `pipelines/stage_a.py`
    - `pipelines/standard_shape.py`
    - `pipelines/release.py`
    - `catalog.py`
- `scripts/`
  - 阶段脚本、导出脚本、推理脚本
- `tests/`
  - 回归测试
- `docs/`
  - 分阶段报告与总报告
- `artifacts/`
  - 导出工件
- `outputs/`
  - JSON 回归结果

---

## 当前边界

目前仓库已经把 **HF 标准入口 + standard-shape full-layer** 这条线打通，但仍有两个明确边界：

1. 本机 `vLLM` 侧仍卡在 **CPU backend 环境**
2. 安全攻击评估（VMA / IA / ISA / IMA 等）尚未启动

因此当前项目的准确定位是：

> **方法主链复现完成，standard-shape full-layer 功能和工件交付已完成，下一步主要在于 vLLM 环境打通或安全评估。**

---

## 许可与说明

本仓库当前主要包含：

- 代码
- 文档
- 脚本
- 测试

默认 `.gitignore` 已忽略：

- `artifacts/`
- `outputs/`
- `model/`

因此远端仓库中默认不包含本地模型权重和大体积实验工件。
