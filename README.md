# AloePri 复现仓库

本仓库是对技术报告  
`Towards Privacy-Preserving LLM Inference via Collaborative Obfuscation (Technical Report)`  
的工程复现与验证。

仓库当前已经从“研究原型”推进到：

- **Qwen：唯一主线为论文一致最终部署线（见主入口文档，final Stage J/K 仍在收口）**
- **Llama-3.2-3B：已完成结构接入、真实 4090 correctness、噪声定标与 Stage K release**

如果你第一次打开这个仓库，建议先看 Qwen 主入口文档中的当前状态与剩余工作定义：

> **[`docs/论文一致最终部署主线.md`](docs/论文一致最终部署主线.md)**

## Qwen 论文一致最终部署主线

当前 Qwen 文档只保留一条主线：

- [`docs/论文一致最终部署主线.md`](docs/论文一致最终部署主线.md)

legacy conservative `Stage H/I/J/K`、buffered redesign、standard-visible bridge 都不再作为并列进度线出现，而只作为证据或历史参考保留。

---

## 1. 仓库当前完成了什么

### Qwen 主线

Qwen 当前只认一条主线：论文一致最终部署线。当前状态以主入口文档为准，已明确 final `Stage J/K` 仍在收口阶段：

- 主入口：[`docs/论文一致最终部署主线.md`](docs/论文一致最终部署主线.md)
- 历史 `standard-shape`/旧 `Stage K` 工件仅作为历史证据，不再作为默认交付路径

### Llama-3.2-3B

Llama 不是从 A 开始重新做一遍，而是在 Qwen 主线成立后重点推进：

- `LlamaArchitectureAdapter`
- Stage I 标准 HF 工件
- Stage J standard-shape full-layer
- 真实 `RTX 4090` 上的 correctness 验证
- 真实噪声定标
- 独立 Stage K release

当前 Llama 已完成的关键结果：

- **Stage I：严格通过**
- **Stage J：full-layer correctness 通过**
- **tiny_a：真实非零噪声工作点通过**
- **Stage K：release 目录已产出**

---

## 2. Stage J 和 Stage K 的区别

这是当前仓库最容易混淆、也最重要的概念。

### Stage J：最终部署物收口

Stage J 的目标是：

> **把论文一致的最终部署物做实并闭环必要等价性，不把中间桥接工件当作最终答案。**

也就是说，Stage J 输出的是：

- **单个标准 HF 工件**
- 它在目录形式上像普通 Hugging Face 模型
- 但参数已经被混淆

典型工件（例如）：

- Llama：
  - `artifacts/stage_j_llama_real_full_square/`
  - `artifacts/stage_j_llama_real_full_square_tiny_a/`

### Stage K：最终线交付包装

Stage K 的目标是：

> **把最终确认后的 Stage J 产物包装成唯一 release 面。**

Stage K 额外提供：

- `catalog.json`
- `deployment_contract.json`
- `profiles/`
- 统一推理入口
- README / profile 选择逻辑

所以：

- **Stage J = 单个标准 HF 混淆模型**
- **Stage K = 若干个 Stage J 工件的统一发布包**

一句话记忆：

> **J 是“模型本体正确”，K 是“交付形态清晰”。**

---

## 3. “标准 HF 混淆模型”是什么意思

这里的“标准 HF”指的是：

- server 侧工件仍然是标准 Hugging Face 目录
- 可以直接：
  - `AutoTokenizer.from_pretrained(...)`
  - `AutoModelForCausalLM.from_pretrained(...)`

典型目录会包含：

- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors`

但是它和官方下载的 HF 模型有一个关键区别：

> **它不是明文模型，而是 obfuscated checkpoint。**

也就是说：

- **server** 持有：
  - 混淆后的标准 HF 工件
- **client** 持有：
  - `client_secret.pt`
  - 负责输入映射和输出恢复

因此使用时：

- 明文 prompt 不直接发给 server
- client 先把输入 token 映射成混淆 token ids
- server 正常用 HF 模型推理
- client 再恢复输出 token / logits

---

## 4. 文档入口

### 总览

- 总报告（阶段 A–K）：[`docs/完整复现总报告_阶段A-K.md`](docs/完整复现总报告_阶段A-K.md)
- 仓库接手指南：[`docs/仓库技术文档.md`](docs/仓库技术文档.md)
- 模块化整理总览：[`docs/阶段A-K模块化整理报告.md`](docs/阶段A-K模块化整理报告.md)

### Qwen / Llama 阶段区分

- 阶段区分说明：[`docs/Qwen与Llama复现阶段区分说明.md`](docs/Qwen与Llama复现阶段区分说明.md)

### Qwen 安全评测

- 安全评测索引：[`docs/qwen_security/README.md`](docs/qwen_security/README.md)
- 安全总报告：[`docs/qwen_security/Qwen安全总报告.md`](docs/qwen_security/Qwen安全总报告.md)
- 部署线弱于 Stage H 的原因分析：[`docs/qwen_security/部署线弱于StageH的原因分析.md`](docs/qwen_security/部署线弱于StageH的原因分析.md)
- 论文部署适配机制整理：[`docs/AloePri 论文中的部署适配机制整理.md`](docs/AloePri 论文中的部署适配机制整理.md)

### Llama 文档

- 快速使用说明：[`docs/Llama-3.2-3B快速使用说明.md`](docs/Llama-3.2-3B快速使用说明.md)
- client/server 使用说明：[`docs/Llama-3.2-3B客户端与Server使用说明.md`](docs/Llama-3.2-3B客户端与Server使用说明.md)
- 标准形状恢复报告：[`docs/Llama-3.2-3B标准形状恢复报告.md`](docs/Llama-3.2-3B标准形状恢复报告.md)
- 云端验证说明：[`docs/Llama-3.2-3B云端验证说明.md`](docs/Llama-3.2-3B云端验证说明.md)
- 噪声定标与 Stage K 推进说明：[`docs/Llama-3.2-3B噪声定标与StageK推进说明.md`](docs/Llama-3.2-3B噪声定标与StageK推进说明.md)

### Qwen 主线

- 主线总入口：[`docs/论文一致最终部署主线.md`](docs/论文一致最终部署主线.md)
- Stage H：[`docs/阶段H_Qwen可部署混淆表达重构报告.md`](docs/阶段H_Qwen可部署混淆表达重构报告.md)
- Stage I：[`docs/阶段I_部署约束验证报告.md`](docs/阶段I_部署约束验证报告.md)
- Stage J：[`docs/阶段J_论文一致部署路线说明.md`](docs/阶段J_论文一致部署路线说明.md)
- Stage K：[`docs/阶段K_Qwen交付包装报告.md`](docs/阶段K_Qwen交付包装报告.md)
- 证据：[`docs/阶段H-K论文对齐检查表.md`](docs/阶段H-K论文对齐检查表.md)
- 证据：[`docs/qwen_security/Qwen安全总报告.md`](docs/qwen_security/Qwen安全总报告.md)
- 证据：[`docs/AloePri 论文中的部署适配机制整理.md`](docs/AloePri%20论文中的部署适配机制整理.md)

---

## 5. 最常用入口

### 环境

- 环境文件：[`environment.qwen-transformers.yml`](environment.qwen-transformers.yml)

### 一键回归

```bash
conda run --no-capture-output -n qwen-transformers pytest -q
```

### 查看阶段目录（机器可读）

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_aloepri_stage_catalog.py
```

输出：

- `outputs/aloepri_stage_catalog.json`

---

## 6. Qwen：当前入口

Qwen 默认入口只有主线文档：

- [`docs/论文一致最终部署主线.md`](docs/论文一致最终部署主线.md)

历史 `standard-shape` 与旧 `Stage K` 工件仅作为历史证据，不作为当前默认路径。

---

## 7. Llama-3.2-3B：最常用工件

### 原始模型路径（云端）

```text
/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B
```

### Stage J 工件

- `artifacts/stage_j_llama_real_full_square/`
- `artifacts/stage_j_llama_real_full_square_tiny_a/`

### Stage K release

- `artifacts/stage_k_llama_release/`

### 推荐 profile

- `tiny_a`

### 最简单推理

```bash
conda run --no-capture-output -n qwen-transformers python scripts/infer_stage_k_release.py \
  --release-dir artifacts/stage_k_llama_release \
  --profile tiny_a \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

### baseline smoke

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_llama_baseline_smoke.py \
  --model-dir /home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B \
  --device cuda \
  --dtype bfloat16
```

### 一键跑真实 baseline / Stage I / Stage J

```bash
bash scripts/run_llama_3b_server_pipeline.sh
```

### 一键推进真实噪声定标与 Stage K

```bash
bash scripts/run_llama_3b_stagek_pipeline.sh
```

---

## 8. 仓库结构

- `src/`
  - 历史阶段实现与核心能力
- `src/aloepri/`
  - 模块化能力层
  - 当前重点包括：
    - `adapters/`
    - `token_ops.py`
    - `pipelines/stage_a.py`
    - `pipelines/standard_shape.py`
    - `pipelines/release.py`
    - `catalog.py`
- `scripts/`
  - 阶段脚本、导出脚本、推理脚本、client/server 辅助脚本
- `tests/`
  - 回归测试
- `docs/`
  - 阶段报告、总报告、使用文档
- `artifacts/`
  - 导出工件
- `outputs/`
  - JSON 回归结果

---

## 9. 当前边界

当前仓库已完成方法主链复现与多条工程证据沉淀，但 Qwen 论文一致最终线仍有关键边界：

1. `vLLM` 侧尚未形成完整稳定路径
2. 安全攻击评估（VMA / IA / ISA / IMA 等）尚未启动

因此当前项目的准确定位是：

> **Llama 线已形成可用交付；Qwen 线以论文一致最终部署线为唯一主线，final Stage J/K 与同口径评测仍在推进。**

---

## 10. 许可与说明

本仓库当前主要包含：

- 代码
- 文档
- 脚本
- 测试

默认 `.gitignore` 已忽略：

- `artifacts/`
- `outputs/`
- `model/`

因此远端仓库默认不包含：

- 本地模型权重
- 大体积实验工件
