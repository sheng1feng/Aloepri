# Qwen 与 Llama 复现阶段区分说明

本文档用于回答三个问题：

1. **阶段 K 到底是干什么的？**
2. **Qwen 主线目前复现到哪一步？**
3. **Llama-3.2-3B 目前复现到哪一步？**

当前仓库同时存在两条相关但不完全相同的路线：

- **Qwen 主线**：最早、最完整、实验最充分的主线
- **Llama 扩展线**：在 Qwen 主线验证基础上，向真实 `Llama-3.2-3B` 推进的结构兼容与标准 HF 工件路线

---

## 1. 阶段 K 是干什么的？

阶段 K 的目标不是再改模型数学，而是：

> **把已经验证通过的 standard-shape full-layer 工件，整理成更适合交付、演示和部署使用的统一发布目录。**

阶段 K 处理的是：

- 工件组织
- profile 管理
- catalog
- deployment contract
- 统一推理入口

阶段 K **不处理**：

- attention / FFN / norm 的数学恢复
- 新的 hidden transform 设计
- vLLM backend 排障
- 安全攻击评估

一句话理解：

> **阶段 J 解决“模型能不能变成标准形状并恢复功能”，阶段 K 解决“这些模型工件能不能被整理成可交付、可选 profile、可直接推理的发布目录”。**

---

## 2. Qwen 路线：当前到哪一步

Qwen 是当前仓库最完整的主线。

### 已完成

- **阶段 A**：词表置换闭环
- **阶段 B**：hidden-space 入口与 attention 子层恢复
- **阶段 C**：block0 完整恢复
- **阶段 D**：多层传播验证
- **阶段 E**：复杂 attention 结构接入与修复
- **阶段 F**：Algorithm 1 / KeyMat 接入
- **阶段 G**：KeyMat 去 bridge 化 / 参数级融合
- **阶段 H**：attention 静态化收敛与噪声定标
- **阶段 I**：标准 HF/vLLM 入口打通
- **阶段 J**：standard-shape full-layer 恢复
- **阶段 J（噪声）**：standard-shape 非零噪声工作点定标
- **阶段 K**：standard-shape 工件统一交付包装

### 当前状态

Qwen 路线已经达到：

> **研究复现主链 + standard-shape full-layer + release 交付包装**

---

## 3. Llama-3.2-3B 路线：当前到哪一步

Llama 不是从 A 开始独立重跑一遍，而是在 Qwen 主线已成立的基础上，优先走：

- adapter 接入
- Stage I 标准 HF 工件
- Stage J standard-shape full-layer
- 云端真实 3B correctness 验证

### 已完成

#### 结构接入

- `LlamaArchitectureAdapter` 已接入
- `AloePriConfig.from_model(...)` / `AloePriEngine.from_model(...)` 已支持 `llama_decoder`

#### 本机阶段

- Stage I mock 导出 / 回归：已完成
- Stage J mock full-layer 导出 / 回归：已完成

#### 云端真实 3B 阶段

已在 4090 云服务器上完成：

- baseline smoke
- Stage I 真实导出
- Stage I artifact sanity
- Stage I remote validation
- Stage J 真实 full-layer 导出
- Stage J remote validation

### 当前结论

Llama 当前已经达到：

> **真实 `Llama-3.2-3B` 可导出为混淆后的标准 HF 格式模型，并完成了 4090 上的 correctness 验证。**

### 还没完成的部分

和 Qwen 相比，Llama 当前还没有**完成**：

- `Stage K` 风格的统一 release 包装
- `vLLM` 侧实跑
- 更系统的噪声扫描 / 安全评估

但当前已经具备继续推进所需的脚本：

- `scripts/run_stage_j_llama_real_noise_calibration.py`
- `scripts/export_stage_k_llama_release.py`
- `scripts/run_llama_3b_stagek_pipeline.sh`

因此 Llama 当前最准确的阶段定位是：

- **结构接入：完成**
- **Stage I：完成**
- **Stage J：完成**
- **Stage K：尚未独立做成 Llama release 包装**

---

## 4. Qwen 与 Llama 的阶段对照

| 能力 | Qwen | Llama-3.2-3B |
|---|---|---|
| 架构接入 | 完成 | 完成 |
| Stage I 标准 HF 工件 | 完成 | 完成 |
| Stage J standard-shape full-layer | 完成 | 完成 |
| Stage J 噪声定标 | 完成 | 脚本已准备，待云端执行 |
| Stage K 统一发布目录 | 完成 | 脚本已准备，待云端执行 |
| 云端真实 correctness | 完成 | 完成 |
| vLLM 实跑 | 环境阻塞 | 尚未开始 |
| 安全/攻击评估 | 尚未开始 | 尚未开始 |

---

## 5. 当前建议如何表述项目状态

当前更准确的项目状态表述应该是：

> **Qwen 主线已经完成 A–K 的完整复现与交付包装；Llama-3.2-3B 已完成架构接入、Stage I、Stage J 和真实 4090 correctness 验证，但尚未独立推进到 Stage K 的统一 release 包装。**

---

## 6. 当前最合理的下一步

如果继续沿 Llama 线推进，最自然的下一步不是回头做 A–H，而是：

1. 给 `Llama-3.2-3B` 做独立的 `Stage K release` 包装
2. 视需要再推进：
   - `vLLM`
   - 安全评估
   - 更强噪声点
