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

补充说明：

这里的 `release` 不等于“又发明一种新模型格式”。

更准确地说：

- **HF 工件层**：
  - 指单个标准 Hugging Face 模型目录
  - 例如：
    - `config.json`
    - `tokenizer.json`
    - `model.safetensors`
  - 只是其中参数已经被混淆

- **Stage K release 层**：
  - 是把多个这样的标准 HF 混淆工件再组织成一个统一交付包
  - 多出来的东西包括：
    - `catalog.json`
    - `deployment_contract.json`
    - `profiles/`
    - 统一推理入口

所以可以这样理解：

> **单个 obfuscated HF checkpoint = “像 Hugging Face 下载模型那样的目录，只是参数被混淆”**  
> **Stage K release = “若干个这样的 obfuscated HF checkpoint，再加一层交付包装”**

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

### 这些 Llama 工件怎么理解

当前最关键的两个工件是：

#### `artifacts/stage_i_llama_real`

- 最小标准 HF 混淆模型
- 只混 `token-space`
- 适合做 correctness baseline 和 client/server 契约验证

#### `artifacts/stage_j_llama_real_full_square`

- full-layer 标准 HF 混淆模型
- 混 `token-space + full hidden-space`
- 更接近真实交付目标

一句话区分：

- `Stage I`：最小混淆 HF 工件
- `Stage J`：完整 full-layer 混淆 HF 工件

### 还没完成的部分

此前和 Qwen 相比，Llama 还没有**完成**：

- `Stage K` 风格的统一 release 包装
- `vLLM` 侧实跑
- 更系统的噪声扫描 / 安全评估

其中前两项中的 Stage K 包装，现在已经完成：

- `scripts/run_stage_j_llama_real_noise_calibration.py`
- `scripts/export_stage_k_llama_release.py`
- `scripts/run_llama_3b_stagek_pipeline.sh`

当前仍未完成的主要是：

- `vLLM` 侧实跑
- 更系统的安全/攻击评估

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
| Stage J 噪声定标 | 完成 | 完成 |
| Stage K 统一发布目录 | 完成 | 完成 |
| 云端真实 correctness | 完成 | 完成 |
| vLLM 实跑 | 环境阻塞 | 尚未开始 |
| 安全/攻击评估 | 尚未开始 | 尚未开始 |

---

## 5. 当前建议如何表述项目状态

当前更准确的项目状态表述应该是：

> **Qwen 主线已经完成 A–K 的完整复现与交付包装；Llama-3.2-3B 也已经完成架构接入、Stage I、Stage J、真实 4090 correctness 验证、真实噪声定标与独立 Stage K release 包装。**

如果需要对接手者进一步解释“它和官网下载的 HF 模型有什么区别”，建议使用下面这句：

> **这些工件在目录结构和加载方式上与官方 HF 模型兼容，但它们不是明文模型，而是 obfuscated checkpoint；server 端只持有混淆后的标准 HF 工件，client 端必须额外持有 `client_secret.pt` 来完成输入映射和输出恢复。**

---

## 6. 当前最合理的下一步

如果继续沿 Llama 线推进，最自然的下一步不是回头做 A–H，而是：

1. 给 `Llama-3.2-3B` 做独立的 `Stage K release` 包装
2. 视需要再推进：
   - `vLLM`
   - 安全评估
   - 更强噪声点
