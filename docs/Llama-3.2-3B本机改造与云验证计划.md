# Llama-3.2-3B 本机改造与云验证计划

本文档用于明确：

1. **本机阶段到底做什么**
2. **每一步对应的闭环模块是什么**
3. **每一步如何验证成功**
4. **哪些内容先不抽象，等验证通过后再模块化**

当前前提：

- 本机只有 CPU，不承担真实 `Llama-3.2-3B` 推理闭环
- 云服务器有 `RTX 4090`，真实 3B 推理验证放在云端完成
- 本地已有：
  - `model/Llama-3.2-3B/config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - 其他配置/许可证文件
- 超大 `safetensors` 分片不在本机开发路径内，本机阶段不依赖真实 3B 权重

---

## 1. 当前真实卡点

当前仓库要支持 `Llama-3.2-3B`，卡点不是“数学路线没想清楚”，而是下面三件具体事情：

### 1.1 架构适配层只有 Qwen

当前模块化入口主要通过：

- `src/aloepri/adapters/qwen.py`
- `AloePriConfig.from_model(...)`
- `AloePriEngine.from_model(...)`

来识别和构造模型配置。

这意味着当前仓库还**没有正式接入 Llama adapter**。

### 1.2 attention backend 大量写死在 Qwen2

当前大量 attention 相关实现直接依赖：

- `transformers.models.qwen2.modeling_qwen2`

这会阻断：

- Stage B–H 旧链路
- Stage J standard-shape 链路
- 后续 Llama full-layer 恢复

因此真正要改的核心，不只是 adapter，而是 **attention backend**。

### 1.3 本机不能承担真实 3B 推理验证

因此本机阶段必须明确采用：

- **小型随机 Llama mock model 做结构验证**
- **真实 3B 云端做推理验证**

不能把“本机代码接入”和“云端真实推理验证”混成一步。

---

## 2. 本机阶段的总目标

本机阶段不追求真实 3B 推理速度，也不做云部署本身。

本机阶段只追求：

> **把仓库代码改造到可以支持 Llama 架构，并产出可直接拿去云端验证的标准工件。**

更具体地说，本机阶段要完成：

1. 识别 `LlamaForCausalLM`
2. 打通 Llama 的 attention backend
3. 跑通 Stage I 的标准导出路径
4. 跑通 Stage J 的 standard-shape 路线
5. 导出可推到云端验证的标准 checkpoint

---

## 3. 这轮开发原则

### 原则 1：先改现有链路，不先做抽象

这轮优先做的是：

- 让 `Llama-3.2-3B` 能接进当前仓库主线
- 一步步验证
- 每通过一个闭环，再继续下一步

这轮**不优先做**：

- 过早统一成完美抽象层
- 先追求 Qwen / Llama 全统一接口
- 先做新的大模块重构

### 原则 2：每一层只做一个闭环

本机改造按闭环推进，不一口气全改：

1. adapter 闭环
2. attention backend 闭环
3. Stage I 闭环
4. Stage J block0 闭环
5. Stage J prefix/full 闭环
6. 云端工件闭环

### 原则 3：通过后再抽象模块化

这轮不是放弃模块化，而是：

> **先验证 Llama 路线在当前仓库主线中成立，再把“已验证通过”的实现提炼进统一模块层。**

---

## 4. 闭环模块设计

下面把本机阶段拆成六个闭环模块。

### 闭环 A：Llama 架构识别闭环

#### 目标

让仓库能够识别 `LlamaForCausalLM` 并提取结构配置。

#### 需要新增/改动

- 新增：`src/aloepri/adapters/llama.py`
- 更新：
  - `src/aloepri/adapters/__init__.py`
  - `src/aloepri/config.py`
  - `src/aloepri/engine.py`

#### 输出

- `LlamaArchitectureAdapter`
- `build_llama_config(...)`
- `is_llama_compatible_model(...)`

#### 验证方式

使用小型随机 `LlamaConfig(...)` 构造 mock model，检查：

- hidden size
- layer count
- attention heads
- kv heads
- rope theta
- tied embeddings

#### 验收标准

- `AloePriConfig.from_model(...)` 可接 Llama
- `AloePriEngine.from_model(...)` 可接 Llama
- 不需要手工填写结构参数

---

### 闭环 B：Llama attention backend 闭环

#### 目标

让当前 attention 路线可以在 Llama 上 forward。

#### 需要新增/改动

- 新增：
  - `src/llama_attention_backend.py` 或等价 helper
- 改动：
  - 所有当前直接依赖 `qwen2.modeling_qwen2` 的 attention 关键路径

#### 范围

本闭环不重写 attention 数学，只替换 backend：

- `apply_rotary_pos_emb`
- `ALL_ATTENTION_FUNCTIONS`
- `eager_attention_forward`
- `LlamaAttention`

#### 验证方式

使用小型随机 Llama mock model：

- 跑 block0 forward
- 检查 `q/k/v/o` shape
- 检查无 NaN / Inf

#### 验收标准

- block0 attention wrapper 在 Llama 上能正常 forward
- rotary / kv-group / head reshape 无明显错位

---

### 闭环 C：Stage I 标准导出闭环（Llama 版）

#### 目标

先打通最稳的部署路径：

- token permutation
- embedding/head materialization
- 标准 HF checkpoint 导出

#### 需要新增/改动

- 新增：
  - `scripts/export_stage_i_llama_checkpoint.py`
  - `scripts/run_stage_i_llama_hf_regression.py`
- 如有需要：
  - `src/stage_i_llama.py`

#### 输出

- `artifacts/stage_i_llama/server/`
- `artifacts/stage_i_llama/client/client_secret.pt`

#### 验证方式

本机仍使用小型随机 Llama mock model 做：

- export sanity
- load sanity
- shape 检查

#### 验收标准

- 导出目录能被 HF 正常加载
- client secret 完整
- special token / tail rows 策略未被破坏

---

### 闭环 D：Stage J block0 标准形状闭环（Llama 版）

#### 目标

在不改变 hidden shape 的前提下，先让 block0 恢复成立。

#### 需要新增/改动

- 新增：
  - `src/stage_j_llama.py`
  - `scripts/run_stage_j_llama_block0.py`

#### 范围

继续沿用当前 standard-shape 路线：

- square monomial hidden transform
- attention / FFN / norm / residual 适配

#### 验证方式

本机用随机 mock model 验证：

- `input_norm_out`
- `attn_out`
- `post_attn_norm_out`
- `mlp_out`
- `block_out`

#### 验收标准

- block0 不报错
- 中间量 shape 正常
- 输出比 `embed/head-only` 明显更合理

---

### 闭环 E：Stage J prefix/full-layer 闭环（Llama 版）

#### 目标

把 Llama 的 standard-shape 路线推进到：

- `prefix-2`
- `prefix-4`
- `full-layer`

#### 需要新增/改动

- 新增：
  - `scripts/run_stage_j_llama_prefix.py`
  - `scripts/export_stage_j_llama_full_checkpoint.py`

#### 输出

- `artifacts/stage_j_llama_full_square/`

#### 本机验证方式

由于本机不跑真实 3B，这一步只做：

- shape
- export
- load
- 小型 mock model smoke test

#### 验收标准

- full-layer 工件能导出
- checkpoint schema 正确
- 可以直接带去云端验证

---

### 闭环 F：云端验证准备闭环

#### 目标

把本机开发成果整理成一套云端可执行方案。

#### 输出

- 云端执行脚本
- 工件目录
- 运行说明
- baseline / obfuscated 对照命令

#### 验收标准

- 本机已经不再需要额外改代码
- 云上只需要：
  - 拉代码
  - 放权重分片
  - 建环境
  - 运行脚本

---

## 5. 本机开发顺序

严格按下面顺序推进。

### Step 1

先完成 **闭环 A：adapter**

通过后再继续。

### Step 2

完成 **闭环 B：attention backend**

只要 block0 forward 能跑，再进入下一步。

### Step 3

完成 **闭环 C：Stage I 导出**

先拿到最小可部署工件。

### Step 4

完成 **闭环 D：Stage J block0**

不直接冲 full-layer。

### Step 5

完成 **闭环 E：Stage J prefix/full-layer**

先 prefix，再 full-layer。

### Step 6

完成 **闭环 F：云端验证准备**

把工件和命令整理好，等待上云。

---

## 6. 每一步必须产出的文档

这轮要求“有意义的改动都以文档记录”，因此每个闭环至少要有：

### A/B 阶段

- `docs/Llama-3.2-3B本机接入报告.md`

记录：

- adapter 设计
- attention backend 改造
- 本机 mock model 验证结果

### C/D/E 阶段

- `docs/Llama-3.2-3B标准形状恢复报告.md`

记录：

- Stage I 导出结果
- Stage J block0 / prefix / full-layer 结果
- 工件路径

### F 阶段

- `docs/Llama-3.2-3B云端验证说明.md`

记录：

- 云端环境准备
- baseline / obfuscated 命令
- 期望输出
- 常见错误

---

## 7. 本机阶段暂不做的事

为了避免范围失控，这轮本机阶段明确先不做：

- 不做 vLLM 接入
- 不做攻击评估
- 不做隐私参数扫描
- 不做多模型统一抽象重构
- 不做 KeyMat 扩维路线迁移到 Llama

只做：

> **Llama 结构接入 + standard-shape 路线 + 云端验证准备**

---

## 8. 阶段完成标准

本机阶段完成，至少要满足：

1. 仓库能识别 `LlamaForCausalLM`
2. attention backend 能在 Llama mock model 上 forward
3. 能导出一版 Llama 标准 HF 工件
4. 能导出一版 Llama standard-shape full-layer 工件
5. 云端验证脚本和说明已准备好

只要这五条成立，就可以进入云服务器 4090 验证。

---

## 9. 当前建议的下一步

下一步直接开始：

> **闭环 A：新增 `LlamaArchitectureAdapter`，并把 `AloePriConfig.from_model(...)` / `AloePriEngine.from_model(...)` 接上。**

这是整个 Llama 路线的入口。

---

## 10. 当前执行进度

### 已完成：闭环 A（Llama 架构识别）

已完成内容：

- 新增 `src/aloepri/adapters/llama.py`
- `AloePriConfig.from_model(...)` 已支持自动分发到 Llama
- `AloePriEngine.from_model(...)` 已支持自动分发到 Llama
- 新增了 Llama adapter 单测

当前状态：

- 仓库已经能够正式识别 `LlamaForCausalLM`
- `Qwen` 与 `Llama` 不再混淆识别
- 对应单测已通过：
  - `tests/test_aloepri_llama_adapter.py`

### 新发现：Stage I / Stage J 本地 smoke 不一定先被 attention backend 阻断

在本机 smoke 中已经确认：

- `Stage I` 标准导出链路可以在 **小型随机 Llama mock model** 上跑通
- `Stage J` standard-shape 构造也可以在 **小型随机 Llama mock model** 上跑通

这意味着：

> 对于当前以 `Stage I / Stage J / Stage K` 为主的 Llama 接入目标，本机阶段不必先把旧的 Qwen-only attention backend 全部抽象完，仍然可以继续推进标准形状主线。

因此后续执行顺序调整为：

1. 继续把 `Stage I` 本地闭环固化
2. 再把 `Stage J` block0 / prefix / full-layer 的 Llama 路线固化
3. 之后再决定是否需要进一步统一旧 attention backend 抽象

### 本机 smoke 结果

本机已新增：

- `scripts/run_llama_local_smoke.py`

输出文件：

- `outputs/llama_local_smoke.json`

当前结果表明：

- `AloePriConfig.from_model(...)` 已正确识别 `llama_decoder`
- `AloePriEngine.from_model(...)` 已正确识别 `llama_decoder`
- Stage I 标准导出链路可在小型随机 Llama mock model 上跑通
- Stage J standard-shape 构造链路可在小型随机 Llama mock model 上跑通

相关测试：

- `tests/test_stage_j_llama_square.py`

当前全量测试状态：

- `pytest -q` 已通过
- 当前总计 `51 passed`

### 已完成：Stage I / Stage J 本机闭环固化

当前已新增并跑通：

- `scripts/export_stage_i_llama_mock_checkpoint.py`
- `scripts/run_stage_i_llama_mock_regression.py`
- `scripts/export_stage_j_llama_full_checkpoint.py`
- `scripts/run_stage_j_llama_mock_regression.py`

产物：

- `artifacts/stage_i_llama_mock/`
- `artifacts/stage_j_llama_mock_full_square/`

结果：

- Stage I mock 回归：`outputs/stage_i_llama/mock_regression.json`
- Stage J mock 回归：`outputs/stage_j_llama/mock_full_regression.json`

当前结论：

> 在本机 CPU 条件下，围绕 `Llama-3.2-3B` 的标准导出链路与 standard-shape full-layer 链路，已经全部推进到“可导出、可加载、可回归”的状态。
