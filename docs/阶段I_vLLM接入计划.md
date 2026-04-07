# 阶段 I：vLLM 接入计划（任务清单与验收标准）

本文档定义“阶段 I（vLLM 接入准备）”的目标、范围、任务分解与验收标准，用于把当前以 Transformers 为主的研究复现形态推进到可被 vLLM 以标准 HF checkpoint 方式加载并服务的工程形态。

> 备注（已根据实际执行收紧）：当前阶段 I 的**必做主线**已经明确收敛到 `I-A`（Stage A 标准 HF/vLLM 入口打通）。  
> 原文中的 `I-B` 现在不再理解为“继续在阶段 I 内完成中间层恢复”，而是作为**为阶段 J 提供输入的可行性探索**。  
> 阶段 J 的任务已单独定义为：在保持标准 checkpoint shape 的前提下，继续完成 non-expanding 方案的 block0 / prefix 功能恢复。

## 0. 背景与动机

当前仓库 Stage F~H 的实现大量依赖替换模块/自定义 forward（例如静态化 Attention、KeyMat 融合 Norm/FFN）：

- Stage H 组装：[stage_h.py](file:///home/shengfeng/Privacy-inference/src/stage_h.py)
- Attention 静态化：[stage_h_attention_static.py](file:///home/shengfeng/Privacy-inference/src/stage_h_attention_static.py)
- KeyMat 融合 RMSNorm：[stage_g_norm.py](file:///home/shengfeng/Privacy-inference/src/stage_g_norm.py)
- KeyMat 融合 FFN：[stage_g_ffn.py](file:///home/shengfeng/Privacy-inference/src/stage_g_ffn.py)

vLLM 的核心前提是加载标准 HF checkpoint 并使用其高性能 kernel（KV cache、PagedAttention、fused norm/MLP）。这意味着：

- “能导出权重”不等于“能被 vLLM 当作标准模型加载并执行”
- 阶段 I 的关键是把现有混淆实现“物化（materialize）成标准权重形态”并定义可部署的在线映射契约

## 1. 阶段 I 总目标（Outcome）

阶段 I 结束时需要满足以下 4 条：

- I-1 标准权重：导出一个可被 vLLM 直接加载的 HF checkpoint（权重命名/shape 与原模型架构一致）
- I-2 在线契约：明确并实现输入映射/输出反映射路径（secret 放置、映射位置、输出还原）
- I-3 回归指标：提供最小但严格的 correctness 回归（logits/greedy token 等）
- I-4 dtype 策略：明确 FP32→BF16 的数值稳定性边界与规避策略

## 2. 范围划分（推荐的两条支线）

### I-A（必做，优先）：vLLM Phase 1（Stage A 可部署化）

目标：只做词表置换闭环（token permutation + embedding/head 行列置换），不引入 KeyMat/非标准 Norm/自定义 Attention。

理由：

- Stage A 的变换可以完全体现在标准权重重排上，不需要改 forward
- 对 vLLM 兼容性最强，是最短闭环

实现基础：

- token/logits 映射工具：[transforms.py](file:///home/shengfeng/Privacy-inference/src/transforms.py)
- Stage A 构造入口：[stage_b.py:prepare_stage_a_model](file:///home/shengfeng/Privacy-inference/src/stage_b.py#L225-L237)

### I-B（可选，第二优先）：vLLM Phase 2（尽量多物化 Stage H，但保持算子语义标准）

目标：在不改变算子语义（尤其不破坏标准 RMSNorm/attention/MLP 语义）的前提下，把更多混淆“写回权重”，让 vLLM kernel 仍可用。

注意：

- 若引入非标准 RMSNorm（例如 metric-based 二次型）或扩维 KeyMat 语义，将直接冲击 vLLM kernel 假设，需要回退或写自定义算子（不建议在阶段 I 做）

## 3. 任务清单（详细）

### I.0 需求与约束确认（Definition of Done）

- 目标：明确“vLLM 接入成功”的标准与部署边界。
- 任务：
  - 确定接入形态：单机验证 / 服务化部署（OpenAI compatible API 或内部 API）
  - 确定 secret 放置：仅客户端持有 / 服务端也持有（影响映射位置）
  - 明确日志/监控策略：是否允许记录明文 token、是否需要脱敏
- 产出：
  - 一份接口契约（输入/输出/密钥/日志）
- 验收标准：
  - 新同学不看代码仅看契约即可实现一个可用 client stub，并能跑通固定 prompts

### I.1 标准 HF checkpoint 导出器（核心）

目标：导出一个 vLLM 能加载的“标准 HF checkpoint”，其结构/权重 key/shape 与原模型一致。

- 任务：
  - 新增脚本（建议）：`scripts/export_stage_i_vllm_checkpoint.py`
  - Phase 1（Stage A）导出逻辑：
    - 生成 `perm_vocab` / `inv_perm_vocab`
    - 写回并保存标准权重：
      - `model.embed_tokens.weight`
      - `lm_head.weight`（以及存在时的 `bias`）
    - 保存 tokenizer/config/generation_config（HF 目录标准）
  - 保存 client secret：
    - `client_secret.pt`（至少包含 `perm_vocab`、`inv_perm_vocab`）
    - 参考 Stage H 的 “server/client” 分离布局：[stage_h_pretrained.py](file:///home/shengfeng/Privacy-inference/src/stage_h_pretrained.py#L39-L83)
- 建议目录布局：
  - `artifacts/stage_i_vllm/`
    - `server/`（标准 HF checkpoint）
    - `client/`（client_secret.pt）
    - `manifest.json`（描述格式/文件清单）
- 验收标准：
  - `AutoModelForCausalLM.from_pretrained(server_dir)` 成功加载
  - vLLM 指向 `server_dir` 能启动并生成（阶段 I 允许先不写自动化启动验证，但至少需要手动验证记录）

### I.2 在线映射实现（client/server 选型与实现）

目标：把“token 置换”和“输出还原”做成可部署流程，并固定约定。

- 任务：
  - 选择一种映射方案并固化：
    - 方案 A（贴论文）：client tokenize→映射→detokenize，server vLLM 走普通 text API
    - 方案 B（更工程）：在服务端进入 vLLM 前做 token 映射（需要插入 preprocessor）
  - 实现最小 client 映射器：
    - 输入映射：`perm_vocab[input_ids]` 或调用 [map_input_ids](file:///home/shengfeng/Privacy-inference/src/transforms.py#L8-L10)
    - 输出还原：
      - logits：优先使用 [restore_logits](file:///home/shengfeng/Privacy-inference/src/transforms.py#L17-L19)
      - token ids：使用 [unmap_output_ids](file:///home/shengfeng/Privacy-inference/src/transforms.py#L12-L14)
- 验收标准：
  - 固定 prompts 下可完成端到端生成，输出可读
  - 不发生 tokenizer special token 破坏（例如 chat template 不一致导致 re-tokenize 行为漂移）

### I.3 回归与指标（必须自动化）

目标：提供明确的 correctness 回归指标，避免导出/映射被重构破坏。

- 任务：
  - 新增回归脚本（建议）：`scripts/run_stage_i_vllm_regression.py`
  - 回归指标建议：
    - last-token logits 的 `max_abs_error` / `mean_abs_error`
    - greedy next token match rate
    - （可选）短生成 exact match rate
  - 复用误差函数：[evaluator.py](file:///home/shengfeng/Privacy-inference/src/evaluator.py)
- 验收标准（Phase 1 的推荐阈值）：
  - FP32 下 Stage A 理论可达到严格一致（误差为 0 或接近 0）
  - 若启用噪声，则必须给出明确阈值（例如 max_abs_error < X，或首 token 命中率 > Y）

### I.4 dtype 与数值稳定性（FP32→BF16）

目标：明确 vLLM 常用 BF16/FP16 下的稳定性边界。

- 任务：
  - 为 Stage I 回归脚本增加 NaN/Inf 检测（logits 与关键中间张量）
  - 给出推荐默认 dtype（Phase 1 应能在 BF16 仍稳定）
  - 若某些配置在 BF16 不稳定，需要记录触发条件与规避策略（例如禁用某些噪声范围、强制 FP32）
- 验收标准：
  - BF16 下固定 prompts 生成不出现 NaN/Inf
  - 指标退化可解释且在阈值内

### I.5（可选）Phase 2 可行性评估表

目标：为后续把更多 Stage H 混淆带入 vLLM 做技术评估输入。

- 任务：
  - 输出一张表：组件→是否可“仅通过写回标准权重实现”→对 vLLM kernel 的影响→预计工程成本
  - 建议重点检查：
    - Attention：是否能完全落到 q/k/v/o 权重重参数化且不改运行时流程
    - FFN：perm/scale 是否能落到 gate/up/down 权重
    - RMSNorm：是否仍是标准 RMSNorm（注意 [stage_g_norm.py](file:///home/shengfeng/Privacy-inference/src/stage_g_norm.py#L24-L45) 的 metric_matrix 语义）
- 验收标准：
  - 对每个组件给出明确结论：可行 / 需要回退设计 / 需要修改 vLLM

## 4. 里程碑（建议）

- I-M1：导出 Stage A 标准 HF checkpoint + client_secret，Transformers 加载 + 回归通过
- I-M2：vLLM 能加载该 checkpoint 启动服务，client 映射/反映射跑通固定 prompts
- I-M3：BF16 下稳定跑通（无 NaN/Inf）+ 指标达标
- I-M4（可选）：完成 Phase 2 可行性评估表，为 Stage J 输入

## 5. 风险登记（阶段 I 重点关注）

- Token 映射如果走“detokenize→re-tokenize”的路径，容易被 tokenizer/chat template 的细节破坏一致性
- dtype：BF16/FP16 下更容易出现溢出/NaN，需要在回归里显式检测并给出规避策略
- “自定义算子语义”会直接阻断 vLLM kernel 复用（尤其是非标准 RMSNorm/扩维 KeyMat 语义）
