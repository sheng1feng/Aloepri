# 阶段 I：vLLM 接入复现报告

> Legacy note: 本文档对应的是 **旧版 Stage I**，其重点是 `Stage A` 标准 HF/vLLM 入口打通与 feasibility probe。新版 `Stage I` 的 canonical 定义见 `docs/阶段I_部署约束验证报告.md`。

本文档记录阶段 I 的实际实现、回归结果与当前边界。阶段 I 采用 `docs/阶段I_vLLM接入计划.md` 的主线定义，并按“**先做 Stage A 可部署化，不做模块化抽象**”执行。

---

## 1. 目标与执行范围

阶段 I 本轮只做以下事情：

1. 把 **Stage A 词表置换闭环** 导出成标准 HF checkpoint 目录，作为 vLLM Phase 1 的可部署对象；
2. 固定 client / server 的在线映射契约；
3. 在 **Transformers 路径** 下做严格 correctness 回归；
4. 给出 **vLLM 可行但当前环境未安装** 的回归脚本与 Phase 2 可行性结论。

本轮明确**不做**：

- 不把 Stage H 的 KeyMat / fused norm / fused attention 直接塞进 vLLM；
- 不做攻击评估；
- 不做隐私参数扫描；
- 不进入 `src/aloepri/` 模块化引擎。

换句话说，阶段 I 的职责现在应**收紧**为：

> **把“可部署的标准 HF/vLLM 形态”打通，并识别出继续进入标准 shape 物化时的真实技术阻塞点。**

---

## 2. 实现概览

### 2.1 新增脚本

- `scripts/export_stage_i_vllm_checkpoint.py`
  - 生成 Stage A 导出目录：
    - `artifacts/stage_i_vllm/server/`
    - `artifacts/stage_i_vllm/client/client_secret.pt`
- `scripts/run_stage_i_artifact_sanity.py`
  - 检查导出前后参数、置换和 token 分区是否一致。
- `scripts/run_stage_i_hf_regression.py`
  - 用 Transformers 加载导出目录，验证与 baseline 的严格一致性。
- `scripts/run_stage_i_vllm_regression.py`
  - 如果环境存在 `vllm`，则直接跑生成回归；当前环境缺少 `vllm`，脚本会输出 `skipped` JSON。
- `scripts/run_stage_i_phase2_feasibility.py`
  - 输出“当前哪些组件有希望继续 materialize 到标准 HF/vLLM 权重”的可行性表。

### 2.2 新增 helper

- `src/stage_i_vllm.py`
  - `export_stage_i_vllm_checkpoint(...)`
  - `load_stage_i_hf_bundle(...)`
  - `summarize_token_partitions(...)`
  - `build_phase2_feasibility_summary()`

### 2.3 在线契约

阶段 I 固定采用：

- **client 持有 secret**
- **token id 级映射**

具体流程：

1. client 明文 prompt → tokenizer → `input_ids`
2. client 调 `src/transforms.py` 中的 `map_input_ids(...)`
3. server 只加载 `artifacts/stage_i_vllm/server/` 里的标准 checkpoint 并运行
4. client 对输出做恢复：
   - logits 路径：`restore_logits(...)`
   - token id 路径：`unmap_output_ids(...)`

这一路径避免了 `detokenize -> re-tokenize`，因此不会额外引入 tokenizer/chat template 漂移。

---

## 3. 导出产物

阶段 I 导出目录：

```text
artifacts/stage_i_vllm/
├── server/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
├── client/
│   └── client_secret.pt
├── manifest.json
└── stage_i_metadata.json
```

其中：

- `server/` 可被 `AutoModelForCausalLM.from_pretrained(...)` 直接加载；
- `client_secret.pt` 至少包含：
  - `perm_vocab`
  - `inv_perm_vocab`
- `stage_i_metadata.json` 记录：
  - seed
  - dtype
  - token 分区摘要
  - Phase / variant 信息

---

## 4. 实际结果

### 4.1 Artifact sanity

结果文件：

- `outputs/stage_i/artifact_sanity.json`

关键结果：

- `server_load_success = true`
- `perm_vocab_match_export = true`
- `inv_perm_vocab_match_export = true`
- `embed_weight_max_abs_diff = 0.0`
- `lm_head_weight_max_abs_diff = 0.0`
- `max_parameter_abs_diff = 0.0`
- `parameter_count_checked = 290`
- `perm_is_valid = true`
- `special_ids_fixed = true`
- `tail_rows_fixed = true`

解释：

- 导出后的 `server/` 目录重新加载后，与内存中构造的 Stage A 模型参数完全一致；
- 说明当前导出不是“近似导出”，而是**标准 HF 权重形态下的精确物化**；
- `special` 与 `tail rows` 的稳定策略在导出过程中未被破坏。

### 4.2 HF 回归（FP32）

结果文件：

- `outputs/stage_i/hf_regression_fp32.json`

汇总结果：

- `avg_full_logits_max_abs_error = 0.0`
- `avg_full_logits_mean_abs_error = 0.0`
- `avg_last_token_logits_max_abs_error = 0.0`
- `avg_last_token_logits_mean_abs_error = 0.0`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`
- `baseline_has_nan_or_inf = false`
- `stage_a_has_nan_or_inf = false`

解释：

- 在 FP32 下，导出的 Stage I 标准 checkpoint 与原 baseline 在 5 条固定 prompt 上保持**严格一致**；
- 这证明：
  - Stage A 的 weight materialization 是正确的；
  - client 侧 token/logits 映射方向没有写反；
  - 导出后的 checkpoint 没有破坏模型功能。

### 4.3 HF 回归（BF16）

结果文件：

- `outputs/stage_i/hf_regression_bf16.json`

汇总结果：

- `avg_full_logits_max_abs_error = 0.0`
- `avg_full_logits_mean_abs_error = 0.0`
- `avg_last_token_logits_max_abs_error = 0.0`
- `avg_last_token_logits_mean_abs_error = 0.0`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`
- `baseline_has_nan_or_inf = false`
- `stage_a_has_nan_or_inf = false`

解释：

- 仅就 Phase 1（Stage A 纯词表置换）而言，BF16 下仍然保持严格一致；
- 因此阶段 I 的推荐默认部署 dtype 可以接受 BF16，不必强制退回 FP32。

### 4.4 vLLM 回归与环境阻塞点

结果文件：

- `outputs/stage_i/vllm_regression.json`
- `outputs/stage_i/vllm_regression_qwenvllm.json`

当前结果：

- `available = false`
- `skipped = true`
- `reason = "vllm is unavailable in the current environment: No module named 'vllm'"`

解释：

- 阶段 I 的 vLLM 回归脚本已实现；
- 但当前 `qwen-transformers` 环境没有安装 `vllm`，因此本轮只能做到：
  - 完成导出格式
  - 完成 client/server 契约
  - 完成 HF correctness 回归
  - 提供可直接执行的 vLLM 回归入口
- 一旦环境补齐 `vllm`，可直接运行：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_i_vllm_regression.py
```

随后本轮又额外做了真正的安装与实跑尝试：

1. 克隆独立环境 `qwen-vllm`
2. 在该环境中安装 `vllm==0.18.1`
3. 用 `qwen-vllm` 环境运行：

```bash
conda run --no-capture-output -n qwen-vllm python scripts/run_stage_i_vllm_regression.py --dtype float32 --device cpu --output-path outputs/stage_i/vllm_regression_qwenvllm.json
```

得到的新结果是：

- `available = true`
- `skipped = true`
- `reason = "vllm import succeeded but runtime initialization failed: RuntimeError: Device string must not be empty"`

这说明：

- 通用 PyPI wheel 虽然安装成功，但**没有在当前机器上激活 CPU backend**；
- 当前机器环境是：
  - 无 NVIDIA GPU
  - x86_64 / AVX2
- 进一步按照官方 CPU 安装思路尝试切换 CPU wheel 时，又遇到当前环境对 GitHub Release 的连接超时，导致官方 CPU wheel 未能拉取成功。

因此，阶段 I 在本机的真实状态应更新为：

> **代码路径、导出目录、client/server 契约和 vLLM 回归脚本都已准备好；但本机尚未拿到可运行的 vLLM CPU backend，因此没有完成最终的 vLLM 实跑闭环。**

进一步地，在阶段 J 完成 non-expanding / full-layer 标准 checkpoint 后，又额外做了一次重新验证：

```bash
conda run --no-capture-output -n qwen-vllm python scripts/run_stage_i_vllm_regression.py \
  --server-dir artifacts/stage_j_full_square/server \
  --client-secret artifacts/stage_j_full_square/client/client_secret.pt \
  --dtype float32 \
  --device cpu \
  --output-path outputs/stage_j/full_layers_square_vllm_regression.json
```

结果仍然是：

- `available = true`
- `skipped = true`
- `reason = "vllm import succeeded but runtime initialization failed: RuntimeError: Device string must not be empty"`

这说明现在的结论已经进一步收敛为：

> **即使换成阶段 J 的 full-layer 标准形状模型，vLLM 这条线在本机仍然卡在 CPU backend 环境，而不是卡在模型 checkpoint 形状或功能正确性。**

---

## 5. Phase 2 可行性评估

结果文件：

- `outputs/stage_i/phase2_feasibility.json`

当前结论：

- **可立即落到标准 HF/vLLM 权重的部分**
  - Stage A vocab permutation
- **当前会被 KeyMat 扩维直接阻断的部分**
  - embedding/head noise + KeyMat
  - FFN fused path
  - Stage H attention staticized path
- **当前不适合直接进 vLLM kernel**
  - RMSNorm fused path

最关键判断是：

> 阶段 I 的 Phase 1 已经成立；但在**当前扩维 KeyMat**方案下，Phase 2 的主要阻碍并不只是 RMSNorm，而是 embedding/head、attention、FFN 的参数 shape 已经先脱离了原始标准 HF 架构。

### 5.1 Phase 2 shape/materialization probe

结果文件：

- `outputs/stage_i/phase2_probe.json`

关键结果：

- baseline hidden size：`896`
- KeyMat expanded size：`1152`
- `all_components_directly_copyable = false`
- `copyable_component_count = 0`
- `blocked_component_count = 10`

最关键的 shape 证据包括：

- `embedding_weight`：`[151936, 1152] -> [151936, 896]`
- `lm_head_weight`：`[151936, 1152] -> [151936, 896]`
- `layer0_q_proj_weight`：`[896, 1152] -> [896, 896]`
- `layer0_o_proj_weight`：`[1152, 896] -> [896, 896]`
- `layer0_gate_proj_weight`：`[4864, 1152] -> [4864, 896]`
- `layer0_down_proj_weight`：`[1152, 4864] -> [896, 4864]`
- `layer0_input_layernorm_metric_matrix`：`[1152, 1152] -> [896]`

这说明：

- 当前 Algorithm-1 KeyMat 不是 square hidden transform，而是把 hidden 从 `896` 扩到 `1152`；
- 因此不仅 RMSNorm 语义已经不是标准 RMSNorm，连 embedding/head、attention、FFN 的参数 shape 也已经不再匹配原始 Qwen2 checkpoint；
- 所以在当前设计下，Phase 2 不是“逐层慢慢写回标准权重”这么简单，而是首先要面对：

> **扩维 KeyMat 与“保持原始标准 HF/vLLM checkpoint 形状不变”之间的结构性冲突。**

---

## 6. 与阶段 G/H 的关系

阶段 I 不是要把 G/H 全量塞进 vLLM，而是做一个**兼容性最强的部署闭环**。

所以阶段 I 当前得到的最重要结论不是“Stage H 已经能跑 vLLM”，而是：

1. 当前仓库已经有一条**完全标准 HF checkpoint 形态**的导出路线；
2. 这条路线在功能上保持与 baseline 的严格一致；
3. client / server 职责边界已经明确；
4. 若要继续做 Phase 2，必须先处理 **KeyMat 扩维与标准 checkpoint shape 不兼容** 这一阶问题，而不能只把注意力放在 RMSNorm 上。

### 6.1 阶段 I、J、K 的边界

为了避免后续编号继续混淆，这里明确三者分工：

- **阶段 I**
  - 关注“标准导出 / client-server 契约 / HF-vLLM 加载与回归”
  - 关键词：**部署入口、标准 checkpoint、可加载**
- **阶段 J**
  - 关注“non-expanding / 标准 shape 方案如何恢复功能正确性”
  - 关键词：**block0、prefix、多层协变恢复**
- **阶段 K**
  - 关注“已经可用的标准形态如何进一步交付/服务化/通用化”
  - 关键词：**导出包装、baseline-free loader、分发体验**

因此：

> 阶段 I 不是去解决所有 non-expanding 中间层恢复问题；那是阶段 J 的任务。  
> 阶段 I 也不是去做更漂亮的产品化交付；那是阶段 K 的任务。

---

## 7. 阶段 I 当前判定

如果按 `docs/阶段I_vLLM接入计划.md` 的 I-A 必做主线来判定，本轮阶段 I 可认为：

- **I-1 标准权重：已完成**
  - 已导出标准 HF checkpoint
  - `AutoModelForCausalLM.from_pretrained(server_dir)` 可直接加载
- **I-2 在线契约：已完成**
  - client/server secret 与映射路径已经固定
- **I-3 回归指标：已完成**
  - HF 路径在 FP32 / BF16 下都严格一致
- **I-4 dtype 策略：基本完成**
  - Phase 1 的 BF16 已验证稳定

当前唯一未在本地环境完成的是：

- **vLLM 实际运行验证**
- **Phase 2 的非扩维标准权重设计**

前者当前是 CPU backend / 网络获取官方 wheel 的环境阻塞；后者当前是方法与标准 checkpoint 形状之间的结构性冲突。

---

## 8. 下一步建议

阶段 I 后续最自然的两条路是：

### 路线 A：补齐 vLLM 环境并完成 M2

目标：

- 实际启动 vLLM
- 让 `server/` 目录直接进入推理服务
- 用现有 `client_secret.pt` 跑通固定 prompts

### 路线 B：重构 Phase 2 的 hidden transform 设计，再继续物化

目标：

- 先把“扩维 KeyMat”改造成“非扩维 / 保持原始 hidden shape 的可逆变换”
- 再继续把更多稳定结构写回标准权重
- 同时保持 vLLM kernel 兼容

对应设计文档：

- `docs/阶段I_Phase2_非扩维可逆变换设计.md`
- `docs/阶段I_Phase2_最小原型报告.md`

按当前 probe 结果，建议顺序：

1. 先切换到 non-expanding square transform
2. 先打通 embed/head-only
3. 再做 block0 / prefix-2
4. 最后再回到 full-layer / vLLM

补充说明：

- 本轮已经实际做完了“square transform + embed/head-only + 标准 checkpoint 导出 + HF 回归”这个最小原型；
- 结果表明：
  - shape 完全对齐
  - 但功能完全不成立
- 因此 Phase 2 的下一步被明确锁定为：

> **不是继续微调 embed/head-only，而是把 non-expanding hidden transform 推进到 block0 的中间层。**

后续这一步已经在阶段 J 中被继续推进，并且目前已经拿到：

- `prefix-1 / 2 / 4 / 8 / full-layer`
- 以及 `full-layer` 标准 checkpoint 导出回归

对应结果见：

- `docs/阶段J_标准形状前缀恢复报告.md`
- `docs/阶段J_标准形状噪声定标报告.md`

---

## 9. 一句话结论

> 阶段 I 已经完成 **Stage A 的标准 HF/vLLM Phase-1 可部署化复现**：导出目录正确、client/server 契约明确、HF 路径在 FP32/BF16 下严格一致；当前剩余缺口一是本机尚未拿到可运行的 vLLM CPU backend，二是当前扩维 KeyMat 与“继续物化到原始标准 checkpoint”之间存在结构性冲突。
