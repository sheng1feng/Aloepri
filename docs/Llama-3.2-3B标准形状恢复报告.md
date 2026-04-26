> Canonical note: 本文档只回答当前 `Llama-3.2-3B` 标准形状恢复与 correctness 证据，不承担全局主线说明。Llama 唯一主线入口见 [docs/Llama-3.2-3B最终部署主线.md](Llama-3.2-3B最终部署主线.md)。

# Llama-3.2-3B 标准形状本机恢复报告

本文档记录当前在**本机 CPU 环境**下，围绕 `Llama-3.2-3B` 所能完成的标准形状（standard-shape）接入与验证工作。

## 1. 本机目标

本机阶段不承担真实 3B 推理，而是完成：

1. `Llama` 架构适配
2. Stage I 标准导出链路 smoke
3. Stage J standard-shape full-layer 链路 smoke
4. 导出可直接带去云端验证的脚本与工件格式

## 2. 当前本机可执行入口

### 2.1 Adapter / Stage I / Stage J 本机 smoke

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_llama_local_smoke.py
```

输出：

- `outputs/llama_local_smoke.json`

### 2.2 Stage I mock 导出

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_i_llama_mock_checkpoint.py
```

输出：

- `artifacts/stage_i_llama_mock/`

### 2.3 Stage I mock 回归

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_i_llama_mock_regression.py
```

输出：

- `outputs/stage_i_llama/mock_regression.json`

### 2.4 Stage J mock full-layer 导出

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_j_llama_full_checkpoint.py
```

输出：

- `artifacts/stage_j_llama_mock_full_square/`

### 2.5 Stage J mock full-layer 回归

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_stage_j_llama_mock_regression.py
```

输出：

- `outputs/stage_j_llama/mock_full_regression.json`

## 3. 当前结论

当前本机阶段已经验证：

- `LlamaArchitectureAdapter` 已接入
- `AloePriConfig.from_model(...)` / `AloePriEngine.from_model(...)` 已支持 `llama_decoder`
- Stage I 标准导出路径可在随机 Llama mock model 上跑通
- Stage J standard-shape full-layer 路径可在随机 Llama mock model 上跑通

这说明：

> 当前仓库针对 `Llama-3.2-3B` 的本机改造，已经从“只能规划”推进到“代码链路已成立，可导出、可回归、可为云端验证准备工件”的状态。

## 4. 已完成的本机结果

### 4.1 本机 smoke

结果文件：

- `outputs/llama_local_smoke.json`

当前结果表明：

- adapter 路径正确识别 `llama_decoder`
- Stage I 导出链路 smoke 正常
- Stage J standard-shape 构造链路 smoke 正常

### 4.2 Stage I mock 回归

结果文件：

- `outputs/stage_i_llama/mock_regression.json`

关键结果：

- `avg_full_logits_max_abs_error = 0.0`
- `avg_last_token_logits_max_abs_error = 0.0`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

说明：

> 基于本地 `Llama-3.2-3B` tokenizer/config 与随机 mock Llama baseline，Stage I 标准导出链路已经严格成立。

### 4.3 Stage J mock full-layer 回归

结果文件：

- `outputs/stage_j_llama/mock_full_regression.json`

关键结果：

- `avg_full_logits_max_abs_error ≈ 4.29e-7`
- `avg_last_token_logits_max_abs_error ≈ 2.68e-7`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

说明：

> 基于本地 `Llama-3.2-3B` tokenizer/config 与随机 mock Llama baseline，Stage J 的 standard-shape full-layer 路线已经在本机闭环成立。

## 5. 云端真实 `Llama-3.2-3B` 验证结果

云端真实模型目录：

- `/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B`

云端仓库目录：

- `/home/nss-d/sf/Aloepri`

### 5.1 Stage I artifact sanity

结果文件：

- `outputs/stage_i_llama/real_artifact_sanity.json`

关键结果：

- `server_load_success = true`
- `perm_vocab_match_export = true`
- `inv_perm_vocab_match_export = true`
- `embed_weight_max_abs_diff = 0.0`
- `lm_head_weight_max_abs_diff = 0.0`
- `max_parameter_abs_diff = 0.0`
- `parameter_count_checked = 254`
- `perm_is_valid = true`
- `special_ids_fixed = true`
- `tail_rows_fixed = true`

解释：

> 真实 `Llama-3.2-3B` 的 Stage I 导出不是近似导出，而是标准 HF 工件形态下的精确物化。

### 5.2 Stage I remote validation

结果文件：

- `outputs/stage_i_llama/real_remote_validation.json`

关键结果：

- `avg_full_logits_max_abs_error = 0.0`
- `avg_full_logits_mean_abs_error = 0.0`
- `avg_last_token_logits_max_abs_error = 0.0`
- `avg_last_token_logits_mean_abs_error = 0.0`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

解释：

> 真实 `Llama-3.2-3B` 上，Stage I（词表空间闭环）已经严格成立。

### 5.3 Stage J remote validation

结果文件：

- `outputs/stage_j_llama/real_remote_validation.json`

关键结果：

- `avg_full_logits_max_abs_error ≈ 0.19385`
- `avg_full_logits_mean_abs_error ≈ 0.01972`
- `avg_last_token_logits_max_abs_error ≈ 0.14805`
- `avg_last_token_logits_mean_abs_error ≈ 0.02167`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

解释：

> 真实 `Llama-3.2-3B` 上，Stage J（standard-shape full-layer）已经在 **generation correctness** 层面完全恢复；同时 logits 仍有小幅数值偏差，但不影响 greedy 和短生成结果。

### 5.4 当前正式结论

基于真实 3B 云端结果，现在可以正式给出结论：

1. **Stage I：通过**
   - 严格数值一致
   - 导出工件正确
   - generation 完全一致

2. **Stage J：通过**
   - full-layer 标准形状路线在真实 3B 上成立
   - generation 完全一致
   - logits 有可解释的小量级偏差，但不构成失败

因此当前 `Llama-3.2-3B` 已经达到：

> **可导出为混淆后的标准 HF 格式模型，并在真实 4090 服务器上完成 correctness 验证。**

## 6. 当前边界

本机阶段尚未完成的内容：

- 未进行真实 `Llama-3.2-3B` 权重推理
- 未验证真实 28-layer / 3072 hidden / 8 KV heads 的数值行为
- 未做 vLLM

上述前两项已在云端完成；当前仍未完成的是：

- `vLLM` 路径
- 更强噪声点下的稳定性评估
- 安全/攻击评估
