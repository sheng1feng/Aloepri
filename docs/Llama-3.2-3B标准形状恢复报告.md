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

## 5. 当前边界

本机阶段尚未完成的内容：

- 未进行真实 `Llama-3.2-3B` 权重推理
- 未验证真实 28-layer / 3072 hidden / 8 KV heads 的数值行为
- 未做 vLLM

这些工作将移交到云端 4090 环境完成。
