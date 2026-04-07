# Llama-3.2-3B 噪声定标与 Stage K 推进说明

本文档说明在 `Llama-3.2-3B` 已完成：

- 结构接入
- Stage I
- Stage J
- 真实 4090 correctness 验证

之后，如何继续把其工作进度推进到与 Qwen 更接近的交付层级。

---

## 1. 当前目标

当前 Llama 路线已经完成：

- Stage I（真实 3B）
- Stage J（真实 3B）

下一步要补齐的是：

1. **真实 Llama 噪声定标**
2. **Llama 专属 Stage K release**

---

## 2. 已新增脚本

### 2.1 真实噪声定标

- `scripts/run_stage_j_llama_real_noise_calibration.py`

输出：

- `outputs/stage_j_llama/real_noise_calibration.json`

### 2.2 导出推荐非零工作点

复用：

- `scripts/export_stage_j_llama_real_checkpoint.py`

例如导出 `tiny_a`：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_j_llama_real_checkpoint.py \
  --model-dir /home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B \
  --export-dir artifacts/stage_j_llama_real_full_square_tiny_a \
  --dtype bfloat16 \
  --device cpu \
  --alpha-e 0.02 \
  --alpha-h 0.01
```

### 2.3 Llama Stage K release

- `src/stage_k_llama_release.py`
- `scripts/export_stage_k_llama_release.py`

输出：

- `artifacts/stage_k_llama_release/`

### 2.4 一键推进脚本

- `scripts/run_llama_3b_stagek_pipeline.sh`

它会依次执行：

1. 真实噪声定标
2. 导出 `tiny_a` 真实工件
3. 跑 `tiny_a` remote validation
4. 导出 `stage_k_llama_release`

---

## 3. 当前状态

当前已经完成的是：

- 真实 `Llama-3.2-3B` 噪声定标
- `tiny_a` 真实工件导出
- `tiny_a` 真实 correctness 验证
- `artifacts/stage_k_llama_release/` 导出

对应关键结果：

### 3.1 真实噪声定标结果

结果文件：

- `outputs/stage_j_llama/real_noise_calibration.json`

当前排序：

- `stable_reference`
- `tiny_a`
- `tiny_b`
- `small_a`
- `small_c`
- `small_b`
- `paper_like`

说明：

- `stable_reference` 仍然是最稳的 correctness 基线
- `tiny_a` 是当前最佳非零工作点
- `paper_like` 在真实 `Llama-3.2-3B` 上明显过强，不适合作为默认部署点

### 3.2 `tiny_a` 真实验证结果

结果文件：

- `outputs/stage_j_llama/real_tiny_a_remote_validation.json`

关键结果：

- `avg_full_logits_max_abs_error = 0.2578125`
- `avg_last_token_logits_max_abs_error = 0.159375`
- `greedy_first_token_match_rate = 1.0`
- `generated_ids_exact_match_rate = 1.0`
- `generated_text_exact_match_rate = 1.0`

说明：

> `tiny_a = (alpha_e=0.02, alpha_h=0.01)` 在真实 `Llama-3.2-3B` 上保持了完整的 generation correctness，因此可作为当前推荐非零工作点。

### 3.3 Llama Stage K release

结果文件：

- `artifacts/stage_k_llama_release/catalog.json`

当前已经包含两个 profile：

- `stable_reference`
- `tiny_a`

其中：

- `recommended_profile = "tiny_a"`
- `stable_reference_profile = "stable_reference"`

因此当前可以正式给出结论：

> **Llama-3.2-3B 已经完成噪声定标与 Stage K release 包装，在交付层级上已经与 Qwen 基本对齐。**
