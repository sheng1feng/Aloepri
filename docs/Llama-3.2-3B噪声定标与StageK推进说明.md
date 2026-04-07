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

- 代码与脚本准备完毕

当前还未完成的是：

- 真实 4090 上运行这一轮噪声定标与 Stage K release 导出

因此当前最准确的表述是：

> **Llama-3.2-3B 已经具备继续推进到与 Qwen 交付层级一致所需的全部脚本与执行路径；剩余工作是把这些脚本在云端真实跑完。**
