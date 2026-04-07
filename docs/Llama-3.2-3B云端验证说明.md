# Llama-3.2-3B 云端验证说明

本文档描述如何把本机已准备好的 `Llama-3.2-3B` 接入代码与工件带到 4090 云服务器上进行真实推理验证。

## 1. 目标

云端只做三件事：

1. 运行 baseline `Llama-3.2-3B`
2. 运行 obfuscated checkpoint
3. 输出 correctness 对比结果

## 2. 先决条件

云服务器应满足：

- NVIDIA RTX 4090
- `nvidia-smi` 可用
- CUDA / PyTorch / Conda 已装好
- 已有真实模型目录，例如：

```text
/data/models/Llama-3.2-3B/
```

且该目录中必须包含：

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors.index.json`
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`

## 3. 本机需要先准备的内容

本机应先完成：

- 代码推送到远端 Git 仓库
- 需要的本机工件导出完成（后续会由真实模型重新导出时可覆盖）

当前本机已准备的参考脚本：

- `scripts/export_stage_i_llama_mock_checkpoint.py`
- `scripts/export_stage_j_llama_full_checkpoint.py`
- `scripts/run_llama_remote_validation.py`
- `scripts/export_stage_i_llama_real_checkpoint.py`
- `scripts/export_stage_j_llama_real_checkpoint.py`
- `scripts/run_llama_baseline_smoke.py`
- `scripts/run_llama_3b_server_pipeline.sh`
- `scripts/run_stage_j_llama_real_noise_calibration.py`
- `scripts/export_stage_k_llama_release.py`
- `scripts/run_llama_3b_stagek_pipeline.sh`

## 4. 云端推荐执行顺序

## 4.0 一次性执行清单（推荐照此顺序）

### A. 服务器准备

1. 登录 4090 云服务器
2. 确认 GPU 可用：

```bash
nvidia-smi
```

3. 拉取最新代码：

```bash
git clone git@github.com:sheng1feng/Aloepri.git
cd Aloepri
git pull origin main
```

4. 创建/更新环境：

```bash
conda env create -f environment.qwen-transformers.yml || true
conda env update -f environment.qwen-transformers.yml --prune
```

5. 确认真正的模型权重目录存在，例如：

```text
/data/models/Llama-3.2-3B/
```

并包含两个真实分片：

- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`

### B. baseline 验证

1. 先验证 baseline 能否正常生成
2. 若 baseline 不通，不继续后面的 obfuscated 验证

### C. 真实工件导出

1. 先导出 Stage I（最小闭环）
2. 再导出 Stage J（standard-shape full-layer）
3. 每导出一步都立即做一次 correctness 回归

### D. correctness 对比

1. baseline vs Stage I
2. baseline vs Stage J
3. 记录：
   - logits 误差
   - greedy 首 token
   - generated ids/text

### E. 结果归档

1. 保留导出工件目录
2. 保留 JSON 回归结果
3. 如有异常，记录报错命令、环境信息和显存占用

### 4.1 baseline smoke

先在云上验证 baseline：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_llama_baseline_smoke.py \
  --model-dir /home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B \
  --device cuda \
  --dtype bfloat16
```

### 4.2 导出真实 Llama 的 obfuscated 工件

先导出 Stage I（真实 3B baseline）：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_i_llama_real_checkpoint.py \
  --model-dir /home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B \
  --export-dir artifacts/stage_i_llama_real \
  --dtype bfloat16 \
  --device cpu
```

说明：

- 这一步会直接加载真实 `Llama-3.2-3B` baseline 再导出
- 导出的模型保存在仓库目录下：
  - `artifacts/stage_i_llama_real/`

再导出 Stage J：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_j_llama_real_checkpoint.py \
  --model-dir /home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B \
  --export-dir artifacts/stage_j_llama_real_full_square \
  --dtype bfloat16 \
  --device cpu
```

同样说明：

- 导出的模型保存在仓库目录下：
  - `artifacts/stage_j_llama_real_full_square/`

### 4.3 correctness 回归

使用：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_llama_remote_validation.py \
  --baseline-model-dir /home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B \
  --server-dir <obfuscated_server_dir> \
  --client-secret <client_secret.pt> \
  --device cuda \
  --dtype bfloat16
```

推荐先跑两轮：

#### 轮次 1：Stage I

- `server-dir` 指向 Stage I 导出目录
- 先确认 token-space 闭环正确

#### 轮次 2：Stage J

- `server-dir` 指向 Stage J full-layer 导出目录
- 再确认 standard-shape full-layer 的 correctness

### 4.4 一键执行

如果你希望按默认路径一次性跑完 baseline、Stage I、Stage J，可以直接执行：

```bash
bash scripts/run_llama_3b_server_pipeline.sh
```

这个脚本默认使用：

- 仓库目录：`/home/nss-d/sf/Aloepri`
- 模型目录：`/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B`

如果路径有变化，可以用环境变量覆盖：

```bash
REPO_DIR=/your/repo MODEL_DIR=/your/model/path bash scripts/run_llama_3b_server_pipeline.sh
```

### 4.5 推进到 Llama Stage K

如果你要继续把 Llama 路线推进到与 Qwen 更接近的交付层级，再执行：

```bash
bash scripts/run_llama_3b_stagek_pipeline.sh
```

这会额外完成：

1. 真实噪声定标
2. `tiny_a` 真实工件导出
3. `tiny_a` correctness 验证
4. `artifacts/stage_k_llama_release/` 导出

## 5. 当前结论

当前代码层面已经把云端验证需要的脚本准备好了：

- `scripts/run_llama_remote_validation.py`

但真正的实跑仍依赖：

- 云端真实权重分片
- 云端 GPU 环境
- Llama Stage I / Stage J 真实工件导出

因此当前阶段的定位是：

> **本机负责把脚本和导出链路准备好，云端负责真实 3B 推理验证。**

---

## 6. 已完成的云端验证结果

当前已在 4090 云服务器上成功完成：

- baseline smoke
- Stage I 真实导出
- Stage I artifact sanity
- Stage I remote validation
- Stage J 真实 full-layer 导出
- Stage J remote validation

对应结果文件：

- `outputs/stage_i_llama/real_artifact_sanity.json`
- `outputs/stage_i_llama/real_remote_validation.json`
- `outputs/stage_j_llama/real_remote_validation.json`

关键结论：

- Stage I：严格成立（full logits = 0，generation 完全一致）
- Stage J：generation 完全一致，logits 存在小幅偏差但不影响恢复正确性

因此这套脚本现在已经不是“准备好待验证”，而是：

> **已经在真实 `Llama-3.2-3B` + 4090 环境上完成过一轮完整验证。**
