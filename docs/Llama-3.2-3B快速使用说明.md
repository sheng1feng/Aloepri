# Llama-3.2-3B 快速使用说明

这份文档面向**不需要理解混淆原理、只想正常使用模型进行推理**的同学。

你只需要知道：

1. 模型在哪里
2. 仓库在哪里
3. 哪个工件是推荐使用的
4. 用什么命令直接推理

不需要先理解：

- KeyMat
- 协变恢复
- attention 结构
- 噪声定标细节

---

## 1. 路径说明

### 仓库路径（云端）

```text
/home/nss-d/sf/Aloepri
```

### 原始 Llama-3.2-3B 模型路径（云端）

```text
/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B
```

这个目录里应包含：

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors.index.json`
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`

---

## 2. 你最应该用哪个工件

如果你只是想直接使用已经准备好的混淆模型进行推理，**优先使用 Stage K release**：

```text
/home/nss-d/sf/Aloepri/artifacts/stage_k_llama_release
```

这是当前最推荐的入口，因为它已经把不同 profile 组织好了。

### 当前可用 profile

- `stable_reference`
  - 零噪声参考版
  - 最适合 correctness 对照和排障

- `tiny_a`
  - 当前推荐的非零噪声版
  - 是默认推荐使用的 profile

### 推荐默认值

如果你不确定选哪个，**直接用 `tiny_a`**。

---

## 3. 一键准备完整 Llama 工件

如果你还没有跑过完整 Llama 处理流程，先在仓库目录执行：

```bash
cd /home/nss-d/sf/Aloepri
bash scripts/run_llama_3b_server_pipeline.sh
bash scripts/run_llama_3b_stagek_pipeline.sh
```

跑完后，你会得到：

- `artifacts/stage_i_llama_real/`
- `artifacts/stage_j_llama_real_full_square/`
- `artifacts/stage_j_llama_real_full_square_tiny_a/`
- `artifacts/stage_k_llama_release/`

如果这些目录已经存在，就不需要重复跑。

---

## 4. 最简单的推理方式

### 方式 A：直接用 Stage K release 推理（推荐）

进入仓库目录：

```bash
cd /home/nss-d/sf/Aloepri
```

运行：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/infer_stage_k_release.py \
  --release-dir artifacts/stage_k_llama_release \
  --profile tiny_a \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

如果你想用零噪声参考版：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/infer_stage_k_release.py \
  --release-dir artifacts/stage_k_llama_release \
  --profile stable_reference \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

### 方式 B：client/server 分离使用

如果你不是一体化运行，而是要把模型交给 server 使用、client 单独做输入输出处理，请看：

- `docs/Llama-3.2-3B客户端与Server使用说明.md`

那份文档会直接告诉你：

- server 该加载哪个目录
- client 该拿哪个 `client_secret.pt`
- client 怎么准备输入
- client 怎么恢复输出

---

## 5. 如果你只想验证原始模型能不能跑

运行 baseline：

```bash
cd /home/nss-d/sf/Aloepri
conda run --no-capture-output -n qwen-transformers python scripts/run_llama_baseline_smoke.py \
  --model-dir /home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B \
  --device cuda \
  --dtype bfloat16
```

这个命令跑的是**原始模型**，不是混淆模型。

---

## 6. 如果你想用单个工件，而不是 release

### Stage I 工件

目录：

```text
/home/nss-d/sf/Aloepri/artifacts/stage_i_llama_real
```

它是：

- 最小标准 HF 混淆模型
- 更适合 correctness baseline

### Stage J 工件

目录：

```text
/home/nss-d/sf/Aloepri/artifacts/stage_j_llama_real_full_square
```

它是：

- full-layer 标准 HF 混淆模型
- 更接近最终完整使用形态

### 推荐

如果你只是想**直接用**，通常不建议手动挑 Stage I / Stage J 目录，而是直接走：

```text
artifacts/stage_k_llama_release
```

因为它已经把 profile 和推荐值整理好了。

---

## 7. 输出文件在哪里

运行过程中生成的结果文件主要在：

```text
/home/nss-d/sf/Aloepri/outputs/
```

其中和 Llama 最相关的是：

- `outputs/stage_i_llama/`
- `outputs/stage_j_llama/`

---

## 8. 常见使用场景

### 场景 1：我只想直接推理

用：

```bash
python scripts/infer_stage_k_release.py --release-dir artifacts/stage_k_llama_release --profile tiny_a --prompt "你好"
```

### 场景 2：我想确认 baseline 正常

用：

```bash
python scripts/run_llama_baseline_smoke.py --model-dir /home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B --device cuda --dtype bfloat16
```

### 场景 3：我想重新导出和验证完整工件

用：

```bash
bash scripts/run_llama_3b_server_pipeline.sh
bash scripts/run_llama_3b_stagek_pipeline.sh
```

---

## 9. 当前推荐结论

对于一般使用者：

- **原始模型路径**：
  - `/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B`
- **推荐直接使用的混淆工件**：
  - `/home/nss-d/sf/Aloepri/artifacts/stage_k_llama_release`
- **推荐默认 profile**：
  - `tiny_a`

一句话：

> **如果你不想理解内部原理，就直接用 `artifacts/stage_k_llama_release` + `profile=tiny_a` 推理。**
