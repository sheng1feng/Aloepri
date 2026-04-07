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
conda run --no-capture-output -n qwen-transformers python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
model_dir = "/data/models/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype="auto").eval().cuda()
prompt = "请用一句话介绍你自己。"
text = tokenizer.apply_chat_template(
    [{"role":"user","content":prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
PY
```

### 4.2 导出真实 Llama 的 obfuscated 工件

先导出 Stage I：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_i_llama_mock_checkpoint.py
```

说明：

- 这一步当前脚本使用的是“本地 Llama 元数据 + mock baseline”的写法
- 上云后，下一步建议优先把它改成**直接加载真实 `/data/models/Llama-3.2-3B/` baseline** 再导出
- 如果你要做真实 3B correctness，这一步最终应替换成真实模型导出，而不是 mock 导出

再导出 Stage J：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_j_llama_full_checkpoint.py
```

同样说明：

- 当前本机代码已经把导出路径、artifact schema、client secret 结构准备好了
- 到云端后，应基于真实 `Llama-3.2-3B` baseline 做正式导出

### 4.3 correctness 回归

使用：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/run_llama_remote_validation.py \
  --baseline-model-dir /data/models/Llama-3.2-3B \
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

## 5. 当前结论

当前代码层面已经把云端验证需要的脚本准备好了：

- `scripts/run_llama_remote_validation.py`

但真正的实跑仍依赖：

- 云端真实权重分片
- 云端 GPU 环境
- Llama Stage I / Stage J 真实工件导出

因此当前阶段的定位是：

> **本机负责把脚本和导出链路准备好，云端负责真实 3B 推理验证。**
