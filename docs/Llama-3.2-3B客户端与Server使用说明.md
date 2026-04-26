> Canonical note: 本文档只回答当前 `Llama-3.2-3B` 的 client/server 使用方式，不承担全局主线说明。Llama 唯一主线入口见 [docs/Llama-3.2-3B最终部署主线.md](Llama-3.2-3B最终部署主线.md)。

# Llama-3.2-3B 客户端与 Server 使用说明

这份文档面向实际使用者，目标是：

- **server 侧**像使用标准 Hugging Face 模型一样使用混淆后的模型
- **client 侧**知道如何准备输入、恢复输出

本文档尽量不解释复杂原理，只讲：

- 路径
- 工件
- 命令
- 使用方式

---

## 1. 先明确两部分文件

### server 侧工件

推荐使用：

```text
/home/nss-d/sf/Aloepri/artifacts/stage_k_llama_release/profiles/tiny_a/server
```

这就是一个**标准 HF 目录**，里面有：

- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors`

所以 server 侧可以像普通 HF 模型一样加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

server_dir = "/home/nss-d/sf/Aloepri/artifacts/stage_k_llama_release/profiles/tiny_a/server"
tokenizer = AutoTokenizer.from_pretrained(server_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(server_dir, trust_remote_code=True, torch_dtype="auto")
```

### client 侧 secret

client 必须持有：

```text
/home/nss-d/sf/Aloepri/artifacts/stage_k_llama_release/profiles/tiny_a/client/client_secret.pt
```

里面保存了：

- `perm_vocab`
- `inv_perm_vocab`

client 负责：

- 输入映射
- 输出恢复

---

## 2. 如果只是本地一体化推理，最简单命令

```bash
cd /home/nss-d/sf/Aloepri
conda run --no-capture-output -n qwen-transformers python scripts/infer_stage_k_release.py \
  --release-dir artifacts/stage_k_llama_release \
  --profile tiny_a \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

这条命令已经把 client/server 的映射逻辑都包在一起了。

---

## 3. 如果是正式 client/server 分离，怎么做

## 3.1 client 准备输入

client 侧执行：

```bash
cd /home/nss-d/sf/Aloepri
conda run --no-capture-output -n qwen-transformers python scripts/llama_client_prepare_request.py \
  --server-dir artifacts/stage_k_llama_release/profiles/tiny_a/server \
  --client-secret artifacts/stage_k_llama_release/profiles/tiny_a/client/client_secret.pt \
  --prompt "请用一句话介绍你自己。" \
  --output-path outputs/llama_client_request.json
```

这会生成：

- `outputs/llama_client_request.json`

里面有：

- `input_ids`
- `mapped_input_ids`
- `attention_mask`

**真正发送给 server 的应该是：**

- `mapped_input_ids`
- `attention_mask`

而不是 plaintext 的 `input_ids`。

---

## 3.2 server 正常加载 HF 模型并推理

server 侧像普通 HF 一样加载：

```python
import json
import torch
from transformers import AutoModelForCausalLM

server_dir = "/home/nss-d/sf/Aloepri/artifacts/stage_k_llama_release/profiles/tiny_a/server"
request_path = "/home/nss-d/sf/Aloepri/outputs/llama_client_request.json"

model = AutoModelForCausalLM.from_pretrained(
    server_dir,
    trust_remote_code=True,
    torch_dtype="auto",
).eval().cuda()

payload = json.load(open(request_path, "r", encoding="utf-8"))
input_ids = torch.tensor([payload["mapped_input_ids"]], device="cuda")
attention_mask = torch.tensor([payload["attention_mask"]], device="cuda")

generated = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=8,
    do_sample=False,
)

generated_token_ids = generated[0, input_ids.shape[1]:].tolist()
print({"generated_token_ids": generated_token_ids})
```

### 关键点

对 server 来说，这就是**普通 HF 模型**：

- 正常 `from_pretrained`
- 正常 `generate`
- 正常接收 `input_ids`

所以如果后续要做：

- model split
- tensor parallel
- 其他基于标准 HF 权重的服务化改造

server 工件本身是兼容这类工作流的。

---

## 3.3 client 恢复输出

假设 server 返回的是：

```json
{"generated_token_ids": [123, 456, 789]}
```

client 侧执行：

```bash
cd /home/nss-d/sf/Aloepri
conda run --no-capture-output -n qwen-transformers python scripts/llama_client_restore_ids.py \
  --server-dir artifacts/stage_k_llama_release/profiles/tiny_a/server \
  --client-secret artifacts/stage_k_llama_release/profiles/tiny_a/client/client_secret.pt \
  --mapped-token-ids "123,456,789" \
  --output-path outputs/llama_client_restored.json
```

这会输出：

- `restored_token_ids`
- `decoded_text`

---

## 4. 如果 server 返回 logits 而不是 token ids

当前仓库里 client 侧恢复 logits 的逻辑在：

```text
src/transforms.py::restore_logits
```

也就是说：

- 如果 server 返回 logits
- client 就应该用 `restore_logits(...)` 先恢复
- 再做 argmax / sampling / decode

如果只是普通生成服务，通常直接恢复 token ids 就够了。

---

## 5. 推荐给使用者的最短说明

如果你要把模型交给别人用，可以直接告诉他：

### 原始模型路径

```text
/home/nss-d/dcy/codes/ModelSplit/models/Llama-3.2-3B
```

### 推荐使用的混淆模型

```text
/home/nss-d/sf/Aloepri/artifacts/stage_k_llama_release/profiles/tiny_a/server
```

### client secret

```text
/home/nss-d/sf/Aloepri/artifacts/stage_k_llama_release/profiles/tiny_a/client/client_secret.pt
```

### 最简单推理命令

```bash
conda run --no-capture-output -n qwen-transformers python scripts/infer_stage_k_release.py \
  --release-dir /home/nss-d/sf/Aloepri/artifacts/stage_k_llama_release \
  --profile tiny_a \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

---

## 6. 当前推荐

如果你只是正常使用：

- **release 目录**：
  - `artifacts/stage_k_llama_release`
- **推荐 profile**：
  - `tiny_a`

如果你要做 correctness 对照或排障：

- 用：
  - `stable_reference`
