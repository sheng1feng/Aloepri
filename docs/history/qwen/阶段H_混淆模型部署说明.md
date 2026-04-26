# 阶段 H 混淆模型部署说明

## 1. 文档目的

本文档面向“如何把当前阶段 H 的混淆模型交付给 server，并完成 client/server 联调”这个具体问题，给出一份**独立、可执行、偏工程化**的说明。

本文档只回答部署相关问题：

1. 当前有哪些可交付工件
2. 推荐交付哪一种
3. server 端需要什么
4. client 端需要什么
5. 输入混淆和输出恢复分别在哪里做
6. 当前这版导出和标准 `save_pretrained` 的关系是什么

不在本文档中展开：

- 方法论推导
- 攻击评估
- 隐私强度分析
- 论文复现结论总览

如果你需要总览，请看：

- `docs/完整复现总报告_阶段A-K.md`
- `docs/阶段H-Attention静态化与噪声定标报告.md`

---

## 2. 当前可交付工件概览

当前阶段 H 已经形成了两套可用导出格式。

## 2.1 方案 A：直接工件格式

目录：

- `artifacts/stage_h_full_obfuscated/`

主要文件：

- `metadata.json`
- `model_state.pt`
- `server_model_state.pt`
- `client_secret.pt`

说明：

- `model_state.pt`
  - 用于本地一体化演示
  - 更适合研发和快速 smoke test
- `server_model_state.pt`
  - 面向 server 侧交付
  - 不包含 client secret
- `client_secret.pt`
  - 面向 client 侧保留
  - 包含 token 侧的映射秘密

当前大小：

- `server_model_state.pt ≈ 4.1G`
- `client_secret.pt ≈ 2.4M`
- 整个目录约 `8.1G`

## 2.2 方案 B：pretrained-like 目录格式

目录：

- `artifacts/stage_h_pretrained/`

结构如下：

- `artifacts/stage_h_pretrained/server/config.json`
- `artifacts/stage_h_pretrained/server/generation_config.json`
- `artifacts/stage_h_pretrained/server/tokenizer.json`
- `artifacts/stage_h_pretrained/server/tokenizer_config.json`
- `artifacts/stage_h_pretrained/server/chat_template.jinja`
- `artifacts/stage_h_pretrained/server/model.safetensors`
- `artifacts/stage_h_pretrained/server/obfuscation_config.json`
- `artifacts/stage_h_pretrained/client/client_secret.pt`
- `artifacts/stage_h_pretrained/manifest.json`

当前大小：

- 整个目录约 `4.3G`

说明：

- 这是当前**最接近 `save_pretrained` 体验**的导出方式
- 但它仍然不是“纯原生 Hugging Face 可直接 `from_pretrained` 的模型”
- 它仍然需要本仓库提供的自定义 loader 来恢复 stage-H 结构

---

## 3. 推荐交付方式

当前推荐用于部署联调的方式是：

> **优先使用 `artifacts/stage_h_pretrained/` 这套 pretrained-like 目录格式。**

原因：

1. 目录结构比 `model_state.pt` 单文件更清晰
2. server / client 的职责边界更明确
3. tokenizer / config / generation_config 都已经放在 server 目录里
4. 更方便后续继续向标准 `save_pretrained` 形态收敛

如果你只是本地演示、快速单机验证，也可以继续使用：

- `artifacts/stage_h_full_obfuscated/model_state.pt`

---

## 4. client / server 的职责划分

这是当前部署里最重要的一条边界。

## 4.1 server 持有什么

server 只应该持有：

### 如果使用直接工件格式

- `artifacts/stage_h_full_obfuscated/server_model_state.pt`
- `artifacts/stage_h_full_obfuscated/metadata.json`

### 如果使用 pretrained-like 格式

- `artifacts/stage_h_pretrained/server/` 整个目录

也就是说，server 侧持有的是：

- 混淆后的模型参数 / buffer
- 运行所需 config / tokenizer / obfuscation metadata

## 4.2 client 持有什么

client 必须单独持有：

### 直接工件格式

- `artifacts/stage_h_full_obfuscated/client_secret.pt`

### pretrained-like 格式

- `artifacts/stage_h_pretrained/client/client_secret.pt`

其中的核心内容是：

- `perm_vocab`
- `inv_perm_vocab`

这两个张量是当前 token 级输入混淆 / 输出恢复的秘密。

## 4.3 什么不能交给 server

当前实现下，**不应**把下面内容交给 server：

- `perm_vocab`
- `inv_perm_vocab`
- client 的原始 plaintext token ids
- client 恢复后的 plaintext output ids

否则当前 token-level privacy 边界就失效了。

---

## 5. 输入混淆和输出恢复在哪里做

## 5.1 输入混淆在哪里做

当前输入侧 token-level 映射在：

- `src/transforms.py`
  - `map_input_ids(...)`

代码位置：

- `src/transforms.py:8`

当前逻辑是：

1. client 本地用 tokenizer 把 prompt 编码成原始 `input_ids`
2. client 用 `perm_vocab` 调 `map_input_ids(...)`
3. 得到混淆 token ids
4. 将这些混淆 token ids 发送给 server

这意味着当前的“输入加密 / 输入混淆”发生在 **client 侧**。

## 5.2 输出恢复在哪里做

当前输出恢复在：

- `src/transforms.py`
  - `restore_logits(...)`
  - `unmap_output_ids(...)`

代码位置：

- `src/transforms.py:13`
- `src/transforms.py:18`

当前主链路里更常用的是：

- server 返回混淆词表空间 logits
- client 用 `restore_logits(...)` 恢复到原词表空间
- client 再做 argmax / decode

因此当前“输出解密 / 输出恢复”也发生在 **client 侧**。

## 5.3 当前推荐的数据流

推荐的数据流是：

### Client

1. `prompt text`
2. `tokenizer(prompt) -> input_ids`
3. `map_input_ids(input_ids, perm_vocab) -> mapped_ids`
4. 发送 `mapped_ids` 给 server

### Server

1. 接收 `mapped_ids`
2. 运行混淆模型
3. 返回：
   - 混淆词表空间 logits
   - 或混淆 token ids

### Client

1. 若 server 返回 logits：
   - `restore_logits(logits, perm_vocab)`
   - 再 argmax / decode
2. 若 server 返回 token ids：
   - `unmap_output_ids(output_ids, inv_perm_vocab)`
   - 再 decode

---

## 6. 当前是否“可以直接交付给 server”

答案是：

> **可以，但要区分“server 工件可交付”和“通用标准模型格式”是两回事。**

## 6.1 可以直接交付的含义

当前“可以直接交付给 server”的准确含义是：

- server 能拿到一份 stage-H 混淆模型工件
- server 能在当前仓库代码下恢复并运行这个模型
- client 端单独保留输入/输出映射秘密

从这个意义上讲，当前答案是：

> **是，已经可以交付给 server。**

## 6.2 当前还不是的含义

但当前还不是：

- 纯 Hugging Face 原生 `AutoModelForCausalLM.from_pretrained(...)` 直接可加载
- 完全脱离本仓库自定义 loader 的“标准通用模型”

所以更准确地说：

> 当前已经有“可部署的 server 工件”，但还不是“完全标准化的通用发布模型格式”。

---

## 7. 当前 loader / 导出入口

## 7.1 直接工件格式

### 保存

- `src/stage_h_artifact.py:66`
  - `save_stage_h_artifact(...)`

### 完整加载（含 client secret）

- `src/stage_h_artifact.py:98`
  - `load_stage_h_artifact(...)`

### 只加载 server 工件

- `src/stage_h_artifact.py:136`
  - `load_stage_h_server_artifact(...)`

### 导出脚本

- `scripts/export_stage_h_model.py`

### 推理脚本

- `scripts/infer_stage_h_model.py`

## 7.2 pretrained-like 格式

### 导出逻辑

- `src/stage_h_pretrained.py`

### 导出脚本

- `scripts/export_stage_h_pretrained.py`

### 推理脚本

- `scripts/infer_stage_h_pretrained.py`

当前这条链已经实际验证通过：

- 导出成功
- 加载成功
- 推理成功

---

## 8. 推荐部署方式

当前推荐的实际部署方式是：

## 8.1 server 侧

部署：

- `artifacts/stage_h_pretrained/server/`

并在服务代码里用：

- `src/stage_h_pretrained.py`

的 loader 重建模型。

## 8.2 client 侧

保留：

- `artifacts/stage_h_pretrained/client/client_secret.pt`

client 负责：

- prompt 本地 tokenization
- `map_input_ids(...)`
- `restore_logits(...)` / `unmap_output_ids(...)`
- 最终 decode

## 8.3 为什么推荐这套

因为它兼顾了：

1. 当前实现真的能跑
2. 工件边界清晰
3. 目录形态已经接近 `save_pretrained`
4. 后续继续标准化最方便

---

## 9. 推荐的最小联调流程

## 9.1 生成工件

```bash
conda run --no-capture-output -n qwen-transformers \
  python scripts/export_stage_h_pretrained.py
```

## 9.2 本地演示推理

```bash
conda run --no-capture-output -n qwen-transformers \
  python scripts/infer_stage_h_pretrained.py \
  --server-dir artifacts/stage_h_pretrained/server \
  --client-secret artifacts/stage_h_pretrained/client/client_secret.pt \
  --prompt "请用一句话介绍你自己。" \
  --max-new-tokens 8
```

## 9.3 client / server 真正联调时建议的 API 粒度

当前最推荐的 API 设计不是“发明文文本给 server”，而是：

- client 发送：
  - `mapped_input_ids`
  - `attention_mask`
- server 返回：
  - `permuted_logits`
  - 或 `permuted_next_token_ids`

这样最符合当前实现的 privacy boundary。

---

## 10. 当前局限

即使已经做到了阶段 H，当前部署链还有几个需要明确的局限：

1. 当前 tokenizer / config / obfuscation metadata 虽然已经被整理到 pretrained-like 目录，但模型仍依赖仓库内自定义 loader；
2. 当前 client secret 仍是单独的 `.pt` 文件，不是更高层封装的 key management 方案；
3. 当前 server API 还没有抽成独立服务程序，仍以导出脚本 / 推理脚本 / loader 为主；
4. 当前尚未进入安全攻击评估阶段，因此“可部署”不等于“隐私强度已经定量证明充分”。

---

## 11. 一句话总结

> **当前已经有一版可交付给 server 的 stage-H 混淆模型工件，并且进一步整理出了一套更接近 `save_pretrained` 的目录格式；输入混淆和输出恢复都在 client 侧完成，server 只运行混淆模型本身。**
