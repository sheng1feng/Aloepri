# 阶段 K：标准形状交付包装报告

本文档记录阶段 K 的第一轮推进结果。

阶段 K 的目标不是再改模型数学，而是把已经验证通过的 **standard-shape full-layer** 工件，整理成一个更适合交付、演示和后续服务化接入的发布目录。

也就是说，阶段 K 处理的是：

- **交付包装**
- **profile 选择**
- **统一推理入口**

而不是：

- 再做协变恢复
- 再做 vLLM backend 排障
- 再做攻击评估

---

## 1. 背景

当前已经有两套经过验证的 standard-shape full-layer 工件：

### 1.1 零噪声参考工件

- `artifacts/stage_j_full_square/`

特点：

- `avg_full_logits_max_abs_error ≈ 1.26e-4`
- `generated_ids/text exact match = 1.0`

适合：

- correctness baseline
- regression
- 排障

### 1.2 推荐非零噪声工件

- `artifacts/stage_j_full_square_tiny_a/`

特点：

- `alpha_e = 0.02`
- `alpha_h = 0.01`
- `generated_ids/text exact match = 1.0`

适合：

- 默认交付
- 演示
- 后续安全性准备

---

## 2. 本轮新增内容

### 2.1 Stage-K release catalog

新增：

- `src/stage_k_release.py`

核心能力：

- 定义 release profiles
- 生成阶段 K 发布目录
- 生成统一 catalog / contract / README

当前默认 profiles：

- `stable_reference`
- `tiny_a`

### 2.2 导出脚本

新增：

- `scripts/export_stage_k_release.py`

作用：

- 把现有阶段 J 工件组织成统一的阶段 K 发布目录

### 2.3 推理脚本

新增：

- `scripts/infer_stage_k_release.py`

作用：

- 不再手工写 server/client 路径
- 只需要指定：
  - `release_dir`
  - `profile`
  - `prompt`

即可推理

---

## 3. 发布目录结构

当前导出目录：

- `artifacts/stage_k_release/`

结构：

```text
artifacts/stage_k_release/
├── catalog.json
├── deployment_contract.json
├── README.md
└── profiles/
    ├── stable_reference -> ../../artifacts/stage_j_full_square
    └── tiny_a -> ../../artifacts/stage_j_full_square_tiny_a
```

注意：

- 当前默认是 **symlink-based packaging**
- 这样不会复制多份 4G+ 模型权重
- 更适合研发与本地交付整理

如果后续需要真正“拷贝出一份可独立打包分发”的目录，可以在导出脚本上继续扩展 `--materialize` 路径。

---

## 4. catalog 内容

文件：

- `artifacts/stage_k_release/catalog.json`

当前提供的信息包括：

- `format = stage_k_release_v1`
- `recommended_profile = tiny_a`
- `stable_reference_profile = stable_reference`
- 每个 profile 的：
  - 名称
  - 描述
  - 推荐用途
  - server/client 相对路径
  - metadata
  - regression summary

这意味着：

> 阶段 K 现在已经不需要用户自己记住“哪个目录是零噪声、哪个目录是 tiny_a”。

---

## 5. deployment contract

文件：

- `artifacts/stage_k_release/deployment_contract.json`

内容明确规定了：

- client 持有：
  - `client_secret.pt`
- server 持有：
  - `server/config.json`
  - `server/generation_config.json`
  - `server/model.safetensors`
  - `server/tokenizer.json`
  - `server/tokenizer_config.json`
  - `server/chat_template.jinja`
- 输入映射函数：
  - `src/transforms.py::map_input_ids`
- 输出恢复函数：
  - `src/transforms.py::restore_logits`
  - `src/transforms.py::unmap_output_ids`

这使得阶段 K 已经具备一个**清晰的 client/server 契约**。

---

## 6. 实测结果

### 6.1 release 导出

命令：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/export_stage_k_release.py
```

结果：

- `artifacts/stage_k_release/catalog.json` 已生成
- 当前 profiles：
  - `stable_reference`
  - `tiny_a`

### 6.2 profile 推理

命令：

```bash
conda run --no-capture-output -n qwen-transformers python scripts/infer_stage_k_release.py --profile tiny_a --prompt "请用一句话介绍你自己。" --max-new-tokens 8
```

实测输出：

- `我是一个人工智能助手，可以帮助您解答`

这说明：

> 阶段 K 这套发布目录不是“只有 catalog，没有可运行入口”，而是已经能按 profile 名称直接推理。  

---

## 7. 当前阶段 K 的定位

阶段 K 当前做成的是：

### 已完成

- 把 standard-shape full-layer 工件整理成统一发布目录
- 提供 profile 选择
- 提供统一 inference CLI
- 提供 catalog / deployment contract / README

### 还没完成

- materialized copy 版本的大体积独立打包
- 更正式的 server API 封装
- 更高级的 baseline-free SDK / client 库

因此当前阶段 K 更准确的表述是：

> **第一轮 standard-shape 交付包装已完成。**

---

## 8. 一句话结论

> 阶段 K 已经把 `stage_j_full_square` 与 `stage_j_full_square_tiny_a` 收拢成一个统一的发布目录，并提供了基于 profile 的推理入口；当前 standard-shape 路线已经不仅是“模型可用”，而且开始具备“可交付使用”的形态。**
