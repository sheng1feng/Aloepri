# 阶段 A-K 复现整理与模块化映射报告

本文档的目标不是再复述每个阶段的实验细节，而是把当前仓库里已经完成的 A–K 复现路线，整理成一套更清晰的**模块化能力地图**。

它回答四个问题：

1. 当前每个阶段到底做了什么
2. 这些阶段现在落在哪些代码文件里
3. 哪些能力已经被提炼成新的模块化入口
4. 还有哪些部分仍然保留为“legacy stage pipeline”

---

## 1. 整体结论

当前仓库已经不是单纯的“按阶段堆脚本”状态，而是开始形成两层结构：

### 第一层：阶段脚本 / 历史实验入口

这些文件仍然保留，作用是：

- 作为实验史
- 作为回归基线
- 作为论文复现路径的可追溯入口

例如：

- `src/stage_b.py`
- `src/stage_c.py`
- `src/stage_d.py`
- `src/stage_e.py`
- `src/stage_f.py`
- `src/stage_g.py`
- `src/stage_h.py`
- `src/stage_i_vllm.py`
- `src/stage_j_block0.py`
- `src/stage_k_release.py`

### 第二层：新的模块化封装入口

本轮整理后，已经开始把最稳定、最跨阶段复用的能力抽到 `src/aloepri/` 下：

- `src/aloepri/token_ops.py`
- `src/aloepri/pipelines/stage_a.py`
- `src/aloepri/pipelines/standard_shape.py`
- `src/aloepri/pipelines/release.py`
- `src/aloepri/catalog.py`

这意味着：

> 当前仓库进入了“**阶段脚本保留 + 模块化封装逐步接管**”的状态。

### 1.1 新增的架构适配层

本轮整理还新增了一层明确的 **Qwen architecture adapter**：

- `src/aloepri/adapters/qwen.py`

它的作用不是绑定某个单一 checkpoint，而是通过**结构契约**来判断模型是否可被当前仓库支持：

- `model.embed_tokens`
- `model.layers`
- `model.norm`
- `lm_head`
- layer 内部具备：
  - `self_attn.q_proj / k_proj / v_proj / o_proj`
  - `input_layernorm / post_attention_layernorm`
  - `mlp.gate_proj / up_proj / down_proj`

也就是说，当前模块化层的正确表述是：

> **支持所有满足 Qwen decoder 结构契约的模型，而不是只支持一个固定大小的 Qwen2.5-0.5B。**

当前已实测验证的模型仍是：

- `Qwen2.5-0.5B-Instruct`

但从代码结构上讲，hidden size、层数、head 数、KV head 数都已经不再硬编码在模块接口里。

---

## 2. 每一阶段的能力抽象

### 阶段 A：词表空间闭环

原始实现重点：

- `perm_vocab / inv_perm_vocab`
- embedding/head 词表置换
- 输入 token 映射
- 输出 logits 恢复

legacy 入口：

- `src/key_manager.py`
- `src/transforms.py`
- `src/obfuscate_embed_head.py`
- `src/stage_b.py` 中的 `prepare_stage_a_model(...)`

新的模块化抽象：

- `src/aloepri/token_ops.py`
  - `build_vocab_keys(...)`
  - `obfuscate_input_ids(...)`
  - `restore_output_logits(...)`
  - `restore_output_ids(...)`
- `src/aloepri/pipelines/stage_a.py`
  - `build_stage_a_bundle(...)`
  - `export_stage_a_standard_checkpoint(...)`
  - `load_stage_a_standard_checkpoint(...)`

一句话概括：

> 阶段 A 已经被抽象成一套明确的 **token-space obfuscation pipeline**。

---

### 阶段 B/C/D：从 block0 到多层 block

原始实现重点：

- hidden transform
- block0 attention 子层恢复
- block0 full block 恢复
- 多层传播验证

legacy 入口：

- `src/hidden_keys.py`
- `src/obfuscate_rmsnorm.py`
- `src/obfuscate_ffn.py`
- `src/stage_b.py`
- `src/stage_c.py`
- `src/stage_d.py`

当前模块化状态：

- 仍主要保留为 legacy stage pipeline
- 还没有完全抽成独立的 stage-neutral builder

整理结论：

> 这些阶段当前仍是“实验级核心逻辑”，还不是最终统一模块接口；后续如果继续模块化，应把它们收敛为 `hidden_basic / block_builder / trace_hooks` 三组能力。

---

### 阶段 E：复杂 attention

原始实现重点：

- `R̂_qk`
- `Ĥ_qk`
- `Ẑ_block`
- `τ_kv`
- `τ_group`

legacy 入口：

- `src/attention_keys.py`
- `src/gqa_layout.py`
- `src/obfuscate_attention_complex.py`
- `src/stage_e.py`

当前模块化状态：

- 已部分映射到：
  - `src/aloepri/layers/attention.py`
- 但仍明显依赖阶段 E / H 的具体实现形态

整理结论：

> attention 复杂结构已经有“模块入口”，但仍属于 **阶段特化封装**，还不算完全 stage-neutral。

---

### 阶段 F/G/H：KeyMat 路线

原始实现重点：

- Algorithm 1 / KeyMat
- bridge 接入
- fused / de-bridge
- attention 静态化
- 噪声定标
- stage-H 工件导出

legacy 入口：

- `src/keymat.py`
- `src/keymat_embed_head.py`
- `src/keymat_norm.py`
- `src/keymat_ffn.py`
- `src/keymat_attention_bridge.py`
- `src/stage_f.py`
- `src/stage_g*.py`
- `src/stage_h*.py`

现有模块化封装：

- `src/aloepri/layers/embeddings.py`
- `src/aloepri/layers/norm.py`
- `src/aloepri/layers/ffn.py`
- `src/aloepri/layers/attention.py`
- `src/aloepri/engine.py`

当前问题：

- 这些封装仍偏向 KeyMat 单一路线
- `engine.py` 更像阶段 F–H 的聚合器，不足以覆盖 A–K 全部阶段
- `keys.py` 仍带有“后续再补全”的逻辑，不适合作为最终统一 keys 层

整理结论：

> `src/aloepri/` 当前最成熟的，是对 **F–H KeyMat 路线的模块化尝试**；它可以保留，但不能直接代表整个仓库的统一模块化层。

---

### 阶段 I：标准 HF/vLLM 入口

原始实现重点：

- 标准 HF checkpoint 导出
- client/server secret 契约
- HF correctness 回归
- vLLM 回归入口
- Phase 2 阻塞定位

legacy 入口：

- `src/stage_i_vllm.py`
- `scripts/export_stage_i_vllm_checkpoint.py`
- `scripts/run_stage_i_hf_regression.py`
- `scripts/run_stage_i_vllm_regression.py`

新的模块化抽象：

- `src/aloepri/pipelines/stage_a.py`

整理结论：

> 阶段 I 已经被很好地收敛成“**标准导出入口**”模块。

---

### 阶段 J：standard-shape 路线

原始实现重点：

- non-expanding square monomial transform
- `prefix-1 / 2 / 4 / 8 / full-layer`
- full-layer 标准 checkpoint 导出
- standard-shape 噪声定标

legacy 入口：

- `src/stage_i_square.py`
- `src/stage_j_block0.py`
- `src/stage_j_noise.py`

新的模块化抽象：

- `src/aloepri/pipelines/standard_shape.py`
  - `build_standard_shape_full_bundle(...)`
  - `export_standard_shape_full_checkpoint(...)`

整理结论：

> 阶段 J 已经形成了一条清晰的 **standard-shape pipeline**，并且是当前最适合作为“标准部署版本”的模块主线。

---

### 阶段 K：统一交付包装

原始实现重点：

- release catalog
- deployment contract
- profile 化工件选择
- 统一 inference CLI

legacy 入口：

- `src/stage_k_release.py`
- `scripts/export_stage_k_release.py`
- `scripts/infer_stage_k_release.py`

新的模块化抽象：

- `src/aloepri/pipelines/release.py`

整理结论：

> 阶段 K 已经被抽象成一套明确的 **release pipeline**。

---

## 3. 当前 `src/aloepri/` 的正确定位

目前 `src/aloepri/` 不能简单理解成“最终统一引擎”，更准确的说法是：

> 它现在是一个**模块化收口层的起点**。

### 已经适合作为公共入口的部分

- `token_ops.py`
- `pipelines/stage_a.py`
- `pipelines/standard_shape.py`
- `pipelines/release.py`
- `catalog.py`

### 暂时仍应视为“过渡态”的部分

- `engine.py`
- `keys.py`
- `layers/*.py`

原因是：

- 它们强依赖 KeyMat 路线
- 还没有把 standard-shape 路线自然纳入统一抽象
- 目前更像 F–H 的模块化实验，而不是 A–K 的总接口

---

## 4. 这轮整理新增了什么

### 新增的模块化代码入口

- `src/aloepri/adapters/qwen.py`
- `src/aloepri/token_ops.py`
- `src/aloepri/pipelines/stage_a.py`
- `src/aloepri/pipelines/standard_shape.py`
- `src/aloepri/pipelines/release.py`
- `src/aloepri/catalog.py`

### 新增的公共导出

- `src/aloepri/__init__.py`
  - 现在已经导出：
    - Qwen architecture adapter
    - stage catalog
    - stage A pipeline
    - standard-shape pipeline
    - release pipeline
    - token ops

### 新增的机器可读清单

- `scripts/export_aloepri_stage_catalog.py`
  - 导出：
    - `outputs/aloepri_stage_catalog.json`

这份 JSON 可以直接回答：

- A–K 每个阶段的目标
- 对应的 legacy 脚本
- 对应的核心模块
- 当前模块化入口
- 对应工件和文档

---

## 5. 当前推荐的使用方式

如果是**看实验史 / 看复现链路**：

- 优先读：
  - `docs/完整复现总报告_阶段A-K.md`
  - 各阶段独立报告
- 优先跑：
  - 现有 `scripts/run_stage_*.py`

如果是**做新开发 / 复用能力**：

- token 侧能力：
  - `src/aloepri/token_ops.py`
- Stage A 标准导出：
  - `src/aloepri/pipelines/stage_a.py`
- standard-shape full-layer：
  - `src/aloepri/pipelines/standard_shape.py`
- release/catalog：
  - `src/aloepri/pipelines/release.py`
  - `src/aloepri/catalog.py`

也就是说：

> 旧脚本依然是“最完整的阶段复现实验入口”；  
> 新模块则是“最值得继续扩展的统一接口层”。  

---

## 6. 后续如果继续整理，最合理的顺序

如果继续做下一轮模块化整理，建议顺序是：

1. **先统一 evaluator / artifact schema**
   - 让阶段脚本输出 schema 更一致
2. **再统一 standard-shape line**
   - 当前这是最接近部署的主线
3. **最后再回头收口 KeyMat line**
   - 把 `engine.py / keys.py / layers/*.py` 真正改成双路线兼容

不建议先强行把所有 stage 脚本都并进一个 super-engine，
因为当前仓库里其实已经存在两条不同路线：

- KeyMat / fused / staticized
- standard-shape / full-layer / release

更现实的做法是：

> **先承认两条主线并存，再在公共能力层逐步收口。**

---

## 7. 一句话结论

> 当前仓库已经完成了从“按阶段堆叠的研究复现”向“模块化能力层 + 阶段脚本基线并存”的第一次整理；其中 Stage A、Stage I、Stage J、Stage K 已经有了比较清晰的模块化入口，而 F–H 的 `src/aloepri/` 仍处于向统一抽象层收口的过渡状态。**
