# 阶段 J：Qwen 全模型部署物化报告

## 1. 新阶段目标

新版 `Stage J` 的目标是把经过 `Stage H` 表达重构、`Stage I` 约束验证之后的部署线，真正落成一个 **Qwen deployable artifact**。

当前这一步先采用一个务实策略：

- 以 `artifacts/stage_h_pretrained` 作为 bootstrap source
- 导出新的 `artifacts/stage_j_qwen_redesign`
- 在新 artifact 上显式写出 lineage 和 legacy reference

## 2. 为什么这样做

这一步的重点不是立刻发明全新的 full-model 数学，而是先把新版 `H -> I -> J` 关系从仓库结构上立起来：

- `Stage H` 提供 deployable expression inventory
- `Stage I` 提供 deployability matrix
- `Stage J` 开始提供新的 deployable artifact lineage

## 3. 与 legacy Stage J 的关系

旧版 `Stage J` 仍然是重要 baseline：

- `artifacts/stage_j_full_square`
- `artifacts/stage_j_full_square_tiny_a`

但它对应的是 conservative standard-shape line，而不是新版 `Stage J` 的唯一目标定义。

## 4. 标准权重证明

当前新版 `Stage J` 已经补上标准权重证明输出：

- `outputs/stage_j/standard_weight_proof.json`
- `artifacts/stage_j_qwen_redesign/manifest.json -> standard_weight_proof`

当前结论是：

- `is_standard_weight_export = false`
- `layout = buffered_stage_style`

这意味着新版 `Stage J` 已经有：

- component-expression manifest
- 真实可运行的 bootstrap artifact

但它还没有达到：

- 标准 `model.* / lm_head.*` 权重键布局可见

所以当前更准确的说法是：

> 新版 `Stage J` 已经完成“部署线物化”的第一步，但尚未完成“标准权重可见性证明”的最后一步。 

## 5. 双轨状态

当前 `Stage J` 已明确拆成两条线：

### 5.1 buffered redesign line

- `artifacts/stage_j_qwen_redesign`
- 更贴近当前 redesign 表达
- 当前在 `VMA / ISA hidden_state` 上已经明显优于旧 conservative line

### 5.2 standard-visible bridge line

- `artifacts/stage_j_qwen_redesign_standard`
- 当前从 buffered redesign source 做最小标准可见 materialization
- 已经可以被标准 Hugging Face 加载器识别和加载
- 明确不宣称已与 buffered redesign line 等价

当前 bridge regression 也已经明确显示：

- `generated_ids_exact_match_rate = 0.0`
- restored logits 误差仍然很大

所以这条线现在更准确的定位是：

> **可加载的标准可见桥接线，而不是已等价的 redesign 导出线。**

这意味着：

> 当前 `Stage J` 的完善方向不再是只做单条线，而是先让“高保真 redesign”与“标准键可见导出”两条线都存在，并把它们的差距写清楚。 
