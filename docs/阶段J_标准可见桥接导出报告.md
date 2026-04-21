# 阶段 J：标准可见桥接导出报告

## 1. 为什么要有这条线

当前 redesigned `Stage J` 有两类目标：

- buffered redesign line
  - 更接近论文允许保留的复杂表达
  - 已经在 `VMA / ISA hidden_state` 上明显优于旧 conservative line
- standard-visible line
  - 需要真正提供标准 `model.* / lm_head.*` 键布局

由于这两者当前还没有完全合流，所以本轮新增一条明确的桥接线：

- `artifacts/stage_j_qwen_redesign_standard`

## 2. 当前桥接线的真实定位

当前这条桥接线：

- 以 `artifacts/stage_j_qwen_redesign` 作为 source
- 对其中 buffered stage-style 权重做最小标准可见 materialization
- 明确标记为 `standard_visible_bridge`
- 明确标记：
  - `equivalence_to_buffered_redesign = false`

所以它不是“已经把 redesign 线标准化成功”的结论，而是：

> 先把标准键可见导出这条工程通路单独立起来，同时诚实保留与 buffered redesign 线之间尚未证明的差距。

## 3. 当前产物

- `artifacts/stage_j_qwen_redesign_standard/manifest.json`
- `manifest.standard_weight_proof`

当前 bridge 的 materialization 策略是：

- `embed / lm_head / qkv / o / gate / up / down`
  - 直接从 buffered redesign 权重映射到标准键
- `config`
  - 按 buffered 权重形状改写 `hidden_size / head_dim`
- `norm`
  - 先用占位的标准可见 materialization，尚不宣称与 buffered redesign 等价

## 4. 当前已验证的能力

当前 `artifacts/stage_j_qwen_redesign_standard/server` 已经通过两层验证：

1. `standard_weight_proof`
   - `is_standard_weight_export = true`
   - `layout = standard_weight_visible`
2. 标准加载器 smoke
   - `AutoConfig.from_pretrained(...)` 可识别
   - `AutoModelForCausalLM.from_pretrained(...)` 可加载

这说明当前 bridge 线已经不只是“形式上写成标准键”，而是：

> **一个真实可被标准 Hugging Face 加载器加载的 Stage-J 标准可见导出物。**

## 5. 当前等价性结论

虽然这条 bridge 线已经可加载，但 `bridge_regression` 当前仍显示：

- `kappa_fused` 是当前最优 bridge norm 策略
- 在该策略下：
  - `avg_restored_full_logits_max_abs_error ≈ 28.41`
  - `generated_ids_exact_match_rate = 0.2`

因此当前它的真实状态是：

> **标准可见且可加载，并且 `kappa_fused` 明显优于其他 norm 策略；但它仍然明显不等价于 buffered redesign line。**

## 6. 下一步

后续如果要把两条线真正合流，目标应是：

1. 把 buffered redesign line 中已经证明有效的表达，继续 materialize 到标准键布局中
2. 最终删除 `equivalence_to_buffered_redesign = false` 这类桥接说明
