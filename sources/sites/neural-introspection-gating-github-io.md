# zjw4321.github.io/neural-introspection-gating-page（Neural Introspection Gating 项目页）

- **标题：** Neural Introspection Gating for Adaptive KV-Cache Reuse in Vision-Language-Action Models
- **类型：** site / project-page
- **URL：** <https://zjw4321.github.io/neural-introspection-gating-page/>
- **配套论文：** [arXiv:2608.10824](https://arxiv.org/abs/2608.10824) — [`sources/papers/neural_introspection_gating_arxiv_2608_10824.md`](../papers/neural_introspection_gating_arxiv_2608_10824.md)
- **代码：** <https://github.com/zjw4321/gated-vla-cache> — [`sources/repos/gated_vla_cache.md`](../repos/gated_vla_cache.md)（已开源，Apache 2.0）
- **入库日期：** 2026-08-13
- **通读：** 已完成
- **核对范围：** `sites/nig.html` 全文（L1–461，含 Meta / JSON-LD 元数据、`zjw4321/gated-vla-cache` GitHub 链接、四阶段 Method 卡片与流程图、OpenVLA / OpenVLA-OFT LIBERO 对照表及 BibTeX）

## 一句话摘要

东京大学 JSK Robotics Laboratory 项目页：展示用 logit-margin 内省门控扩展 VLA-Cache；链接官方仓库 `zjw4321/gated-vla-cache` 并给出 OpenVLA / OpenVLA-OFT 在 LIBERO 四套件上的成功率–TFLOPs 完整对照表。

## 公开信息要点（核实更新）

- **页首：** LIBERO-Long **54.8%**；算力 **1.54 TFLOPs**；Accepted at IROS 2026；发布日期（JSON-LD）`2026-07-09`。
- **方法四阶段：** static patch detection → task-relevant filtering → entropy-adaptive reuse → logit-margin gate（零额外推理开销提取自前向已生成的 logits）。
- **开源：** GitHub 按钮已启用并指向 <https://github.com/zjw4321/gated-vla-cache> → **已开源**。

## 为何值得保留

- 记录了 `Accepted at IROS 2026`、官方 GitHub 入口与 JSON-LD 元数据。
- 提供 OpenVLA 与 OpenVLA-OFT 双模型 LIBERO 四套件「观测启发式 vs 模型内置信」完整对照数字。

## 关联资料

- 论文：[`sources/papers/neural_introspection_gating_arxiv_2608_10824.md`](../papers/neural_introspection_gating_arxiv_2608_10824.md)
- 仓库：[`sources/repos/gated_vla_cache.md`](../repos/gated_vla_cache.md)
- Wiki：[wiki/entities/paper-neural-introspection-gating.md](../../wiki/entities/paper-neural-introspection-gating.md)
