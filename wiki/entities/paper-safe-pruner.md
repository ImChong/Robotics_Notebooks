---
type: entity
tags:
  - paper
  - vla
  - efficiency
  - token-pruning
  - tsinghua
status: complete
updated: 2026-09-29
arxiv: "2605.29662"
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md
sources:
  - ../../sources/papers/safe_pruner_arxiv_2605_29662.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md
summary: "SAFE-Pruner（arXiv:2605.29662 v3，清华/GigaAI）：future-aware token 剪枝，近 2× 加速。"
---

# SAFE-Pruner

**SAFE-Pruner: Semantic Attention-Guided Future-Aware Token Pruning for Efficient Vision-Language-Action Manipulation**（arXiv:[2605.29662](https://arxiv.org/abs/2605.29662)）— **清华大学、极佳视界（GigaAI）**。多模空间 [2026.08.17–08.23 周报](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md) 策展条目；细节以 arXiv 为准。

## 一句话定义

future-aware token 剪枝，近 2× 加速。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| RL | Reinforcement Learning | 强化学习后训练或微调 |
| TTA | Test-Time Augmentation / Adaptation | 测试时增强或适配 |
| SR | Success Rate | 任务成功率 |
| LIBERO | LIBERO Benchmark | 常见操作仿真基准套件 |

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 清华大学、极佳视界（GigaAI） |
| **评测** | LIBERO、SIMPLER；Astribot S1 实机 |
| **开源** | 待核实（截至 2026-09-29） |

## 为什么重要

- 纳入 [一周 VLA 趋势（2026.08.17 第一篇）](../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md) 横切面索引。
- 与 [VLA](../methods/vla.md) 方法页及同周其他 **16/16 独立 canonical 节点** 交叉对照。

## 评测与指标

- **设置：** LIBERO、SIMPLER 仿真及 Astribot S1 实机（见核心信息）；即插即用，不改模型权重。
- **主结果：** 最高 **1.89×** 加速，成功率降幅 **< 1.5%**，并比 SoTA 剪枝方法高最多 **1.9%**（数值摘自 arXiv 摘要，完整表格与基线设定以原文为准）。
- **机制：** 利用「语义注意力一致性」预测深层 token 显著性，避免浅层线索提前剪掉深层需要的 token；注意力转移时刷新参考时间步。

## 与其他工作对比

| 维度 | SAFE-Pruner | 对照 |
|------|-------------|------|
| 加速手段 | 视觉 token 剪枝（前瞻深层注意力） | [Shallow-π](./paper-shallow-pi.md)：层蒸馏 18→6，>2× 加速 |
| 是否训练 | 免训练、即插即用 | Shallow-π 需蒸馏训练 |
| 复用对象 | 丢弃冗余视觉 token | [Neural Introspection Gating](./paper-neural-introspection-gating.md)：按置信度门控 KV 缓存复用 |

## 结论

**SAFE-Pruner 在本库中作为 arXiv:2605.29662 的 canonical 详情节点；部署与复现前请对照原文 PDF/HTML 与项目页。**

1. **canonical 唯一性** — 全库仅此一页绑定 arXiv:2605.29662。
2. **读法** — 先读公众号策展摘要，再读 arXiv 方法与实验节。
3. **开源** — 待核实（截至 2026-09-29）。
4. **安全/评测类**（若适用）— 勿把任务成功率等同于安全或授权跟随。

## 关联页面

- [VLA](../methods/vla.md)
- [Manipulation](../tasks/manipulation.md)
- [一周 VLA 趋势地图（2026.08.17）](../overview/vla-weekly-trends-2026-08-17-part1-technology-map.md)

## 参考来源

- [safe_pruner_arxiv_2605_29662.md](../../sources/papers/safe_pruner_arxiv_2605_29662.md)
- [多模空间周报归档](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md)
- arXiv：<https://arxiv.org/abs/2605.29662>
- 项目页：<https://msssl.github.io/SAFE-Pruner>

## 推荐继续阅读

- [arXiv 摘要页](https://arxiv.org/abs/2605.29662)
