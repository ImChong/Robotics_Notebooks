---
type: overview
tags: [topic, topic-wbt, whole-body-tracking, motion-tracking, humanoid]
status: complete
updated: 2026-06-17
summary: "全身运动跟踪（WBT）专题汇总：参考采集→重定向→跟踪训练→跨具身→真机部署的端到端流水线，对比 SONIC/BeyondMimic/SD-AMP/Heracles 等路线。"
---

# 全身运动跟踪 WBT（专题汇总）

> **图谱专题视图**：本页是知识图谱「🕺 WBT」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=wbt) 筛选时，本节点为汇总锚点。

## 一句话定义

**Whole-Body Tracking（WBT）** 让人形机器人 **全身** 跟踪一段参考动作（舞蹈、格斗、日常动作等），覆盖从参考数据到可部署跟踪策略的完整工程链。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBT | Whole-Body Tracking | 全身参考动作跟踪 |
| DeepMimic | DeepMimic | 早期深度跟踪模仿代表 |
| AMP | Adversarial Motion Prior | 运动先验约束 RL 风格 |
| SD-AMP | (Unified Walk-Run-Recovery) | 走-跑-恢复统一跟踪族 |
| Any2Any | Any-to-Any Transfer | 跨形态/跨动作跟踪迁移 |

## 为什么重要

- **人形「像身体」的核心指标**：任务奖励之外还要像参考 motion。
- **2024–2026 论文爆发区**：SONIC、BeyondMimic、Heracles 等形成可复现谱系。
- **与重定向 / Sim2Real 强耦合**：WBT 质量上限由参考与迁移共同决定。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 流水线 | 六阶段端到端 | [Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md) |
| 方法 | SONIC / BeyondMimic | [SONIC Motion Tracking](../methods/sonic-motion-tracking.md)、[BeyondMimic](../methods/beyondmimic.md) |
| 对比 | 主流 WBT 路线 | [SONIC vs BeyondMimic vs SD-AMP vs Heracles](../comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md) |
| Query | 方法选型 | [Humanoid Motion Tracking Method Selection](../queries/humanoid-motion-tracking-method-selection.md) |
| 栈 | 42 篇 RL 运动控制 | [Humanoid RL Motion Control Body System Stack](./humanoid-rl-motion-control-body-system-stack.md) |

## 与其他专题的关系

- **[动作重定向](./topic-motion-retargeting.md)**：WBT 上游参考来源。
- **[跨具身](./topic-cross-embodiment.md)**：Any2Any 等跨形态跟踪。
- **[IL/RL](./topic-learning.md)**：跟踪策略主流用 RL + 模仿奖励。

## 关联页面

- [Motion Retargeting](../concepts/motion-retargeting.md)
- [Behavior Foundation Model](../concepts/behavior-foundation-model.md)
- [Humanoid AMP Motion Prior Survey](./humanoid-amp-motion-prior-survey.md)

## 参考来源

- 本库归纳自 [Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md) 及 WBT 方法/对比页
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`wbt` 命中规则）
