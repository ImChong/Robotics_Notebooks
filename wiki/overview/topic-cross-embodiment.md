---
type: overview
tags: [topic, topic-cross-embodiment, transfer, any2any, retargeting]
status: complete
updated: 2026-07-06
related:
  - ../queries/cross-embodiment-transfer-strategy.md
  - ../entities/paper-any2any-cross-embodiment-wbt.md
  - ../entities/paper-last-hd-latent-physical-reasoning.md
summary: "跨具身迁移专题汇总：不同机器人形态、仿真与真机之间的策略/动作迁移，重定向、域随机与 Any2Any 类方法的选型与失败模式。"
---

# 跨具身迁移（专题汇总）

> **图谱专题视图**：本页是知识图谱「🔀 跨具身迁移 (Cross-Embodiment)」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=cross-embodiment) 筛选时，本节点为汇总锚点。

## 一句话定义

**跨具身迁移** 研究如何把在 **某一机器人形态、仿真环境或数据源** 上学到的技能，迁移到 **不同骨架、尺寸或硬件平台**，而不完全重训。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Embodiment | Robot Embodiment | 机器人形态/硬件实例 |
| Transfer | Cross-Embodiment Transfer | 跨形态技能迁移 |
| Retargeting | Motion Retargeting | 跨骨架动作映射（迁移前置） |
| DR | Domain Randomization | 扩宽训练分布以提升迁移 |
| OXE | Open X-Embodiment | 跨具身开源数据倡议 |

## 为什么重要

- **数据与硬件碎片化**：不可能每个形态都从零采集全套示范。
- **Sim2Real 的姊妹问题**：不仅是 sim→real，还有 human→robot、大→小人形。
- **WBT 与 VLA 共同痛点**：参考动作与策略接口需对齐目标机体。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| Query | 迁移策略决策树 | [Cross-Embodiment Transfer Strategy](../queries/cross-embodiment-transfer-strategy.md) |
| 概念 | 重定向与迁移 | [Motion Retargeting](../concepts/motion-retargeting.md) |
| 实体 | Any2Any WBT | [Any2Any Cross-Embodiment WBT](../entities/paper-any2any-cross-embodiment-wbt.md) |
| 实体 | LaST-HD 人手→机器人 VLA | [LaST-HD](../entities/paper-last-hd-latent-physical-reasoning.md) |
| 概念 | 角色动画 vs 机器人 | [Character Animation vs Robotics](../concepts/character-animation-vs-robotics.md) |
| 数据 | 跨具身数据集 | [Open X-Embodiment](../concepts/open-x-embodiment.md) |

## 与其他专题的关系

- **[动作重定向](./topic-motion-retargeting.md)**：跨骨架动作对齐。
- **[WBT](./topic-wbt.md)**：跟踪策略的跨形态扩展。
- **[Sim2Real](./topic-sim2real.md)**：仿真-真机是跨具身特例。

## 关联页面

- [Sim2Real](../concepts/sim2real.md)
- [Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md)
- [Domain Randomization](../concepts/domain-randomization.md)

## 参考来源

- 本库归纳自 [Cross-Embodiment Transfer Strategy](../queries/cross-embodiment-transfer-strategy.md) 及 motion-retargeting / sim2real 交叉页
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`cross-embodiment` 命中规则）
