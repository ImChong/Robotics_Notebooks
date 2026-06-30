---
type: overview
tags: [topic, topic-sim2real, deployment, domain-randomization, transfer]
status: complete
updated: 2026-06-30
summary: "Sim2Real 专题汇总：图谱专题锚点；详细知识见 concepts/sim2real，本页仅作专题导航。"
---

# Sim2Real（专题汇总）

> **主内容页**：[Sim2Real（概念总览）](../concepts/sim2real.md) — 方法、工程流程与交叉引用以该页为准；本页仅服务图谱专题筛选锚点。
>
> **图谱专题视图**：本页是知识图谱「🔁 仿真到现实 (Sim2Real)」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=sim2real) 筛选时，本节点为汇总锚点。

## 一句话定义

**Sim2Real（Simulation to Real）** 研究如何把 **仿真里训练好的策略** 稳定迁移到真实机器人，弥合动力学、感知、延迟与接触上的分布差距。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真到真机迁移 |
| DR | Domain Randomization | 随机化仿真参数扩宽训练分布 |
| SysID | System Identification | 辨识真机/仿真参数 |
| RMA | Rapid Motor Adaptation | 在线适应电机/动力学差异 |
| PITL | Processor-in-the-Loop | 控制环硬件在环仿真 |

## 为什么重要

- **机器人学习默认在仿真里做**：真机采样贵且危险。
- **「能跑仿真 ≠ 能跑真机」**：差距常出现在执行器、接触与感知延迟。
- **与 WBT / Locomotion 全栈相关**：高动态技能对 Sim2Real 更敏感。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 概念 | Sim2Real 总览 | [Sim2Real](../concepts/sim2real.md) |
| 方法 | 域随机化 | [Domain Randomization](../concepts/domain-randomization.md) |
| 对比 | 迁移路线选型 | [Sim2Real Approaches](../comparisons/sim2real-approaches.md) |
| 对比 | 残差 vs Real2Sim vs 真机 RL | [Sim2Real vs Real2Sim Fine-Tuning](../comparisons/sim2real-vs-real2sim-fine-tuning.md) |
| 工程 | 部署清单 | [Sim2Real Checklist](../queries/sim2real-checklist.md)（含快速部署检查） |
| 工程 | 处理器在环 | [Processor-in-the-Loop Sim2Real](../concepts/processor-in-the-loop-sim2real.md) |

## 与其他专题的关系

- **[安全微调](./topic-safe-fine-tuning.md)**：Sim2Real 链路的最后一段在线适配。
- **[IL/RL](./topic-learning.md)**：训练范式与迁移策略 jointly 设计。
- **[通信协议](./topic-communication.md)**：真机延迟与仿真假设对齐。

## 关联页面

- [System Identification](../concepts/system-identification.md)
- [Privileged Training](../concepts/privileged-training.md)
- [Data Flywheel](../concepts/data-flywheel.md)

## 参考来源

- 本库归纳自 [Sim2Real](../concepts/sim2real.md) 及 comparisons/queries 迁移系列页
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`sim2real` 命中规则）
