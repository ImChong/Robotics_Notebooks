---
type: entity
tags: [paper, humanoid, hardware, perception, duke]
status: complete
updated: 2026-09-14
arxiv: "2609.08905"
related:
  - ../tasks/manipulation.md
  - ../tasks/loco-manipulation.md
  - ./paper-humanoid-leg-generative-design-dynamics.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/papers/visible_reachable_workspace_humanoid_arxiv_2609_08905.md
summary: "Visible-Reachable Workspace（arXiv:2609.08905）：visible-reachable workspace metric; Duke Humanoid V2 31-DoF dual RGB-D; coverage 38%→97%; less task time and energy；截至入库日未见官方代码。"
---

# Visible-Reachable Workspace（arXiv:2609.08905）

**Visible-Reachable Workspace**（*Visible-Reachable Workspace for Perception-Aware Humanoid Design*，[arXiv:2609.08905](https://arxiv.org/abs/2609.08905)）由 **杜克大学（Duke University）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)）。

## 一句话定义

面向感知的人形机器人可见—可达工作空间设计 — visible-reachable workspace metric。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VRW | Visible-Reachable Workspace | 可见可达工作空间 |
| RGB-D | RGB + Depth | 彩色深度相机 |
| DoF | Degrees of Freedom | 自由度 |

## 为什么重要

操作任务常需边看边够；只优化可达域会忽视自遮挡与感知盲区。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 杜克大学（Duke University） |
| **开源** | **未见/待发布**（步骤 2.5 核查：截至 2026-09-14 无可运行官方仓库） |

## 核心原理

定义 VRW 度量联合头/眼与手臂构型；硬件 co-design 传感器位姿与关节链；V2 相对 V1 覆盖率大幅提升。

### 流程总览

```mermaid
flowchart LR
  metric[VRW 度量] --> design[构型优化]
  design --> v2[Duke Humanoid V2]
  v2 --> dual[双 RGB-D]
  dual --> tasks[操作任务评测]
```

## 源码运行时序图

**不适用** — 本文为理论/硬件/系统/数据类工作，arXiv 未提供可运行训练或部署仓库。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 未见官方仓库；以 arXiv 为准 |
| 复现入口 | 论文方法与超参；代码发布后再补 `sources/repos/` |
| 部署注意 | 设计阶段用 VRW 热力图选相机位；31-DoF 与双深度需权衡重量。 |

## 实验与评测

覆盖率、任务时间、能耗；多 household 操作任务。

## 结论

VRW 指标驱动的 co-design 让人形在操作任务上同时获得感知与可达性。

1. 38%→97% 覆盖率是核心量化收益。
2. 双 RGB-D 减少盲区。
3. 任务时间与能耗同步下降。
4. 硬件设计应感知先行。
5. 31-DoF 是工程折中。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 仅可达工作空间优化 | 忽视可见性 |
| 固定头胸相机布局 | 未系统 co-design |
| [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) | 该闭环从 ①传感层 往下选算法；本页把问题往上推一层——**相机装在哪、关节链怎么排**，决定了后面各层能看到什么。感知栈选型的前置约束，不是它的替代 |

## 局限与风险

动态行走中 VRW 变化未建模；仅 Duke 平台验证。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [loco-manipulation](../tasks/loco-manipulation.md)
- [./paper-humanoid-leg-generative-design-dynamics.md](./paper-humanoid-leg-generative-design-dynamics.md)

## 参考来源

- [visible_reachable_workspace_humanoid_arxiv_2609_08905.md](../../sources/papers/visible_reachable_workspace_humanoid_arxiv_2609_08905.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.08905](https://arxiv.org/abs/2609.08905)
