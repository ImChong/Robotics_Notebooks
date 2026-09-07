---
type: entity
tags: ['paper', 'humanoid', 'loco-manipulation', 'behavior-tree', 'affordance-template', 'runtime-editing']
status: complete
updated: 2026-09-07
arxiv: "2609.01518"
summary: "IHMC/UWF（arXiv:2609.01518）：机载 Affordance Template + 行为树 + 可运行时编辑场景动作；全身控制并发行走与操作；H1-2/Alex 推门 34 s、六球分拣 45 s（扰动）；未见官方代码。"
related:
  - ../tasks/loco-manipulation.md
  - ../concepts/whole-body-control.md
  - ./paper-bridge-humanoid.md
  - ../concepts/behavior-tree-vla-orchestration.md
sources:
  - ../../sources/papers/ihmc_fast_resilient_locom_manipulation_arxiv_2609_01518.md
---

# IHMC 快速抗扰可编辑人形 loco-manipulation 系统

**IHMC Fast Resilient Loco-Manipulation System**（[arXiv:2609.01518](https://arxiv.org/abs/2609.01518)）由 **佛罗里达人机认知研究所（IHMC）、西佛罗里达大学（UWF）** 提出（公众号周更 ingest 见 [策展索引](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)）。

## 一句话定义

人形 loco-manipulation 的速度与韧性，可以来自 **行为架构本身**——而不只是更大的策略网络。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AT | Affordance Template | 目标中心可操作区域模板 |
| BT | Behavior Tree | 行为树组织逻辑与子树复用 |
| WBC | Whole-Body Control | 并发身体运动与行走的低层执行 |

## 为什么重要

工业/应急场景需要 **小时级** 改行为而非 **天级** 重训；IHMC 把感知、接触、行走与操作员 UI 绑成 **机载可编辑** 闭环。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 佛罗里达人机认知研究所（IHMC）、西佛罗里达大学（UWF） |
| **开源** | 见 [工程实践](#工程实践) |

## 核心原理

系统组合 **object-centric Affordance Templates**、**行为树** 与 **behavior scene / primitive scene actions**：操作员 UI 与机器人状态持续同步，可在运行时修补逻辑与感知。动作原语经 **全身控制器** 并发执行行走与上身运动。

### 流程总览

```mermaid
flowchart LR
  ui[操作员 UI] --> bt[行为树]
  bt --> at[Affordance Templates]
  at --> scene[场景动作/感知]
  scene --> wbc[全身控制器]
  wbc --> robot[H1-2 / Alex]
```

## 源码运行时序图

**不适用** — 截至 **2026-09-07** 无可运行官方代码（或本文为硬件/协议类工作）。

## 工程实践

| 项 | 说明 |
|----|------|
| 开源状态 | 见论文摘录与项目页核查结论 |
| 复现入口 | 以 arXiv 为准 |

## 实验与评测

| 任务 | 结果 |
|------|------|
| 推门穿越 | **34 s**（Alex） |
| 六球按色分拣 | **45 s**（人类扰动下） |
| 九球双桌 | **2 min 8 s** |
| 行为创作 | 专家数小时内从零或改编行为 |

## 结论

**架构即能力：** 可运行时编辑的行为树 + 模板化感知，让人形 loco-manipulation 在真机上达到与近期 RL 门策略可比的 **速度**，并保留 **扰动韧性** 与 **快速任务变体** 能力。

1. 推门时间与文献中学习型人形门策略 **同量级竞争**。
2. 六类任务变体覆盖门、分拣、双桌等；强调 **operator-in-the-loop** 修复。
3. Alex：29 DoF 全电驱 + 头部双目 + PSYONIC 手，**无外部跟踪**。
4. 贡献在 **runtime 结构**，非新学习算法。
5. 未见开源栈——复现依赖 IHMC 内部系统。

## 局限与风险

未与 MoveIt/BehaviorTree.CPP 等做实验对照；数字来自自报演示与文献表。

## 关联页面

- [loco-manipulation](../tasks/loco-manipulation.md)
- [whole-body-control](../concepts/whole-body-control.md)
- [paper-bridge-humanoid.md](./paper-bridge-humanoid.md)
- [行为树 × VLA 编排](../concepts/behavior-tree-vla-orchestration.md)

## 参考来源

- [ihmc_fast_resilient_locom_manipulation_arxiv_2609_01518.md](../../sources/papers/ihmc_fast_resilient_locom_manipulation_arxiv_2609_01518.md)
- [公众号周更策展](../../sources/blogs/wechat_shenlan_weekly_papers_2026-09-04.md)

## 推荐继续阅读

- [https://arxiv.org/abs/2609.01518](https://arxiv.org/abs/2609.01518)
