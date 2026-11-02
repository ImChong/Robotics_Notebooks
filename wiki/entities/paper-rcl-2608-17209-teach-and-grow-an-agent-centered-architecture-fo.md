---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2608.17209"
summary: "We present Teach-and-Grow Learning (TGL), an agent-centered architecture for general robot learning. Our LIBERO evaluation attains state-of-the-art performance; controlled studies expose skill induction, persistent reuse"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2608_17209_teach-and-grow-an-agent-centered-archite.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Teach and Grow

**Teach and Grow: An Agent-Centered Architecture for General Robot Learning** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 214/564** 篇，分组 **Related resources**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

We present Teach-and-Grow Learning (TGL), an agent-centered architecture for general robot learning. Our LIBERO evaluation attains state-of-the-art performance; controlled studies expose skill induction, persistent reuse, and agent-directed adaptation.

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- We present Teach-and-Grow Learning (TGL), an agent-centered architecture for general robot learning. Our LIBERO evaluation attains state-of-the-art performance; controlled studies expose skill induction, persistent reuse, and agent-directed adaptation.
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    D["少量成功示教 Dn<br/>示教是加速器而非前提"] --> AG
    TASK["任务指令 + 当前观测"] --> AG
    subgraph STATE["持久状态（模型权重 θ 固定，不做梯度更新）"]
        LIB["Skill Library B<br/>已验证的 Skill Block"]
        MEM["Experience Memory M<br/>上下文、结果、诊断、修复"]
    end
    LIB --> AG
    MEM --> AG
    AG["AI Agent<br/>理解任务、检索候选 block、组成工作计划"] --> ABS["跨示教抽象<br/>按任务状态变化切段并对齐<br/>合成共享策略 ρ"]
    ABS --> BLK["新 Skill Block<br/>目标 g / 范围 S / 策略 ρ / 接地 γ / 执行器 Π / 检验 v / 恢复 R"]
    AG --> SEL["选择 block 与执行器 πi"]
    SEL --> GR["在当前场景接地<br/>物体位姿、抓取、路径"]
    GR --> EX["执行器闭环执行<br/>VLA / 规划器 / 伺服 / 触觉策略等"]
    EX --> VER["执行后观测 + 执行器证据<br/>通过 / 失败 / 不确定"]
    VER -->|"符合预期，继续下一块"| AG
    VER -->|"不符：重看、换执行器或路线、请求示教"| AG
    BLK --> VAL["准入校验<br/>范围、执行器兼容、结果检验、恢复、来源"]
    VAL -->|"通过"| LIB
    VER -->|"结果、失败、修复"| MEM
```

图依据：论文 §III（Skill Block、Experience Memory）、§IV（Agentic 架构与组合、Few-Shot Teaching、Skill Block 在当前场景执行）/ Fig.2、Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 214/564 |
| 分组 | Related resources |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2608.17209> |
| 子类 / 象限 | 理论与规划 · 综述与技术资源 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

We present Teach-and-Grow Learning (TGL), an agent-centered architecture for general robot learning. Our LIBERO evaluation attains state-of-the-art performance; controlled studies expose skill induction, persistent reuse, and agent-directed adaptation.

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2608.17209) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Related resources** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Teach and Grow」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Related resources**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2608_17209_teach-and-grow-an-agent-centered-archite.md`](../../sources/papers/rcl_awesome_wam_2608_17209_teach-and-grow-an-agent-centered-archite.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2608.17209>
- [arXiv PDF](https://arxiv.org/pdf/2608.17209)：已通读方法部分（实验与附录以略读为准）：§III（A-D）、§IV（A-E）与 §V 开头的实验设定

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2608.17209)
