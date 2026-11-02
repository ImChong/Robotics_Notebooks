---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-28
arxiv: "2401.16013"
venue: "ICRA 2024"
code: https://github.com/rail-berkeley/serl
summary: "RCL Awesome WAM 清单收录（Foundational work）；细节以原文 PDF / 项目页为准。"
related:
  - ../comparisons/self-improving-agents-embodied-lifelong-learning.md
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2401_16013_serl-a-software-suite-for-sample-efficie.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# SERL

**SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 179/564** 篇，分组 **Foundational work**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

RCL Awesome WAM 清单收录（Foundational work）；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- RCL Awesome WAM 清单收录（Foundational work）；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    PD["先验数据：演示或次优数据"] --> RB["RLPD 采样：每个 batch 一半来自先验数据、一半来自在线 replay buffer"]
    subgraph LN["Learner 节点（§4.1 RLPD，基于 SAC 的 off-policy actor-critic）"]
        RB --> UP["高 UTD 比率更新 Q 函数与 actor，critic 加 layer norm"]
    end
    subgraph AC["Actor 节点（与 learner 并行）"]
        PO["策略 π(a|s) 选择动作"]
    end
    UP --> PO
    subgraph ENV["机器人环境（Gym 风格 adapter，§4.4）"]
        HC["策略以 10Hz 给出目标"] --> IC["阻抗控制器 1kHz 跟踪<br/>限制参考位姿误差 |e| ≤ Δ（§4.5）"]
        IC --> RO["Franka 机器人执行并返回观测"]
    end
    PO --> HC
    RO --> RW
    subgraph RWS["奖励 §4.2"]
        RW["二值分类器 r = log p(e|s) / VICE / 手写奖励"]
    end
    RW --> RB
    RO -.-> RS
    subgraph RSS["免复位训练 §4.3"]
        RS["前向策略执行任务 + 后向策略回到初始状态<br/>两个独立 RL agent，各有自己的 Q 函数与奖励"]
    end
```

图依据：论文 §4.1–§4.5 / Fig.2 / Fig.4

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 179/564 |
| 分组 | Foundational work |
| 出处 | ICRA 2024 |
| 论文 | <https://ieeexplore.ieee.org/document/10610040/> |
| 代码/项目 | <https://github.com/rail-berkeley/serl> |
| 子类 / 象限 | 动作策略基础 · 综述与技术资源 · 不适用 |

## 核心机制（归纳）

### 策展导读要点

RCL Awesome WAM 清单收录（Foundational work）；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://ieeexplore.ieee.org/document/10610040/) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **Foundational work** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「SERL」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Foundational work**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- [自我改进智能体与具身终身学习对比](../comparisons/self-improving-agents-embodied-lifelong-learning.md) — SERL 在其中代表「真机在线 RL 自我改进」的软件栈：off-policy RL、奖励、复位与控制组件
- 列表实体：[Awesome World-Action Models（RCL）](paper-rcl-wam-robot-learning-control-survey.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2401_16013_serl-a-software-suite-for-sample-efficie.md`](../../sources/papers/rcl_awesome_wam_2401_16013_serl-a-software-suite-for-sample-efficie.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://ieeexplore.ieee.org/document/10610040/>
- [arXiv PDF](https://arxiv.org/pdf/2401.16013)：已通读方法部分（实验与附录以略读为准）：§4 方法（4.1 RLPD、4.2 奖励、4.3 免复位、4.4 软件组件、4.5 阻抗控制器）、Fig.2/4

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://ieeexplore.ieee.org/document/10610040/)
