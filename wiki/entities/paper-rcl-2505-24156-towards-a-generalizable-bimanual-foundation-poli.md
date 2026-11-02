---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-10-06
arxiv: "2505.24156"
venue: "arXiv preprint"
summary: "CogRobot adapts video diffusion to bimanual control through three learned components: instruction-conditioned flow prediction, flow-conditioned RGB prediction, and a task-specific goal-reaching action policy. Flow offers"
related:
  - paper-rcl-wam-robot-learning-control-survey.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2505_24156_towards-a-generalizable-bimanual-foundat.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Towards a Generalizable Bimanual Foundation Policy via Flow-based Video Predi...

**Towards a Generalizable Bimanual Foundation Policy via Flow-based Video Prediction** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 499/564** 篇，分组 **WAMs**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

CogRobot adapts video diffusion to bimanual control through three learned components: instruction-conditioned flow prediction, flow-conditioned RGB prediction, and a task-specific goal-reaching action policy. Flow offers an intermediate description of motion without requiring action labels for the video models. The...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- CogRobot adapts video diffusion to bimanual control through three learned components: instruction-conditioned flow prediction, flow-conditioned RGB prediction, and a task-specific goal-reaching action policy. Flow offers an intermediate description of motion without requiring action labels for the video models. The...
- 想横向对照同一分组的其他工作，可以从 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome World-Action Models](paper-rcl-wam-robot-learning-control-survey.md) 与站内 WAM / VLA 方法页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["初始观测 o_0 + 任务指令 l"] --> B["阶段一 Text-to-Flow<br/>CogVideoX 微调, 生成光流序列 F_0:N"]
    B --> B1["光流转 flow video<br/>幅值映射饱和度, 方向映射色轮颜色<br/>规则转换, 无需额外训练 flow VAE"]
    B1 --> C["阶段二 Flow-to-Video<br/>flow 潜变量与带噪视频潜变量<br/>在通道维拼接后输入 CogVideoX"]
    A --> C
    C --> D["预测双臂轨迹视频<br/>每帧视为机器人要到达的目标观测"]
    D --> E["Goal-reaching Diffusion Policy<br/>π(a_0:n | o_0, o_n)"]
    E --> F["可执行的低层双臂动作"]
    G["训练细节: 先在 RDT + RoboMIND 混合数据上微调<br/>再在下游仿真/真机演示上微调<br/>真值光流由 FlowFormer++ 提取"] -.-> B
```

图依据：论文 §3 Methods（§3.1 Text-to-Flow、§3.2 Flow-to-Video、§3.3 Diffusion Policy from Videos）、§5.1 Architecture Details / Fig.1(b)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 499/564 |
| 分组 | WAMs |
| 出处 | arXiv preprint |
| 论文 | <https://arxiv.org/abs/2505.24156> |
| 子类 / 象限 | 视觉规划与IDM · Q4 · Dual-system × IDM |

## 核心机制（归纳）

### 策展导读要点

CogRobot adapts video diffusion to bimanual control through three learned components: instruction-conditioned flow prediction, flow-conditioned RGB prediction, and a task-specific goal-reaching action policy. Flow offers an intermediate description of motion without requiring action labels for the video models. The...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2505.24156) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Towards a Generalizable Bimanual Foundation Policy via Flow-based Video Predi...」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **WAMs**，以及 Contribution 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/rcl_awesome_wam_2505_24156_towards-a-generalizable-bimanual-foundat.md`](../../sources/papers/rcl_awesome_wam_2505_24156_towards-a-generalizable-bimanual-foundat.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2505.24156>
- [arXiv PDF](https://arxiv.org/pdf/2505.24156)：已通读方法部分（实验与附录以略读为准）：§3.1–3.3、§5.1 实验设定

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2505.24156)
