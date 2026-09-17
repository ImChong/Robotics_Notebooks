---
type: entity
tags:
  - paper
  - magnetic
  - manipulation
  - hardware
status: complete
updated: 2026-09-15
arxiv: "2609.12883"
code: https://ubi-coro.github.io/MagBotSim/magbots.html
related:
  - ../tasks/manipulation.md
  - ./paper-unipart.md
  - ../overview/vla-tamp-planning-11-papers-technology-map.md
sources:
  - ../../sources/papers/gripper-magbot_arxiv_2609_12883.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md
summary: "三 MagLev mover 耦合成低成本并联 6-DoF 操作器并集成 1-DoF 夹爪；仿真与真机展示 pick-and-place。"
---

# Gripper MagBot（arXiv:2609.12883）

**Gripper MagBot**（[From Transportation to Manipulation: Enabling Grasping in Magnetic Robotics](https://arxiv.org/abs/2609.12883)）来自 [具身智能小站 11 篇 VLA/TAMP 盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)。磁悬浮平台擅长搬运却常需额外机械臂抓取；Gripper MagBot 把搬运单元本身变成可抓取操作器。

## 一句话定义

**三 MagLev mover 耦合成低成本并联 6-DoF 操作器并集成 1-DoF 夹爪。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| TAMP | Task and Motion Planning | 任务与运动规划 |
| OOD | Out-of-Distribution | 分布外泛化评测 |

## 为什么重要

- 磁悬浮平台擅长搬运却常需额外机械臂抓取；Gripper MagBot 把搬运单元本身变成可抓取操作器。
- 开源状态：**部分开源**（步骤 2.5 核查，2026-09-14）。
- 与 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.12883](https://arxiv.org/abs/2609.12883) |
| **项目页** | https://sites.google.com/view/gripper-magbot |
| **代码** | https://ubi-coro.github.io/MagBotSim/magbots.html |
| **开源** | **部分开源** |
| **文内指标** | 默认与单轨两种构型；仿真与真机 pick-and-place 演示。 |


## 源码运行时序图

**不适用**（截至 2026-09-14 项目页未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 要点 | 默认与单轨两种构型；仿真与真机 pick-and-place 演示。 |

- **读法：** 本页为清单摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

- **串联关节机械臂** — 常规 6-DoF 臂靠 **串联电机 + 减速器** 堆自由度，成本与惯量都压在关节上；MagBot 把三个 MagLev mover 在平面上耦合出 6-DoF，运动副换成 **磁悬浮无接触驱动**，代价转移到导轨与控制侧。
- **[并联关节运动学](../concepts/humanoid-parallel-joint-kinematics.md)** — MagBot 属并联构型家族：工作空间比串联小、耦合更强，但等效末端惯量低；该页给出并联链路的正/逆解与标定为何更麻烦，是读本页硬件章节的前置。
- **[抓取位姿估计](../methods/grasp-pose-estimation.md)** — 本文的 1-DoF 夹爪把抓取自由度压到最低，等于把难度从 **手指构型规划** 转回 **本体定位精度**；与 [AnyGrasp vs GraspNet](../comparisons/anygrasp-vs-graspnet.md) 那条「多指/多位姿」路线正好相反。
- **[ArtManip](./paper-artmanip.md)（同批）** — 同批中的另一极：ArtManip 在 **多指灵巧手** 上求 in-hand 接触内操作，MagBot 在 **极简末端** 上求低成本本体；两者共同勾出本期「手的自由度该放在哪」的取舍面。
- **[STAR](./paper-star-vtla.md)（同批）** — STAR 往末端加 **触觉感知**，MagBot 往本体减 **机械自由度**；一个加信息、一个减机构。

- **读法：** 以上为知识库内 **路线级** 对照；与原文 baseline 的逐项定量比较以 **原文 PDF** 为准（[参考来源](#参考来源)）。开源状态为 **部分开源**（MagBotSim 仿真侧），硬件复现口径以项目页为准。

## 结论

**Gripper MagBot 适合作为本期「部分开源」边界下的快速索引页，部署前请核对项目页/仓库可运行性。**

1. 核心贡献：磁悬浮平台擅长搬运却常需额外机械臂抓取；Gripper MagBot 把搬运单元本身变成可抓取操作器。
2. 开源结论：**部分开源** — 以项目页实际链接为准（入库日 2026-09-14）。
3. 横向对照见 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)
- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [gripper-magbot_arxiv_2609_12883.md](../../sources/papers/gripper-magbot_arxiv_2609_12883.md)
- [wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)
- [arXiv:2609.12883](https://arxiv.org/abs/2609.12883)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.12883)
- [项目页](https://sites.google.com/view/gripper-magbot)
- [https://ubi-coro.github.io/MagBotSim/magbots.html](https://ubi-coro.github.io/MagBotSim/magbots.html)
