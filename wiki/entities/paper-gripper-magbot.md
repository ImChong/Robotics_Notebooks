---
type: entity
tags:
  - paper
  - magnetic
  - manipulation
  - hardware
status: complete
updated: 2026-09-14
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

- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准。

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
