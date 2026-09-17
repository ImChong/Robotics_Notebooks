---
type: entity
tags: ['paper', 'vla', 'active-perception', 'manipulation', 'cmu', 'hkust']
status: complete
updated: 2026-09-17
arxiv: "2609.18514"
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ./paper-real-time-expo-ft.md
  - ../overview/perception-action-transfer-9-papers-technology-map.md
sources:
  - ../../sources/papers/activescale_arxiv_2609_18514.md
  - ../../sources/sites/activescale.md
  - ../../sources/blogs/wechat_embodied_station_9_papers_perception_action_transfer_2026-09-17.md
summary: "ActiveScale（arXiv:2609.18514）：历史帧 + 相机 pose token 的主动感知 VLA；1000h 人机 mid-training + AMP 平台；五任务 mean SR 30%→70%；代码待发布。"
---

# ActiveScale（arXiv:2609.18514）

**ActiveScale**（*Scaling Active Perception for Robots across Model, Data, and Hardware*，[arXiv:2609.18514](https://arxiv.org/abs/2609.18514)，[项目页](http://active-scale.github.io/)）来自 [具身智能小站 9 篇盘点](../../sources/blogs/wechat_embodied_station_9_papers_perception_action_transfer_2026-09-17.md)：CMU + HKUST(GZ) 把 **主动视角变化** 写进 VLA 训练闭环——模型、千小时数据与 AMP 移动操作硬件协同。

## 一句话定义

**固定视角看不见目标时，用历史视频 + 显式相机位姿监督让 VLA 学会「先转到能看见的地方再做」——并把这条链扩展到 1000 小时人机混合 mid-training 与可规模采集的 AMP 平台。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| AMP | Active Perception Mobile Manipulation Platform | 本文主动感知移动操作硬件 |
| SR | Success Rate | 任务完全成功率 |
| TP | Task Progress | 分阶段任务进度 |
| FOV | Field of View | 视场角 |

## 为什么重要

- 遮挡抽屉、包内、桌下等场景里，**看哪里** 与 **做什么** 同等关键；纯固定相机 VLA 易在主动搜索任务上失效。
- **模型–数据–硬件** 三件套同设计：pose token、egocentric mid-training、单操作员 AMP 采集形成可扩展 recipe。
- 开源结论：**待发布**（步骤 2.5，2026-09-17）。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.18514](https://arxiv.org/abs/2609.18514) |
| **开源** | **待发布** |
| **模型** | 基于 π₀.5；当前帧 + 三帧历史（间隔 16 帧）；每帧 camera token + 9D pose 头；推理移除 pose 头 |
| **数据** | Stage1：1000h 人机 1:1 mid-training；Stage2：AMP 任务专属 post-training |
| **文内指标** | 五任务 mean SR 30.0%→70.0%，TP 41.6%→78.4%；264.5 Hz（50 块，RTX 4090） |

## 源码运行时序图

**不适用**（截至 2026-09-17 项目页未发布可运行代码仓库）。

## 实验与评测

- Bag / Drawer / Table（遮挡下操作）+ Pot / Box（主动搜索）；每任务 150 demo、20 rollouts。
- Ablation：mid-training、history-only、history+pose 逐项贡献 SR/TP。
- **读法：** 索引级摘要；硬件栈（Cobot-Magic + Quest 2）与 baseline 协议以原文为准。

## 结论

**ActiveScale 把主动感知从「额外相机控制脚本」升格为 VLA 的可监督接口——成效高度依赖 mid-training 规模与 AMP 级采集闭环。**

1. 待代码发布前，以项目页 demo 与五任务协议为跟踪锚点。
2. Pose 监督 + 历史帧是 SR/TP 双涨的主因；仅加 history 不够。
3. 与 [Real-Time EXPO-FT](./paper-real-time-expo-ft.md) 互补：一个解决 **看哪里**，一个解决 **动作跟不上观测**。
4. 部署前先确认是否与现有 π 系 VLA 权重/动作空间兼容。

## 关联页面

- [vla](../methods/vla.md)
- [manipulation](../tasks/manipulation.md)
- [Real-Time EXPO-FT](./paper-real-time-expo-ft.md)
- [9 篇技术地图](../overview/perception-action-transfer-9-papers-technology-map.md)

## 参考来源

- [activescale_arxiv_2609_18514.md](../../sources/papers/activescale_arxiv_2609_18514.md)
- [wechat_embodied_station_9_papers_perception_action_transfer_2026-09-17.md](../../sources/blogs/wechat_embodied_station_9_papers_perception_action_transfer_2026-09-17.md)
- [arXiv:2609.18514](https://arxiv.org/abs/2609.18514)

## 推荐继续阅读

- [ActiveScale 项目页](http://active-scale.github.io/)
- [arXiv PDF](https://arxiv.org/pdf/2609.18514)
