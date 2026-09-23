---
type: entity
tags:
  - paper
  - navigation
  - visual-tracking
  - vla
  - evt-bench
status: complete
updated: 2026-09-23
arxiv: "2505.23189"
related:
  - ./light-origins.md
  - ../overview/lightorigins-3blogs-technology-map.md
  - ./paper-lightnav-0.md
  - ./light-react.md
  - ./paper-light-loco-parkour.md
sources:
  - ../../sources/papers/trackvla_arxiv_2505_23189.md
  - ../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md
summary: "TrackVLA：提出 EVT-Bench 野外具身视觉跟踪基准；LightNav-0 第三阶段 GRPO 在线 RL 在该基准干扰跟踪上继续提升。"
---

# TrackVLA

**TrackVLA**（TrackVLA: Embodied Visual Tracking in the Wild）在 [Light Origins · LightNav-0：以规模化 Real2Sim2Real 实现零样本通用导航](https://www.lightorigins.com/blog/lightnav-0) 中被引用。

## 一句话定义

**提出 EVT-Bench 野外具身视觉跟踪基准；LightNav-0 第三阶段 GRPO 在线 RL 在该基准干扰跟踪上继续提升。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| VLN | Vision-and-Language Navigation | 视觉-语言导航 |
| ER | Embodied Reasoning | 具身推理；LightNav 第一阶段中期训练 |
| RL | Reinforcement Learning | 强化学习 |
| R2S2R | Real-to-Sim-to-Real | 真场景→仿真合成→真机部署 |

## 为什么重要

- LightNav 三阶段训练 RL 阶段的评测锚点。
- 博客 ingest 独立节点（非重复 stub）；见 [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)。

## 核心信息

| 项 | 内容 |
|----|------|
| **类型** | paper |
| **出处** | 2025 |
| **开源** | **待核实** |
| **arXiv** | [2505.23189](https://arxiv.org/abs/2505.23189) |


## 源码运行时序图

**不适用**（截至入库日无官方可运行实现，或仅有博客/论文叙述）。

## 实验与评测

- **自带标尺：** 提出 **EVT-Bench**（野外具身视觉跟踪基准），把「跟住一个移动目标」从定性演示变成可打分的任务，覆盖干扰物、遮挡与目标重识别等野外条件。
- **主指标形态：** 跟踪类任务关注 **持续跟随时长 / 跟丢率 / 干扰下的重捕获**，与 ObjectNav 的 SR/SPL 不是同一套量，**不可混用**。
- **被谁用：** [LightNav-0](./paper-lightnav-0.md) 第三阶段 GRPO 在线 RL 在该基准的 **干扰跟踪** 子项上继续提升——说明 EVT-Bench 已被当作外部训练信号的验收面。
- **数值口径：** 本页为博客 ingest 级摘要，**未复核逐项分数**；各方法成绩 **以 [原文](https://arxiv.org/abs/2505.23189) 为准**；开源状态 **待核实**。

## 与其他工作对比

| 维度 | TrackVLA / EVT-Bench（本页） | ObjectNav 类基准（如 [HM3D-OVON](./paper-hm3d-ovon.md)） | 纯视觉目标跟踪（非具身） |
|------|-------------------------------|-----------------------------------------------------------|---------------------------|
| 目标 | **移动** 目标，需持续跟随 | 静止目标，找到即成功 | 移动目标，但只输出框 |
| 动作耦合 | 跟踪结果直接驱动本体运动 | 导航策略 | 无本体 |
| 主要失败模式 | 干扰物换人、遮挡后跟错 | 找不到 / 绕远 | 跟丢但无后果 |
| 指标 | 跟随时长 / 跟丢与重捕获 | SR / SPL | IoU / 成功率曲线 |

- **具身跟踪的难点不在「框准」，在「跟丢后果不可逆」：** 非具身跟踪跟丢一帧可以下一帧补回，具身跟踪跟错方向会把本体带离目标，误差是 **累积且有物理代价** 的。
- **在评测闭环中的位置：** 属 [评测闭环](../queries/embodied-eval-benchmark-selection-loop.md) 的 ③ 策略成功率层；仿真跟踪成绩外推真机仍需 ④ 层校准。

## 结论

**TrackVLA 在 Light Origins 三篇 Tech Blog 引用链中承担「LightNav 三阶段训练 RL 阶段的评测锚点。…」角色——部署前以 arXiv/项目页与开源状态为准。**

1. 开源：**待核实**；勿凭博客脚注臆断可复现性。
2. 与 [Light REACT](./light-react.md) / [LightNav-0](./paper-lightnav-0.md) / [Light-Loco-Parkour](./paper-light-loco-parkour.md) 按能力轴交叉阅读。
3. 定量指标以原文 PDF 为准；本页为博客 ingest 级摘要。

## 关联页面

- [亮源新创（Light Origins）](./light-origins.md)
- [lightorigins-3blogs-technology-map](../overview/lightorigins-3blogs-technology-map.md)
- [LightNav-0](./paper-lightnav-0.md)
- [Light REACT](./light-react.md)

## 参考来源

- [trackvla_arxiv_2505_23189.md](../../sources/papers/trackvla_arxiv_2505_23189.md)
- [lightorigins_lightnav_0_2026-09-01.md](../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md)
- [Tech Blog](https://www.lightorigins.com/blog/lightnav-0)

## 推荐继续阅读

- [Light Origins 官网](https://www.lightorigins.com/)
- [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)
