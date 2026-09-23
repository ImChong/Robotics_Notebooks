---
type: entity
tags:
  - paper
  - navigation
  - objectnav
  - open-vocabulary
  - benchmark
status: complete
updated: 2026-09-23
arxiv: "2409.01535"
related:
  - ../queries/embodied-eval-benchmark-selection-loop.md
  - ./light-origins.md
  - ../overview/lightorigins-3blogs-technology-map.md
  - ./paper-lightnav-0.md
  - ./light-react.md
  - ./paper-light-loco-parkour.md
sources:
  - ../../sources/papers/hm3d_ovon_2024.md
  - ../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md
summary: "HM3D-OVON：HM3D 上开放词汇目标导航设定；LightNav-0 十项单目评测之一。"
---

# HM3D-OVON

**HM3D-OVON**（Open-Vocabulary Object Goal Navigation with Embodied Foundation Models）在 [Light Origins · LightNav-0：以规模化 Real2Sim2Real 实现零样本通用导航](https://www.lightorigins.com/blog/lightnav-0) 中被引用。

## 一句话定义

**HM3D 上开放词汇目标导航设定；LightNav-0 十项单目评测之一。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| VLN | Vision-and-Language Navigation | 视觉-语言导航 |
| ER | Embodied Reasoning | 具身推理；LightNav 第一阶段中期训练 |
| RL | Reinforcement Learning | 强化学习 |
| R2S2R | Real-to-Sim-to-Real | 真场景→仿真合成→真机部署 |

## 为什么重要

- LightNav 跨任务泛化评测矩阵中的 ObjectNav 轴。
- 博客 ingest 独立节点（非重复 stub）；见 [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)。

## 核心信息

| 项 | 内容 |
|----|------|
| **类型** | paper |
| **出处** | 2024 |
| **开源** | **待核实** |
| **arXiv** | [2409.01535](https://arxiv.org/abs/2409.01535) |


## 源码运行时序图

**不适用**（截至入库日无官方可运行实现，或仅有博客/论文叙述）。

## 实验与评测

- **评测设定：** HM3D 场景上的 **开放词汇目标导航（OVON）**——目标类别在评测期可以是训练未见的自然语言名词，因此不能用闭集 ObjectNav 的类别表兜底。
- **主指标：** 导航类基准通用的 **SR（成功率）** 与 **SPL（按路径长度加权的成功率）**；SR 高而 SPL 低意味着「能找到但绕远」。
- **在 LightNav 中的位置：** 作为 [LightNav-0](./paper-lightnav-0.md) **十项单目评测之一** 被引用，用于验证零样本导航的开放词汇覆盖面。
- **数值口径：** 本页为博客 ingest 级摘要，**未复核逐项分数**；各方法的 SR/SPL **以 [原文](https://arxiv.org/abs/2409.01535) 为准**，勿用博客脚注回填。

## 与其他工作对比

| 维度 | HM3D-OVON（本页） | 闭集 ObjectNav 基准 | [INSIGHT-Bench](./insight-bench.md) |
|------|-------------------|---------------------|--------------------------------------|
| 目标指定方式 | **开放词汇** 自然语言 | 固定类别表 | LightNav 自建导航评测集 |
| 场景来源 | HM3D 真实扫描 | 多为同类扫描/合成场景 | Real2Sim2Real 合成 + 真扫 |
| 考什么 | 语义泛化 + 探索效率 | 探索效率为主 | LightNav 管线端到端导航能力 |
| 易骗人的地方 | 长尾类别上均值 SR 掩盖崩溃 | 类别表内过拟合 | 与训练场景分布重叠 |

- **与闭集基准不可直接比 SR：** 开放词汇设定的失败模式（同义词、细粒度类别、不存在的目标）在闭集表里根本不出现，两套 SR 不是同一个量。
- **在选型闭环里的位置：** 属 [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) 的 **③ 策略任务成功率层**；仿真 SR 外推真机仍需该闭环 ④ 层的校准。

## 结论

**HM3D-OVON 在 Light Origins 三篇 Tech Blog 引用链中承担「LightNav 跨任务泛化评测矩阵中的 ObjectNav 轴。…」角色——部署前以 arXiv/项目页与开源状态为准。**

1. 开源：**待核实**；勿凭博客脚注臆断可复现性。
2. 与 [Light REACT](./light-react.md) / [LightNav-0](./paper-lightnav-0.md) / [Light-Loco-Parkour](./paper-light-loco-parkour.md) 按能力轴交叉阅读。
3. 定量指标以原文 PDF 为准；本页为博客 ingest 级摘要。

## 关联页面

- [亮源新创（Light Origins）](./light-origins.md)
- [lightorigins-3blogs-technology-map](../overview/lightorigins-3blogs-technology-map.md)
- [LightNav-0](./paper-lightnav-0.md)
- [Light REACT](./light-react.md)
- [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) — 本页归其 ③ 策略任务成功率层：开放词汇 ObjectNav 的 SR/SPL，均值成功率会掩盖长尾类别失败

## 参考来源

- [hm3d_ovon_2024.md](../../sources/papers/hm3d_ovon_2024.md)
- [lightorigins_lightnav_0_2026-09-01.md](../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md)
- [Tech Blog](https://www.lightorigins.com/blog/lightnav-0)

## 推荐继续阅读

- [Light Origins 官网](https://www.lightorigins.com/)
- [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)
