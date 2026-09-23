---
type: entity
tags:
  - paper
  - vln
  - dataset
  - multilingual
  - navigation
status: complete
updated: 2026-09-23
arxiv: "2010.07954"
related:
  - ./light-origins.md
  - ../overview/lightorigins-3blogs-technology-map.md
  - ./paper-lightnav-0.md
  - ./light-react.md
  - ./paper-light-loco-parkour.md
sources:
  - ../../sources/papers/rxr_emnlp_2020.md
  - ../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md
summary: "RxR：多语言 VLN 数据集与 dense spatiotemporal grounding；LightNav-0 SFT 数据池组分之一。"
---

# RxR

**RxR**（Room-Across-Room: Multilingual VLN with Dense Spatiotemporal Grounding）在 [Light Origins · LightNav-0：以规模化 Real2Sim2Real 实现零样本通用导航](https://www.lightorigins.com/blog/lightnav-0) 中被引用。

## 一句话定义

**多语言 VLN 数据集与 dense spatiotemporal grounding；LightNav-0 SFT 数据池组分之一。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| VLN | Vision-and-Language Navigation | 视觉-语言导航 |
| ER | Embodied Reasoning | 具身推理；LightNav 第一阶段中期训练 |
| RL | Reinforcement Learning | 强化学习 |
| R2S2R | Real-to-Sim-to-Real | 真场景→仿真合成→真机部署 |

## 为什么重要

- LightNav 第二阶段对齐 SFT 的公开导航语料对照。
- 博客 ingest 独立节点（非重复 stub）；见 [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)。

## 核心信息

| 项 | 内容 |
|----|------|
| **类型** | paper |
| **出处** | EMNLP 2020 |
| **开源** | **数据集公开** |
| **arXiv** | [2010.07954](https://arxiv.org/abs/2010.07954) |
| **重定向就绪度** | 指令–轨迹语料而非动作数据，**不涉及动作重定向**；作为策略输入需按目标本体的相机视场、步长与动作空间重新离散化，跨本体迁移口径 **待核实** |


## 源码运行时序图

**不适用**（截至入库日无官方可运行实现，或仅有博客/论文叙述）。

## 实验与评测

- **评的是什么：** RxR 是 **数据集/基准**，主指标沿用 VLN 家族的路径保真类指标（SR、SPL、以及对齐路径形状的 nDTW 一族）；其特色在 **多语言**（英/印地/泰卢固）与 **dense spatiotemporal grounding**——指令的每一段与轨迹的每一步对齐。
- **它为什么比早期 VLN 难：** 指令更长、跨语言、且带逐时刻的视觉落点标注，模型不能靠「走到某个显著地标就算对」蒙混过关。
- **在 LightNav 中的位置：** 作为 [LightNav-0](./paper-lightnav-0.md) 第二阶段 **对齐 SFT** 的公开导航语料对照，衡量的是语料覆盖面而非模型能力。
- **数值口径：** 本页为博客 ingest 级摘要，**未复核逐项分数**；各方法成绩 **以 [原文](https://arxiv.org/abs/2010.07954)（EMNLP 2020）与官方 leaderboard 为准**。

## 与其他工作对比

| 维度 | RxR（本页） | 早期单语 VLN 指令集 | [SRDF](./paper-srdf-vln-flywheel.md) 等数据自举工作 |
|------|-------------|---------------------|------------------------------------------------------|
| 数据来源 | **人工采集**，多语言 | 人工采集，单语 | **模型自生成 + 自精炼** |
| 标注粒度 | dense spatiotemporal grounding | 指令级 | 视管线而定 |
| 规模上限 | 受人工成本约束 | 同左 | 可扩展，但受生成质量约束 |
| 主要风险 | 采集成本高 | 地标捷径 | 自举回路里的错误累积 |

- **人工语料与自举语料是互补的两端：** RxR 提供 **高保真但有限** 的对齐信号，SRDF 一类飞轮提供 **可扩展但需过滤** 的量；LightNav 的 SFT 数据池同时用到两者，正是因为单独任何一端都不够。
- **多语言不是附加项：** 跨语言的指代与空间词用法差异，会暴露模型是在 **理解空间关系** 还是在 **记英文模板**——这是 RxR 相对单语基准最难被替代的价值。

## 结论

**RxR 在 Light Origins 三篇 Tech Blog 引用链中承担「LightNav 第二阶段对齐 SFT 的公开导航语料对照。…」角色——部署前以 arXiv/项目页与开源状态为准。**

1. 开源：**数据集公开**；勿凭博客脚注臆断可复现性。
2. 与 [Light REACT](./light-react.md) / [LightNav-0](./paper-lightnav-0.md) / [Light-Loco-Parkour](./paper-light-loco-parkour.md) 按能力轴交叉阅读。
3. 定量指标以原文 PDF 为准；本页为博客 ingest 级摘要。

## 关联页面

- [亮源新创（Light Origins）](./light-origins.md)
- [lightorigins-3blogs-technology-map](../overview/lightorigins-3blogs-technology-map.md)
- [LightNav-0](./paper-lightnav-0.md)
- [Light REACT](./light-react.md)

## 参考来源

- [rxr_emnlp_2020.md](../../sources/papers/rxr_emnlp_2020.md)
- [lightorigins_lightnav_0_2026-09-01.md](../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md)
- [Tech Blog](https://www.lightorigins.com/blog/lightnav-0)

## 推荐继续阅读

- [Light Origins 官网](https://www.lightorigins.com/)
- [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)
