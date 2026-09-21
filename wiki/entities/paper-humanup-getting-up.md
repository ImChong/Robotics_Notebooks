---
type: entity
tags:
  - paper
  - humanoid
  - fall-recovery
  - getting-up
  - unitree-g1
status: complete
updated: 2026-09-21
arxiv: "2502.12152"
related:
  - ./light-origins.md
  - ../overview/lightorigins-3blogs-technology-map.md
  - ./paper-lightnav-0.md
  - ./light-react.md
  - ./paper-light-loco-parkour.md
sources:
  - ../../sources/papers/humanup_getting_up_arxiv_2502_12152.md
  - ../../sources/blogs/lightorigins_light_react_2026-09-09.md
summary: "HUMANUP：两阶段 RL：先发现起身轨迹再 refine 为可部署策略；G1 六地形俯卧/仰卧起身 78.3% 成功率。"
---

# HUMANUP

**HUMANUP**（Learning Getting-Up Policies for Real-World Humanoid Robots）在 [Light Origins · Light REACT：面向规模化部署的全身韧性智能](https://www.lightorigins.com/blog/light-react) 中被引用。

## 一句话定义

**两阶段 RL：先发现起身轨迹再 refine 为可部署策略；G1 六地形俯卧/仰卧起身 78.3% 成功率。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| VLN | Vision-and-Language Navigation | 视觉-语言导航 |
| ER | Embodied Reasoning | 具身推理；LightNav 第一阶段中期训练 |
| RL | Reinforcement Learning | 强化学习 |
| R2S2R | Real-to-Sim-to-Real | 真场景→仿真合成→真机部署 |

## 为什么重要

- Light REACT 韧性金字塔 L2「跌倒后起身恢复行走」的直接前序。
- 博客 ingest 独立节点（非重复 stub）；见 [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)。

## 核心信息

| 项 | 内容 |
|----|------|
| **类型** | paper |
| **出处** | RSS 2025 |
| **开源** | **待核实** |
| **arXiv** | [2502.12152](https://arxiv.org/abs/2502.12152) |


## 源码运行时序图

**不适用**（截至入库日无官方可运行实现，或仅有博客/论文叙述）。

## 结论

**HUMANUP 在 Light Origins 三篇 Tech Blog 引用链中承担「Light REACT 韧性金字塔 L2「跌倒后起身恢复行走」的直接前序。…」角色——部署前以 arXiv/项目页与开源状态为准。**

1. 开源：**待核实**；勿凭博客脚注臆断可复现性。
2. 与 [Light REACT](./light-react.md) / [LightNav-0](./paper-lightnav-0.md) / [Light-Loco-Parkour](./paper-light-loco-parkour.md) 按能力轴交叉阅读。
3. 定量指标以原文 PDF 为准；本页为博客 ingest 级摘要。

## 关联页面

- [亮源新创（Light Origins）](./light-origins.md)
- [lightorigins-3blogs-technology-map](../overview/lightorigins-3blogs-technology-map.md)
- [LightNav-0](./paper-lightnav-0.md)
- [Light REACT](./light-react.md)

## 参考来源

- [humanup_getting_up_arxiv_2502_12152.md](../../sources/papers/humanup_getting_up_arxiv_2502_12152.md)
- [lightorigins_light_react_2026-09-09.md](../../sources/blogs/lightorigins_light_react_2026-09-09.md)
- [Tech Blog](https://www.lightorigins.com/blog/light-react)

## 推荐继续阅读

- [Light Origins 官网](https://www.lightorigins.com/)
- [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)
