---
type: entity
tags:
  - paper
  - locomotion
  - transformer-xl
  - in-context-learning
  - cross-embodiment
status: complete
updated: 2026-09-21
arxiv: "2509.23745"
related:
  - ./light-origins.md
  - ../overview/lightorigins-3blogs-technology-map.md
  - ./paper-lightnav-0.md
  - ./light-react.md
  - ./paper-light-loco-parkour.md
sources:
  - ../../sources/papers/locoformer_corl_2025.md
  - ../../sources/blogs/lightorigins_light_react_2026-09-09.md
summary: "LocoFormer：大规模 PPO + 程序生成机器人 + Transformer-XL 跨 episode 记忆；未见形态/电机故障下 test-time 适应。"
---

# LocoFormer

**LocoFormer**（LocoFormer: Generalist Locomotion via Long-Context Adaptation）在 [Light Origins · Light REACT：面向规模化部署的全身韧性智能](https://www.lightorigins.com/blog/light-react) 中被引用。

## 一句话定义

**大规模 PPO + 程序生成机器人 + Transformer-XL 跨 episode 记忆；未见形态/电机故障下 test-time 适应。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| VLN | Vision-and-Language Navigation | 视觉-语言导航 |
| ER | Embodied Reasoning | 具身推理；LightNav 第一阶段中期训练 |
| RL | Reinforcement Learning | 强化学习 |
| R2S2R | Real-to-Sim-to-Real | 真场景→仿真合成→真机部署 |

## 为什么重要

- Light REACT 对比 Transformer 64 帧上下文时引用；LocoFormer 代表「长上下文运动适应」前序。
- 博客 ingest 独立节点（非重复 stub）；见 [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)。

## 核心信息

| 项 | 内容 |
|----|------|
| **类型** | paper |
| **出处** | CoRL 2025 |
| **开源** | **待发布** |
| **arXiv** | [2509.23745](https://arxiv.org/abs/2509.23745) |


## 源码运行时序图

**不适用**（截至入库日无官方可运行实现，或仅有博客/论文叙述）。

## 结论

**LocoFormer 在 Light Origins 三篇 Tech Blog 引用链中承担「Light REACT 对比 Transformer 64 帧上下文时引用；Lo…」角色——部署前以 arXiv/项目页与开源状态为准。**

1. 开源：**待发布**；勿凭博客脚注臆断可复现性。
2. 与 [Light REACT](./light-react.md) / [LightNav-0](./paper-lightnav-0.md) / [Light-Loco-Parkour](./paper-light-loco-parkour.md) 按能力轴交叉阅读。
3. 定量指标以原文 PDF 为准；本页为博客 ingest 级摘要。

## 关联页面

- [亮源新创（Light Origins）](./light-origins.md)
- [lightorigins-3blogs-technology-map](../overview/lightorigins-3blogs-technology-map.md)
- [LightNav-0](./paper-lightnav-0.md)
- [Light REACT](./light-react.md)

## 参考来源

- [locoformer_corl_2025.md](../../sources/papers/locoformer_corl_2025.md)
- [lightorigins_light_react_2026-09-09.md](../../sources/blogs/lightorigins_light_react_2026-09-09.md)
- [Tech Blog](https://www.lightorigins.com/blog/light-react)

## 推荐继续阅读

- [Light Origins 官网](https://www.lightorigins.com/)
- [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)
