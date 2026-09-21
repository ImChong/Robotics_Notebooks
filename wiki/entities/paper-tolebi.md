---
type: entity
tags:
  - paper
  - humanoid
  - fault-tolerant
  - locomotion
  - reinforcement-learning
status: complete
updated: 2026-09-21
arxiv: "2602.05596"
related:
  - ./light-origins.md
  - ../overview/lightorigins-3blogs-technology-map.md
  - ./paper-lightnav-0.md
  - ./light-react.md
  - ./paper-light-loco-parkour.md
sources:
  - ../../sources/papers/tolebi_arxiv_2602_05596.md
  - ../../sources/blogs/lightorigins_light_react_2026-09-09.md
summary: "TOLEBI：在线关节状态估计 + fallibility rewards 学双足容错行走；仿真注入关节锁定/掉电/扰动，TOCABI 真机验证。"
---

# TOLEBI

**TOLEBI**（TOLEBI: Learning Fault-Tolerant Bipedal Locomotion via Online Status Estimation and Fallibility Rewards）在 [Light Origins · Light REACT：面向规模化部署的全身韧性智能](https://www.lightorigins.com/blog/light-react) 中被引用。

## 一句话定义

**在线关节状态估计 + fallibility rewards 学双足容错行走；仿真注入关节锁定/掉电/扰动，TOCABI 真机验证。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| VLN | Vision-and-Language Navigation | 视觉-语言导航 |
| ER | Embodied Reasoning | 具身推理；LightNav 第一阶段中期训练 |
| RL | Reinforcement Learning | 强化学习 |
| R2S2R | Real-to-Sim-to-Real | 真场景→仿真合成→真机部署 |

## 为什么重要

- Light REACT 韧性金字塔「硬件受损后调整移动」的对照：TOLEBI 显式估计关节状态而非仅靠交互历史 ICL。
- 博客 ingest 独立节点（非重复 stub）；见 [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)。

## 核心信息

| 项 | 内容 |
|----|------|
| **类型** | paper |
| **出处** | ICRA 2026 |
| **开源** | **待核实** |
| **arXiv** | [2602.05596](https://arxiv.org/abs/2602.05596) |


## 源码运行时序图

**不适用**（截至入库日无官方可运行实现，或仅有博客/论文叙述）。

## 结论

**TOLEBI 在 Light Origins 三篇 Tech Blog 引用链中承担「Light REACT 韧性金字塔「硬件受损后调整移动」的对照：TOLEBI 显…」角色——部署前以 arXiv/项目页与开源状态为准。**

1. 开源：**待核实**；勿凭博客脚注臆断可复现性。
2. 与 [Light REACT](./light-react.md) / [LightNav-0](./paper-lightnav-0.md) / [Light-Loco-Parkour](./paper-light-loco-parkour.md) 按能力轴交叉阅读。
3. 定量指标以原文 PDF 为准；本页为博客 ingest 级摘要。

## 关联页面

- [亮源新创（Light Origins）](./light-origins.md)
- [lightorigins-3blogs-technology-map](../overview/lightorigins-3blogs-technology-map.md)
- [LightNav-0](./paper-lightnav-0.md)
- [Light REACT](./light-react.md)

## 参考来源

- [tolebi_arxiv_2602_05596.md](../../sources/papers/tolebi_arxiv_2602_05596.md)
- [lightorigins_light_react_2026-09-09.md](../../sources/blogs/lightorigins_light_react_2026-09-09.md)
- [Tech Blog](https://www.lightorigins.com/blog/light-react)

## 推荐继续阅读

- [Light Origins 官网](https://www.lightorigins.com/)
- [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)
