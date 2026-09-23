---
type: entity
tags:
  - dataset
  - navigation
  - gaussian-splatting
  - simulation
status: complete
updated: 2026-09-23
related:
  - ./light-origins.md
  - ../overview/lightorigins-3blogs-technology-map.md
  - ./paper-lightnav-0.md
  - ./light-react.md
  - ./paper-light-loco-parkour.md
sources:
  - ../../sources/papers/habitatgs_lightnav_2026.md
  - ../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md
summary: "HabitatGS：INSIGHT-Bench 训练片段 10.9% 来自 HabitatGS；无标注场景经 Molmo2 多视角目标检测后接入统一目标库。"
---

# HabitatGS

**HabitatGS**（HabitatGS：高斯泼溅场景重建（LightNav 数据引擎来源））在 [Light Origins · LightNav-0：以规模化 Real2Sim2Real 实现零样本通用导航](https://www.lightorigins.com/blog/lightnav-0) 中被引用。

## 一句话定义

**INSIGHT-Bench 训练片段 10.9% 来自 HabitatGS；无标注场景经 Molmo2 多视角目标检测后接入统一目标库。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| VLN | Vision-and-Language Navigation | 视觉-语言导航 |
| ER | Embodied Reasoning | 具身推理；LightNav 第一阶段中期训练 |
| RL | Reinforcement Learning | 强化学习 |
| R2S2R | Real-to-Sim-to-Real | 真场景→仿真合成→真机部署 |

## 为什么重要

- LightNav Real2Sim2Real 场景格式之一。
- 博客 ingest 独立节点（非重复 stub）；见 [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)。

## 核心信息

| 项 | 内容 |
|----|------|
| **类型** | dataset |
| **出处** | LightNav Tech Blog 引用 |
| **开源** | **待核实** |
| **重定向就绪度** | 场景资产而非动作数据，**不涉及动作重定向**；作为策略输入前需按目标形态适配（相机高度/视场、碰撞体与可通行性、导航图分辨率），具体适配口径 **待核实** |



## 结论

**HabitatGS 是 LightNav / LightParkour 管线中的关键组件——读博客数字前先对齐本页定义与开源边界。**

1. 状态：**待核实**
2. 与机构页 [亮源新创（Light Origins）](./light-origins.md) 三段范式对照阅读。
3. 工程复现以官方后续发布为准。

## 关联页面

- [亮源新创（Light Origins）](./light-origins.md)
- [lightorigins-3blogs-technology-map](../overview/lightorigins-3blogs-technology-map.md)
- [LightNav-0](./paper-lightnav-0.md)
- [Light REACT](./light-react.md)

## 参考来源

- [habitatgs_lightnav_2026.md](../../sources/papers/habitatgs_lightnav_2026.md)
- [lightorigins_lightnav_0_2026-09-01.md](../../sources/blogs/lightorigins_lightnav_0_2026-09-01.md)
- [Tech Blog](https://www.lightorigins.com/blog/lightnav-0)

## 推荐继续阅读

- [Light Origins 官网](https://www.lightorigins.com/)
- [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)
