---
type: entity
tags:
  - paper
  - locomotion
  - transformer-xl
  - in-context-learning
  - cross-embodiment
status: complete
updated: 2026-09-23
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

## 实验与评测

- **训练规模与设定：** 大规模 PPO + **程序生成机器人形态**，用 Transformer-XL 维持 **跨 episode** 的长上下文记忆。
- **考的是 test-time 适应，不是训练分布内成功率：** 评测重点为 **未见形态** 与 **电机故障** 下能否在部署期内自行适应——这类指标对「训练时见过多少形态」极其敏感，读数前必须对齐形态采样范围。
- **数值口径：** 本页为博客 ingest 级摘要，**未复核逐项分数**；各设定成功率与适应曲线 **以 [原文](https://arxiv.org/abs/2509.23745)（CoRL 2025）为准**。
- **复现边界：** 代码 **待发布**（入库日口径），暂无法独立重跑。

## 与其他工作对比

| 维度 | LocoFormer（本页） | 显式系统辨识 / 在线参数估计 | 域随机化 + 单一策略 |
|------|---------------------|------------------------------|----------------------|
| 适应机制 | **上下文内适应**：把历史 episode 当输入，不改权重 | 在线估参再调控制器 | 不适应，靠训练分布覆盖 |
| 跨形态能力 | 程序生成形态大规模训练 | 需为每类本体重建模型 | 取决于随机化范围 |
| 故障场景 | 电机故障作为分布外条件之一 | 需故障模型显式建模 | 落在随机化外即失效 |
| 主要代价 | 长上下文的显存与推理成本 | 建模与辨识工程量 | 保守策略带来的性能损失 |

- **「in-context 适应」不是免费的泛化：** 它把适应能力压进 **训练时见过的形态分布**；分布外形态上，长上下文只能帮助更快收敛到一个 **训练分布内的近似**，不能凭空生成新控制律。
- **与 [Light REACT](./light-react.md) 的关系：** 被引作「全身韧性智能」中 **硬件退化后仍可用** 的能力锚点；该定位说明它服务的是部署期鲁棒性，而非峰值敏捷性，读指标时不要与 parkour 类工作横比。

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
