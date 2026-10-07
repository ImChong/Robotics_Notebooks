---
type: entity
tags:
  - paper
  - locomotion
  - transformer-xl
  - in-context-learning
  - cross-embodiment
  - skild-ai
status: complete
updated: 2026-10-07
arxiv: "2509.23745"
related:
  - ./skild-ai.md
  - ./skild-s1.md
  - ./light-origins.md
  - ../overview/lightorigins-3blogs-technology-map.md
  - ./paper-lightnav-0.md
  - ./light-react.md
  - ./paper-light-loco-parkour.md
sources:
  - ../../sources/sites/skild-ai-timeline-audit-2026-10-07.md
  - ../../sources/papers/locoformer_corl_2025.md
  - ../../sources/blogs/lightorigins_light_react_2026-09-09.md
summary: "LocoFormer：大规模 PPO + 程序生成机器人 + Transformer-XL 跨 episode 记忆；未见形态/电机故障下 test-time 适应。"
---

# LocoFormer

**LocoFormer** 是 Skild AI 署名的跨本体运动控制论文；[官方项目页](https://generalist-locomotion.github.io/)连接公司博客。[Light REACT](./light-react.md)是后续引用者，不是项目归属。

## 一句话定义

**大规模 PPO + 程序生成机器人 + Transformer-XL 跨 episode 记忆；未见形态/电机故障下 test-time 适应。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PPO | Proximal Policy Optimization | 大规模仿真强化学习 |
| TXL | Transformer-XL | 跨片段、跨试次保留记忆 |
| ICL | In-Context Learning | 从部署历史适应，不更新权重 |
| OOD | Out of Distribution | 未见形态与动力学条件 |

## 为什么重要

- Light REACT 对比 Transformer 64 帧上下文时引用；LocoFormer 代表「长上下文运动适应」前序。
- 博客 ingest 独立节点（非重复 stub）；见 [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)。

## 核心信息

| 项 | 内容 |
|----|------|
| **类型** | paper |
| **出处** | CoRL 2025 |
| **机构 / 作者** | Skild AI / Min Liu、Deepak Pathak、Ananye Agarwal |
| **日期** | 公司博客 2025-09-24；arXiv v1 提交 2025-09-28 |
| **项目页** | [generalist-locomotion.github.io](https://generalist-locomotion.github.io/) |
| **开源** | 2026-10-07 官方项目页未见代码、权重或数据下载入口；不推定待发布 |
| **arXiv** | [2509.23745](https://arxiv.org/abs/2509.23745) |


## 源码运行时序图

**不适用**（截至入库日无官方可运行实现，或仅有博客/论文叙述）。

## 实验与评测

- **训练规模与设定：** 大规模 PPO + **程序生成机器人形态**，用 Transformer-XL 维持 **跨 episode** 的长上下文记忆。
- **考的是 test-time 适应，不是训练分布内成功率：** 评测重点为 **未见形态** 与 **电机故障** 下能否在部署期内自行适应——这类指标对「训练时见过多少形态」极其敏感，读数前必须对齐形态采样范围。
- **数值口径：** 本页为博客 ingest 级摘要，**未复核逐项分数**；各设定成功率与适应曲线 **以 [原文](https://arxiv.org/abs/2509.23745)（CoRL 2025）为准**。
- **复现边界：** 官方项目页未见可运行资产入口（2026-10-07），暂无法独立重跑。

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

**LocoFormer 是 Skild AI 的长上下文跨本体运动策略；公司路线应纳入其 2025 年运动适应阶段，部署前以官方论文/项目页与资产开放状态为准。**

1. 开源：官方项目页未见资产入口；勿凭博客引用臆断可复现性。
2. 与 [Light REACT](./light-react.md) / [LightNav-0](./paper-lightnav-0.md) / [Light-Loco-Parkour](./paper-light-loco-parkour.md) 按能力轴交叉阅读。
3. 定量指标以原文 PDF 为准；本页为博客 ingest 级摘要。

## 关联页面

- [Skild AI（论文署名机构）](./skild-ai.md)
- [S1（操作域视频上下文学习）](./skild-s1.md)

- [亮源新创（Light Origins）](./light-origins.md)
- [lightorigins-3blogs-technology-map](../overview/lightorigins-3blogs-technology-map.md)
- [LightNav-0](./paper-lightnav-0.md)
- [Light REACT](./light-react.md)

## 参考来源

- [Skild AI 官方时间线核查（2026-10-07）](../../sources/sites/skild-ai-timeline-audit-2026-10-07.md)

- [locoformer_corl_2025.md](../../sources/papers/locoformer_corl_2025.md)
- [lightorigins_light_react_2026-09-09.md](../../sources/blogs/lightorigins_light_react_2026-09-09.md)
- [Tech Blog](https://www.lightorigins.com/blog/light-react)

## 推荐继续阅读

- [Light Origins 官网](https://www.lightorigins.com/)
- [3 篇技术地图](../overview/lightorigins-3blogs-technology-map.md)
