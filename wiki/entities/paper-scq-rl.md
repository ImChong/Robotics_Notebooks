---
type: entity
tags:
  - paper
  - rl
  - offline-rl
  - deployment
status: complete
updated: 2026-09-15
arxiv: "2609.12749"
related:
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
  - ../overview/vla-tamp-planning-11-papers-technology-map.md
sources:
  - ../../sources/papers/scq-rl_arxiv_2609_12749.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md
summary: "用严格为正的 sigmoid-bounded 熵项稳定保守 Q 学习与 offline-to-online 策略更新；覆盖 D4RL、视觉任务与四类真机。"
---

# SCQ（arXiv:2609.12749）

**SCQ**（[SCQ: Stabilizing Conservative Q-Learning with Sigmoid-Bounded Entropy](https://arxiv.org/abs/2609.12749)）来自 [具身智能小站 11 篇 VLA/TAMP 盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)。标准 log-entropy 可为负并扰动策略更新；SCQ 保留保守 Q 正则同时约束熵贡献。

## 一句话定义

**用严格为正的 sigmoid-bounded 熵项稳定保守 Q 学习与 offline-to-online 策略更新。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| TAMP | Task and Motion Planning | 任务与运动规划 |
| OOD | Out-of-Distribution | 分布外泛化评测 |

## 为什么重要

- 标准 log-entropy 可为负并扰动策略更新；SCQ 保留保守 Q 正则同时约束熵贡献。
- 开源状态：**待发布**（步骤 2.5 核查，2026-09-14）。
- 与 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.12749](https://arxiv.org/abs/2609.12749) |
| **项目页** | https://scq-rl.github.io |
| **开源** | **待发布** |
| **文内指标** | D4RL、视觉任务与四类真实机器人；one-shot 演示初始化后无需 HIL 在线改进。 |


## 源码运行时序图

**不适用**（截至 2026-09-14 项目页未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 要点 | D4RL、视觉任务与四类真实机器人；one-shot 演示初始化后无需 HIL 在线改进。 |

- **读法：** 本页为清单摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

- **标准最大熵 RL（SAC 一系）** — 用 $\log$ 形式的熵项，取值可为负，在保守 Q 正则同时生效时会与之相互拉扯；SCQ 换成 **严格为正的 sigmoid-bounded 熵**，保留探索激励但不让熵项反号。差别只在正则项形状，不改 actor–critic 骨架，见 [PPO vs SAC](../comparisons/ppo-vs-sac.md)。
- **原版保守 Q 学习（CQL）** — 靠压低 OOD 动作的 Q 值换安全，代价是过保守、在线阶段爬不动；SCQ 的卖点正是 **offline→online 接缝处** 不塌，one-shot 演示初始化后可继续在线改进。
- **[在线 vs 离线 RL](../comparisons/online-vs-offline-rl.md)** — 该页给出两种数据来源方式的根本差异；SCQ 属于其中 **offline-to-online 过渡** 这一档，不是纯离线也不是纯在线。
- **[ArtManip](./paper-artmanip.md)（同批）** — 同为 RL 路线但改动层级不同：ArtManip 动的是 **环境与课程**（物体生成、随机化、奖励课程），SCQ 动的是 **目标函数里的正则项**；一个换题面，一个换解法。
- **[DATAFARM](./paper-datafarm.md)（同批）** — 两篇都在回答「怎么少花真机数据」：DATAFARM 用合成示范补 BC，SCQ 用离线数据 + 少量在线交互补 RL；模仿与强化两条路的同一诉求，见 [RL vs IL](../comparisons/rl-vs-il.md)。

- **读法：** 以上为知识库内 **路线级** 对照；与原文 baseline（D4RL 各任务、视觉任务与四类真机）的逐项定量比较以 **原文 PDF** 为准（[参考来源](#参考来源)）。开源状态为 **待发布**，暂无法按代码口径复现对照。

## 结论

**SCQ 适合作为本期「待发布」边界下的快速索引页，部署前请核对项目页/仓库可运行性。**

1. 核心贡献：标准 log-entropy 可为负并扰动策略更新；SCQ 保留保守 Q 正则同时约束熵贡献。
2. 开源结论：**待发布** — 以项目页实际链接为准（入库日 2026-09-14）。
3. 横向对照见 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)
- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [scq-rl_arxiv_2609_12749.md](../../sources/papers/scq-rl_arxiv_2609_12749.md)
- [wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)
- [arXiv:2609.12749](https://arxiv.org/abs/2609.12749)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.12749)
- [项目页](https://scq-rl.github.io)
