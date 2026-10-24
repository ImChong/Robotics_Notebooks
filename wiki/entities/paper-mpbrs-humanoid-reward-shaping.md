---
type: entity
tags: [paper, humanoid, locomotion, reinforcement-learning, reward-shaping, sim2sim]
status: draft
updated: 2026-10-24
project_id: mpbrs-unified-humanoid-reward-configuration
arxiv: "2610.30334"
related:
  - ../methods/reinforcement-learning.md
  - ../concepts/reward-design.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/mpbrs_arxiv_2610_30334.md
summary: "MPBRS evaluates one potential-based reward configuration across nine humanoid robots, requiring robot-specific metadata but no per-robot reward redesign."
---

# MPBRS：九种人形机器人共享奖励配置

**一句话定义：** MPBRS 研究如何用同一组多势函数奖励结构与权重训练多种人形机器人，只提供机器人特定的结构映射和默认姿态元数据。

## 核心方法

奖励塑形项常需要大量手工调权，也可能互相冲突、对单一机器人过拟合，或因小幅权重变化而不稳定。MPBRS 研究 Multi-Potential-Based Reward Shaping：利用多个势函数的差分塑形奖励，并提出统一的人形 locomotion 奖励配置。

论文在九种人形机器人上评估同一奖励结构与权重；每台机器人只需提供关节/连杆映射和默认姿态等机器人特定元数据，而无需重新设计奖励项。

## 论文摘要报告

与标准 shaping 相比，MPBRS 在跨机器人评估中呈现更一致的缩放行为和更低的调参敏感性。摘要未列明全部机器人、指标和数值，具体结果需以论文正文与实验表为准。

## 工程启示

- 共享奖励配置有助于减少跨机器人迁移时重复手工调参。
- 统一的是奖励结构与权重；机器人特定的关节映射、连杆映射和默认姿态仍需正确提供。
- 验证时需区分奖励可移植性、策略性能和各机器人本体差异，避免把相同配置理解为无需适配。

## 开放状态

本次核查的 arXiv 页面未列官方项目页或代码链接，因此开源状态待核实。

## 参考来源

- [MPBRS 论文归档](../../sources/papers/mpbrs_arxiv_2610_30334.md)
- [arXiv:2610.30334](https://arxiv.org/abs/2610.30334)

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Reward Design](../concepts/reward-design.md)
- [Locomotion](../tasks/locomotion.md)
