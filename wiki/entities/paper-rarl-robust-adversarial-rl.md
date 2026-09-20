---
type: entity
tags:
  - paper
  - robust-rl
  - sim2real
  - adversarial-training
status: complete
updated: 2026-09-20
arxiv: "1703.02702"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_18_rarl-robust-adversarial-rl.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "训练对手网络施加扰动力，与策略对抗以提升鲁棒性。"
---

# Robust adversarial reinforcement learning（FreeDof [18/44]）

**Robust adversarial reinforcement learning (RARL)**（[arXiv:1703.02702](https://arxiv.org/abs/1703.02702)）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[18/44]**，归类 **域随机化**。

## 一句话定义

训练对手网络施加扰动力，与策略对抗以提升鲁棒性。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RARL | Robust Adversarial Reinforcement Learning | 本文方法 |
| RL | Reinforcement Learning | 强化学习 |
| DR | Domain Randomization | 域随机化 |

## 为什么重要

- 文内对抗训练代表；扰动若不符合物理会导致过度保守。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **域随机化** 节点。
- 开源结论：**待核实**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | ICML 2017 |
| **文内章节** | 域随机化 |
| **要点** | min-max 博弈：策略 vs 扰动生成器。 |
| **开源** | **待核实** |


## 源码运行时序图

**不适用（待核实）** — 截至 2026-09-20 以项目页/论文 Code availability 为准；入库未核验可运行入口。


## 结论

**对抗训练可补 DR 尾部，但扰动物理合理性必须约束。**

1. 文内角色：域随机化 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：min-max 博弈：策略 vs 扰动生成器。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_18_rarl-robust-adversarial-rl.md](../../sources/papers/freedof_sim2real_18_rarl-robust-adversarial-rl.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/1703.02702)
- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
