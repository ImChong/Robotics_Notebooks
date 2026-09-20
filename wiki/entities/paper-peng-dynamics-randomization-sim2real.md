---
type: entity
tags:
  - paper
  - domain-randomization
  - sim2real
  - dynamics-randomization
status: complete
updated: 2026-09-20
arxiv: "1710.06537"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_10_peng-dynamics-randomization-sim2real.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "将域随机化从视觉扩展到质量、摩擦、时延等动力学参数，训练对参数分布鲁棒的策略。"
---

# Sim-to-real transfer of robotic control with dynamics randomization

**Sim-to-real transfer of robotic control with dynamics randomization**（[arXiv:1710.06537](https://arxiv.org/abs/1710.06537)）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[10/44]**，归类 **域随机化**。

## 一句话定义

将域随机化从视觉扩展到质量、摩擦、时延等动力学参数，训练对参数分布鲁棒的策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DR | Domain Randomization | 域随机化 |
| Sim2Real | Simulation to Real | 仿真到真机 |
| ICRA | International Conference on Robotics and Automation | 机器人旗舰会 |

## 为什么重要

- 文内 DR 打底第二篇；腿足与操作 Sim2Real 的共同引用起点。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **域随机化** 节点。
- 开源结论：**待核实**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | ICRA 2018 |
| **文内章节** | 域随机化 |
| **要点** | 在仿真中对动力学参数采样，优化期望回报下的策略。 |
| **开源** | **待核实** |


## 源码运行时序图

**不适用（待核实）** — 截至 2026-09-20 以项目页/论文 Code availability 为准；入库未核验可运行入口。


## 结论

**理解 DR 保守性之前，先读此文建立「随机化参数空间」直觉。**

1. 文内角色：域随机化 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：在仿真中对动力学参数采样，优化期望回报下的策略。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_10_peng-dynamics-randomization-sim2real.md](../../sources/papers/freedof_sim2real_10_peng-dynamics-randomization-sim2real.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/1710.06537)
- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
