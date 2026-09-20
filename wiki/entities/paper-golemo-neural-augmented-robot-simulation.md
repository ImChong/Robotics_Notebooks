---
type: entity
tags:
  - paper
  - residual-learning
  - sim2real
  - neural-augmented-simulation
status: complete
updated: 2026-09-20
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_30_golemo-neural-augmented-robot-simulation.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "保留解析物理模型，用 RNN 学习残差修正不可建模的历史相关误差（回差、延迟等）。"
---

# Sim-to-real transfer with neural-augmented robot simulation

**Sim-to-real transfer with neural-augmented robot simulation**（CoRL 2018）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[30/44]**，归类 **残差学习**。

## 一句话定义

保留解析物理模型，用 RNN 学习残差修正不可建模的历史相关误差（回差、延迟等）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RNN | Recurrent Neural Network | 循环神经网络 |
| Sim2Real | Simulation to Real | 仿真到真机 |
| CoRL | Conference on Robot Learning | 机器人学习会议 |

## 为什么重要

- 文内「替换 vs 叠加」分歧中的叠加/灰盒代表。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **残差学习** 节点。
- 开源结论：**待核实**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | CoRL 2018 |
| **文内章节** | 残差学习 |
| **要点** | 物理仿真 + 神经网络残差项；循环结构表达时序误差。 |
| **开源** | **待核实** |


## 源码运行时序图

**不适用（待核实）** — 截至 2026-09-20 以项目页/论文 Code availability 为准；入库未核验可运行入口。


## 结论

**灰盒组合保留可解释性，适合主误差可物理解释、剩余结构复杂的平台。**

1. 文内角色：残差学习 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：物理仿真 + 神经网络残差项；循环结构表达时序误差。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_30_golemo-neural-augmented-robot-simulation.md](../../sources/papers/freedof_sim2real_30_golemo-neural-augmented-robot-simulation.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
