---
type: entity
tags:
  - paper
  - domain-randomization
  - sim2real
  - bayesian-optimization
status: complete
updated: 2026-09-20
arxiv: "2003.02471"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_15_muratore-bayesian-optimization-domain-randomization.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "用贝叶斯优化调 DR 分布，只需成功率等稀疏表现信号，无需逐时刻轨迹对齐。"
---

# Data-efficient domain randomization with Bayesian optimization（FreeDof [15/44]）

**Data-efficient domain randomization with Bayesian optimization**（[arXiv:2003.02471](https://arxiv.org/abs/2003.02471)）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[15/44]**，归类 **域随机化**。

## 一句话定义

用贝叶斯优化调 DR 分布，只需成功率等稀疏表现信号，无需逐时刻轨迹对齐。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BO | Bayesian Optimization | 贝叶斯优化 |
| DR | Domain Randomization | 域随机化 |
| Sim2Real | Simulation to Real | 仿真到真机 |

## 为什么重要

- 降低真机数据代价的 DR 调参路线。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **域随机化** 节点。
- 开源结论：**待核实**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | arXiv 2020 |
| **文内章节** | 域随机化 |
| **要点** | BO 搜索随机化超参，以任务表现作为黑盒目标。 |
| **开源** | **待核实** |


## 源码运行时序图

**不适用（待核实）** — 截至 2026-09-20 以项目页/论文 Code availability 为准；入库未核验可运行入口。


## 结论

**真机数据贵时，稀疏奖励 BO 比全轨迹拟合更可行，但样本效率仍有限。**

1. 文内角色：域随机化 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：BO 搜索随机化超参，以任务表现作为黑盒目标。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_15_muratore-bayesian-optimization-domain-randomization.md](../../sources/papers/freedof_sim2real_15_muratore-bayesian-optimization-domain-randomization.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2003.02471)
- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
