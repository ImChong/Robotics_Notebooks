---
type: entity
tags:
  - paper
  - differentiable-simulation
  - locomotion
  - sim2real
status: complete
updated: 2026-09-20
arxiv: "2404.02887"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_41_schwarke-differentiable-simulation-locomotion-corl.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "可微接触模型突破腿足部署瓶颈，首个完全在可微仿真中训练并零样本上真机的腿足 locomotion。"
---

# Learning deployable locomotion control via differentiable simulation（FreeDof [41/44]）

**Learning deployable locomotion control via differentiable simulation**（[arXiv:2404.02887](https://arxiv.org/abs/2404.02887)）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[41/44]**，归类 **可微仿真**。

## 一句话定义

可微接触模型突破腿足部署瓶颈，首个完全在可微仿真中训练并零样本上真机的腿足 locomotion。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真到真机 |
| CoRL | Conference on Robot Learning | 机器人学习会议 |
| Contact | Contact model | 接触动力学模型 |

## 为什么重要

- 文内可微仿真横切工具代表；腿足此前卡在接触梯度。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **可微仿真** 节点。
- 开源结论：**待核实**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | CoRL 2025 |
| **文内章节** | 可微仿真 |
| **要点** | 兼顾梯度信息量与物理保真的接触模型 + 端到端策略优化。 |
| **开源** | **待核实** |


## 源码运行时序图

**不适用（待核实）** — 截至 2026-09-20 以项目页/论文 Code availability 为准；入库未核验可运行入口。


## 结论

**可微仿真可用于辨识或直接训策略，但非光滑接触梯度仍需谨慎。**

1. 文内角色：可微仿真 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：兼顾梯度信息量与物理保真的接触模型 + 端到端策略优化。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_41_schwarke-differentiable-simulation-locomotion-corl.md](../../sources/papers/freedof_sim2real_41_schwarke-differentiable-simulation-locomotion-corl.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2404.02887)
- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
