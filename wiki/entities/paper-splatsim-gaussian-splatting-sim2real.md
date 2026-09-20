---
type: entity
tags:
  - paper
  - sim2real
  - gaussian-splatting
  - manipulation
status: complete
updated: 2026-09-20
arxiv: "2409.10161"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_39_splatsim-gaussian-splatting-sim2real.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "用 3D Gaussian Splatting 从真实场景重建可渲染仿真，实现 RGB 操控策略零样本迁移。"
---

# SplatSim: zero-shot sim2real transfer of RGB manipulation policies using Gaussian splatting（FreeDof [39/44]）

**SplatSim: zero-shot sim2real transfer of RGB manipulation policies using Gaussian splatting**（[arXiv:2409.10161](https://arxiv.org/abs/2409.10161)）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[39/44]**，归类 **视觉 Sim2Real**。

## 一句话定义

用 3D Gaussian Splatting 从真实场景重建可渲染仿真，实现 RGB 操控策略零样本迁移。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| 3DGS | 3D Gaussian Splatting | 三维高斯溅射 |
| Sim2Real | Simulation to Real | 仿真到真机 |
| RGB | Red Green Blue | 视觉像素策略 |

## 为什么重要

- 文内视觉 gap 旁支代表，与动力学 Sim2Real 正交。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **视觉 Sim2Real** 节点。
- 开源结论：**待核实**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | ICRA 2025 |
| **文内章节** | 视觉 Sim2Real |
| **要点** | Real-to-sim 场景重建 + 策略在重建渲染中训练。 |
| **开源** | **待核实** |


## 源码运行时序图

**不适用（待核实）** — 截至 2026-09-20 以项目页/论文 Code availability 为准；入库未核验可运行入口。


## 结论

**观测层 gap 应走视觉/渲染路线，勿用动力学 SysID 硬修。**

1. 文内角色：视觉 Sim2Real 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：Real-to-sim 场景重建 + 策略在重建渲染中训练。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_39_splatsim-gaussian-splatting-sim2real.md](../../sources/papers/freedof_sim2real_39_splatsim-gaussian-splatting-sim2real.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2409.10161)
- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
