---
type: entity
tags:
  - paper
  - simulation
  - physics-engine
  - sim2real
status: complete
updated: 2026-09-20
venue: "ICRA 2015"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_20_erez-simulation-tools-comparison-icra-2015.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "横比主流物理引擎的速度–精度权衡，说明引擎选择本身影响 Sim2Real。"
---

# Simulation tools for model-based robotics: comparison of Bullet, Havok, MuJoCo, ODE and PhysX

**Simulation tools for model-based robotics: comparison of Bullet, Havok, MuJoCo, ODE and PhysX**（ICRA 2015）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[20/44]**，归类 **域随机化**。

## 一句话定义

横比主流物理引擎的速度–精度权衡，说明引擎选择本身影响 Sim2Real。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ODE | Open Dynamics Engine | 开源动力学引擎 |
| PhysX | NVIDIA PhysX | 商业物理引擎 |
| Sim2Real | Simulation to Real | 仿真到真机 |

## 为什么重要

- 支撑文内「gap 出在引擎本身」论点。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **域随机化** 节点。
- 开源结论：**不适用**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | ICRA 2015 |
| **文内章节** | 域随机化 |
| **要点** | 统一任务下 benchmark 多引擎接触与积分行为差异。 |
| **开源** | **不适用** |


## 源码运行时序图

**不适用（不适用）** — 经典文献或策展资源，无可运行官方代码仓。


## 实验与评测

- **本页为索引级节点**（FreeDof 44 篇梳理 [20/44]）：正文固化文内角色与机制要点，**未转存原文实验表**。
- **回原文须核对的证据**：本页要点是「统一任务下 benchmark 多引擎接触与积分行为差异」，对应证据是同一算例下各引擎的速度–精度曲线与接触 / 积分行为差异表。
- **读法：** 先对齐平台、任务、指标定义与成功阈值，再读任何数字；勿从公众号摘录外推。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **文内路线** | 归类 **域随机化**；同路线其他节点见 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) |
| **横比口径** | 结论绑定 2015 年的引擎版本与 CPU 硬件；速度结论不能照搬到今天的 GPU 并行训练栈。 |
| **开源状态** | **不适用** — 部署 / 复现前以项目页或原文 Code availability 为准 |

## 结论

**换引擎有时比调参更有效；PolySim 类多引擎训练有明确动机。**

1. 文内角色：域随机化 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：统一任务下 benchmark 多引擎接触与积分行为差异。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_20_erez-simulation-tools-comparison-icra-2015.md](../../sources/papers/freedof_sim2real_20_erez-simulation-tools-comparison-icra-2015.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
