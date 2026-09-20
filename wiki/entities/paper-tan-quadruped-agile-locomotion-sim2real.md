---
type: entity
tags:
  - paper
  - sim2real
  - quadruped
  - domain-randomization
status: complete
updated: 2026-09-20
arxiv: "1804.10332"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_11_tan-quadruped-agile-locomotion-sim2real.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "腿足 Sim2Real 经典流水线：电机辨识、时延补偿、动力学随机化与推力扰动。"
---

# Sim-to-real: learning agile locomotion for quadruped robots（FreeDof [11/44]）

**Sim-to-real: learning agile locomotion for quadruped robots**（[arXiv:1804.10332](https://arxiv.org/abs/1804.10332)）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[11/44]**，归类 **域随机化**。

## 一句话定义

腿足 Sim2Real 经典流水线：电机辨识、时延补偿、动力学随机化与推力扰动。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 仿真到真机 |
| DR | Domain Randomization | 域随机化 |
| RSS | Robotics: Science and Systems | 机器人科学系统会议 |

## 为什么重要

- 文内第三篇打底；展示 DR 与 SysID 组件如何组合成可部署四足系统。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **域随机化** 节点。
- 开源结论：**待核实**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | RSS 2018 |
| **文内章节** | 域随机化 |
| **要点** | 多阶段校准 + 随机化 + RL 训练；强调工程模块顺序。 |
| **开源** | **待核实** |


## 源码运行时序图

**不适用（待核实）** — 截至 2026-09-20 以项目页/论文 Code availability 为准；入库未核验可运行入口。


## 结论

**DR 不是单点技巧，而是与辨识、时延补偿绑定的系统配方。**

1. 文内角色：域随机化 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：多阶段校准 + 随机化 + RL 训练；强调工程模块顺序。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_11_tan-quadruped-agile-locomotion-sim2real.md](../../sources/papers/freedof_sim2real_11_tan-quadruped-agile-locomotion-sim2real.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/1804.10332)
- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
