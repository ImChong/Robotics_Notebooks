---
type: entity
tags:
  - paper
  - simulation
  - contact-model
  - sim2real
status: complete
updated: 2026-09-20
arxiv: "2110.00541"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_21_acosta-validating-simulators-real-world-impacts.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "用方块抛落与 Cassie 跳跃落地真机冲击数据检验 Drake/MuJoCo/Bullet。"
---

# Validating robotics simulators on real-world impacts（FreeDof [21/44]）

**Validating robotics simulators on real-world impacts**（[arXiv:2110.00541](https://arxiv.org/abs/2110.00541)）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[21/44]**，归类 **域随机化**。

## 一句话定义

用方块抛落与 Cassie 跳跃落地真机冲击数据检验 Drake/MuJoCo/Bullet。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RA-L | Robotics and Automation Letters | IEEE 机器人快报 |
| Sim2Real | Simulation to Real | 仿真到真机 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触动力学仿真器 |

## 为什么重要

- 文内说明接触刚度可辨识性随任务变化——方块 vs 人形落地敏感性不同。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **域随机化** 节点。
- 开源结论：**待核实**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | RA-L 2022 |
| **文内章节** | 域随机化 |
| **要点** | 对比三引擎在冲击阶段的轨迹与接触力复现。 |
| **开源** | **待核实** |


## 源码运行时序图

**不适用（待核实）** — 截至 2026-09-20 以项目页/论文 Code availability 为准；入库未核验可运行入口。


## 结论

**接触参数是否可辨取决于实验条件；不能脱离任务谈参数敏感性。**

1. 文内角色：域随机化 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：对比三引擎在冲击阶段的轨迹与接触力复现。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_21_acosta-validating-simulators-real-world-impacts.md](../../sources/papers/freedof_sim2real_21_acosta-validating-simulators-real-world-impacts.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2110.00541)
- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
