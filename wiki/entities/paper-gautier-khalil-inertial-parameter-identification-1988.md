---
type: entity
tags:
  - paper
  - system-identification
  - base-parameters
  - sim2real
status: complete
updated: 2026-09-20
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_02_gautier-khalil-inertial-parameter-identification-1988.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "提出机器人惯性参数基参数（base parameters）概念：部分参数只能成组辨识。"
---

# On the identification of the inertial parameters of robots

**On the identification of the inertial parameters of robots**（CDC 1988）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[02/44]**，归类 **系统辨识**。

## 一句话定义

提出机器人惯性参数基参数（base parameters）概念：部分参数只能成组辨识。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SysID | System Identification | 系统辨识 |
| QR | QR decomposition | 矩阵分解求基参数 |
| DOF | Degrees of Freedom | 自由度 |

## 为什么重要

- FreeDof 文内强调 PD 增益与惯量退化方向，其理论根源在基参数与可辨识性分析。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **系统辨识** 节点。
- 开源结论：**不适用**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | CDC 1988 |
| **文内章节** | 系统辨识 |
| **要点** | 通过 QR 分解等线性代数工具找出动力学回归矩阵的基，避免对不可辨识参数做无意义估计。 |
| **开源** | **不适用** |


## 源码运行时序图

**不适用（不适用）** — 经典文献或策展资源，无可运行官方代码仓。


## 结论

**读 PACE 退化方向解析前，先读此文可理解「秩亏不是数值 bug 而是结构问题」。**

1. 文内角色：系统辨识 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：通过 QR 分解等线性代数工具找出动力学回归矩阵的基，避免对不可辨识参数做无意义估计。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_02_gautier-khalil-inertial-parameter-identification-1988.md](../../sources/papers/freedof_sim2real_02_gautier-khalil-inertial-parameter-identification-1988.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
