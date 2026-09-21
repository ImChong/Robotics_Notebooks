---
type: entity
tags:
  - paper
  - system-identification
  - experiment-design
  - sim2real
status: complete
updated: 2026-09-20
venue: "IEEE TAC 2009"
related:
  - ../comparisons/sim2real-four-routes-identifiability.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/freedof_sim2real_06_gevers-identification-information-matrix-2009.md
  - ../../sources/papers/freedof_sim2real_44_catalog.md
  - ../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md
summary: "从信息矩阵与实验设计理论回答：激励要多丰富才足以区分待辨参数。"
---

# Identification and the information matrix: how to get just sufficiently rich?

**Identification and the information matrix: how to get just sufficiently rich?**（IEEE TAC 2009）收录于 [自由度FreeDof · Sim2Real 四条路线梳理](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md) 参考文献 **[06/44]**，归类 **系统辨识**。

## 一句话定义

从信息矩阵与实验设计理论回答：激励要多丰富才足以区分待辨参数。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FIM | Fisher Information Matrix | Fisher 信息矩阵 |
| SysID | System Identification | 系统辨识 |
| TAC | Transactions on Automatic Control | IEEE 控制汇刊 |

## 为什么重要

- SPI-Active 与主动激励的理论根源；理解「激励不足」的数学含义。
- 在 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) 中作为 **系统辨识** 节点。
- 开源结论：**不适用**（步骤 2.5，2026-09-20）。

## 核心机制

| 项 | 内容 |
|----|------|
| **出处** | IEEE TAC 2009 |
| **文内章节** | 系统辨识 |
| **要点** | 分析输入信号 richness 与 Fisher 信息矩阵可逆性的关系。 |
| **开源** | **不适用** |


## 源码运行时序图

**不适用（不适用）** — 经典文献或策展资源，无可运行官方代码仓。


## 实验与评测

- **本页为索引级节点**（FreeDof 44 篇梳理 [06/44]）：正文固化文内角色与机制要点，**未转存原文实验表**。
- **回原文须核对的证据**：本页要点是「输入信号 richness 与 Fisher 信息矩阵可逆性的关系」，对应证据是该关系的**理论条件推导**，而非某台机器人上的实测表。
- **读法：** 先对齐平台、任务、指标定义与成功阈值，再读任何数字；勿从公众号摘录外推。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **文内路线** | 归类 **系统辨识**；同路线其他节点见 [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md) 与 [四条路线对比](../comparisons/sim2real-four-routes-identifiability.md) |
| **横比口径** | 属辨识理论的一般结论；落到具体机器人须自行核算 FIM 条件数，不能直接引用文内算例。 |
| **开源状态** | **不适用** — 部署 / 复现前以项目页或原文 Code availability 为准 |

## 结论

**动手做辨识实验前，用此文校准对「足够激励」的预期，避免采集无效数据。**

1. 文内角色：系统辨识 路线上的参考节点，非重复 arXiv 页面。
2. 机制要点：分析输入信号 richness 与 Fisher 信息矩阵可逆性的关系。…
3. 部署/复现前请对照原文与项目页，勿直接外推公众号数字。

## 关联页面

- [Sim2Real 四条路线（可辨识性）](../comparisons/sim2real-four-routes-identifiability.md)
- [44 篇技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
- [Sim2Real](../concepts/sim2real.md)

## 参考来源

- [freedof_sim2real_06_gevers-identification-information-matrix-2009.md](../../sources/papers/freedof_sim2real_06_gevers-identification-information-matrix-2009.md)
- [wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md](../../sources/blogs/wechat_freedof_sim2real_four_routes_dr_to_residual_2026-08-23.md)
- [freedof_sim2real_44_catalog.md](../../sources/papers/freedof_sim2real_44_catalog.md)

## 推荐继续阅读

- [44 篇 Sim2Real 技术地图](../overview/freedof-sim2real-44-papers-technology-map.md)
