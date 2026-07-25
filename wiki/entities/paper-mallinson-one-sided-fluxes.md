---
type: entity
tags: [paper, magnetics, halbach, permanent-magnet, ampex, recording]
status: complete
updated: 2026-07-25
venue: "IEEE Trans. Magn. 1973"
related:
  - ../concepts/halbach-array.md
  - ./paper-halbach-permanent-multipole-magnets.md
  - ./paper-zhu-howe-halbach-pm-machines-review.md
  - ./ironless-qdd-actuator.md
sources:
  - ../../sources/papers/mallinson_one_sided_fluxes_1973.md
summary: "Mallinson 1973：平面结构单侧磁通磁化图案一手文；常幅旋转磁化使一侧聚磁、另一侧场消；磁带记录语境，后世平面 Halbach 直觉来源。"
---

# One-sided fluxes — A magnetic curiosity?（Mallinson 1973）

## 一句话定义

**J. C. Mallinson（Ampex，[IEEE Trans. Magn. 1973](https://doi.org/10.1109/TMAG.1973.1067714)）** 证明存在一类 **平面磁化图案**，使磁通几乎只从一侧表面逃逸——常幅旋转磁化矢量是最简情形；这是后世 **平面 Halbach / 单侧聚磁** 讨论的学术一手来源。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PM | Permanent Magnet | 永磁体 / 磁化介质 |
| IEEE | Institute of Electrical and Electronics Engineers | 发表期刊所属学会 |
| TMAG | IEEE Transactions on Magnetics | 本文期刊 |
| Halbach | Halbach Array | 后世对旋转磁化阵列的通称 |
| OA | Open Access | 本文非 OA |

## 为什么重要

- **先于** Halbach 圆柱配方，把「一侧有场、一侧几乎没有」写成可引用定理级叙述。
- 给出 DIY 最常用的直觉图景：两套正交磁化分布叠加 → 一侧相加、一侧相消。
- 机器人/关节语境读它，是为了分清 **平面单侧** 与 **圆柱孔径多极** 不是同一篇一手文献。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 安派克斯（Ampex） |
| **形态** | 平面磁化结构 / 磁带记录物理 |
| **关键机制** | 常幅旋转磁化矢量；旋转方向决定无场侧 |
| **开源** | **不适用**（理论论文；非 OA） |

## 方法

- 解析/概念推导：构造使一侧法向磁通为零的磁化分布族。
- 讨论磁带写、接触印刷、透印中可能已部分出现该效应。

## 实验与评测

- 以理论与记录物理讨论为主；**不是**电机台架论文。
- 结论导向：若能增强单侧效应，磁带性能可显著改善（作者当时动机）。

## 与其他工作对比

| 工作 | 关系 |
|------|------|
| [Halbach 1980](./paper-halbach-permanent-multipole-magnets.md) | 圆柱/多极、易轴配方与分段；材料外场可为零 |
| [Zhu & Howe 2001](./paper-zhu-howe-halbach-pm-machines-review.md) | 把旋转磁化搬到电机转子与应用 |
| DIY 分段样机 | [Ironless QDD](./ironless-qdd-actuator.md) 用 90° 步进逼近 |

## 结论

**总判：** 这是「单侧磁通」概念的一手平面文献；读 Halbach 阵列应从这里建立平面直觉，再进圆柱配方与电机综述。

- 记住：**旋转磁化方向**决定聚磁侧。
- 工程离散化（90° 磁钢）是逼近，不是定理假设本身。
- 勿用本文指标直接估电机 \(K_t\)。
- 与 Halbach 1980 **互补**：平面 vs 圆柱/多极。
- 引用时写 DOI `10.1109/TMAG.1973.1067714`，避免只写「Halbach 发明了单侧磁通」。

## 源码运行时序图

**不适用**（理论/记录物理论文，无可运行代码或制造仓）。

## 工程实践

| 项 | 说明 |
|----|------|
| 阅读顺序 | 本文 → Halbach 1980 → Zhu & Howe 2001 → Ironless FEMM |
| 获取全文 | IEEE Xplore（付费）；本库只存 DOI 与摘录 |

## 局限与风险

- **非 OA**；摘要级 ingest 不足以替代精读公式细节。
- 应用语境是磁带，不是关节电机——迁移需改几何与材料假设。

## 关联页面

- [Halbach Array 概念](../concepts/halbach-array.md)
- [Halbach 1980](./paper-halbach-permanent-multipole-magnets.md) · [Zhu & Howe 2001](./paper-zhu-howe-halbach-pm-machines-review.md)

## 参考来源

- [sources/papers/mallinson_one_sided_fluxes_1973.md](../../sources/papers/mallinson_one_sided_fluxes_1973.md)

## 推荐继续阅读

- DOI：<https://doi.org/10.1109/TMAG.1973.1067714>
- Halbach 1980 OA：<https://escholarship.org/content/qt20b829tr/qt20b829tr.pdf>
