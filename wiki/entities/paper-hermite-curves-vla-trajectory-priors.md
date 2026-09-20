---
type: entity
tags:
  - paper
  - vla
  - trajectory
  - manipulation
status: complete
updated: 2026-09-19
arxiv: "2608.01265"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md
sources:
  - ../../sources/papers/hermite_curves_vla_trajectory_priors_arxiv_2608_01265.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md
summary: "用 Hermite 曲线作 VLA 轨迹先验：训练阶段引导动作接近平滑曲线即可提升成功率，部署无额外计算；对比直接预测/曲线修正等变体，训练期曲线引导最实用。"
---

# Hermite Curves VLA（arXiv:2608.01265）

**Hermite Curves VLA**（[arXiv:2608.01265](https://arxiv.org/abs/2608.01265)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第一篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) **架构模块** 段。

## 一句话定义

**用 Hermite 曲线作 VLA 轨迹先验：训练阶段引导动作接近平滑曲线即可提升成功率，部署无额外计算；对比直接预测/曲线修正等变体，训练期曲线引导最实用。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| TTT | Test-Time Training | 部署阶段无标注数据的在线适配 |

## 为什么重要

- 用 Hermite 曲线作 VLA 轨迹先验：训练阶段引导动作接近平滑曲线即可提升成功率，部署无额外计算；对比直接预测/曲线修正等变体，训练期曲线引导最实用。
- 开源状态：**待核实**（步骤 2.5，入库日 2026-09-19）。
- 与 [vla weekly trends 2026 08 10 part1 technology map](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md) 同批工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.01265](https://arxiv.org/abs/2608.01265) |
| **项目页** | https://aopolin-lv.github.io/Hermite |
| **开源** | **待核实** |
| **文内评测** | LIBERO、LIBERO-Plus；实机 Franka / Cybopal / ARX |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

- **文内口径：** LIBERO、LIBERO-Plus；实机 Franka / Cybopal / ARX
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

> 下表只做**定位对照**：本页与下列同批各页均为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 的索引级摘要，未逐条核对原文实验表，彼此**不共享同一评测协议**，不可据此横比数字。

| 对照 | 差异读法 |
|------|----------|
| [Cross-View Action Consistency](./paper-cross-view-action-consistency-vla.md) | 同批「架构模块」段，同属**训练期加约束、部署零额外计算**一族；差别在约束打在哪：Hermite 约束轨迹的**形状先验**（贴近平滑曲线），Cross-View 约束**跨视角的动作一致性** |
| [TDHD](./paper-tdhd-surgical-dual-arm.md) | 同批同为治理固定长度执行的累积偏差，手段一训一测：Hermite 训练期把动作引向平滑曲线，TDHD 执行期靠双计划分歧提前截断重规划。前者管不到运行时决策，后者不改训练目标 |
| [Action Chunking](../methods/action-chunking.md) | 曲线先验作用的正是 chunk 内那段轨迹；该页讲 chunk 为何在 open-loop 段漂移，本文给的是「让这段漂得更平滑」的一种约束形式 |
| [VLA](../methods/vla.md) | 论文自比了「直接预测曲线参数 / 曲线修正」等变体，结论是训练期引导最实用——即**不改推理接口**才是本条主张的关键，读法应与那些改动作头结构的路线分开 |

## 结论

**Hermite Curves VLA 适合作为本期「架构模块」路线的快速索引页。**

1. 核心贡献：用 Hermite 曲线作 VLA 轨迹先验：训练阶段引导动作接近平滑曲线即可提升成功率，部署无额外计算；对比直接预测/曲线修正等变体，训练期曲线引导最实用。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part1.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md)
- [arXiv:2608.01265](https://arxiv.org/abs/2608.01265)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.01265)
