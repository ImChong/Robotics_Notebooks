---
type: entity
tags:
  - paper
  - tactile
  - dexterous
  - vla
  - manipulation
status: complete
updated: 2026-09-14
arxiv: "2609.12549"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ./paper-artmanip.md
  - ../overview/vla-tamp-planning-11-papers-technology-map.md
sources:
  - ../../sources/papers/star-vtla_arxiv_2609_12549.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md
summary: "200 小时双臂灵巧数据集 + VTLA 三部分配方：视触联合预训练、稀疏全局触觉 token、稀疏未来触觉预测；四任务平均 61% 成功率。"
---

# STAR（arXiv:2609.12549）

**STAR**（[STAR: Sparse Tactile Representation Learning in Vision-Tactile-Language-Action Models for Dexterous Manipulation](https://arxiv.org/abs/2609.12549)）来自 [具身智能小站 11 篇 VLA/TAMP 盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)。灵巧手触觉空间稀疏、时间断续；STAR 把稀疏触觉变成可用的未来接触信号。

## 一句话定义

**200 小时双臂灵巧数据集 + VTLA 三部分配方：视触联合预训练、稀疏全局触觉 token、稀疏未来触觉预测。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| TAMP | Task and Motion Planning | 任务与运动规划 |
| OOD | Out-of-Distribution | 分布外泛化评测 |

## 为什么重要

- 灵巧手触觉空间稀疏、时间断续；STAR 把稀疏触觉变成可用的未来接触信号。
- 开源状态：**待发布**（步骤 2.5 核查，2026-09-14）。
- 与 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.12549](https://arxiv.org/abs/2609.12549) |
| **项目页** | https://stardex-web.github.io/Star/ |
| **开源** | **待发布** |
| **文内指标** | 10,576 轨迹 / 65 任务；每任务 100 条后训练轨迹，四真实任务平均成功率 61%。 |


## 源码运行时序图

**不适用**（截至 2026-09-14 项目页未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 要点 | 10,576 轨迹 / 65 任务；每任务 100 条后训练轨迹，四真实任务平均成功率 61%。 |

- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准。

## 结论

**STAR 适合作为本期「待发布」边界下的快速索引页，部署前请核对项目页/仓库可运行性。**

1. 核心贡献：灵巧手触觉空间稀疏、时间断续；STAR 把稀疏触觉变成可用的未来接触信号。
2. 开源结论：**待发布** — 以项目页实际链接为准（入库日 2026-09-14）。
3. 横向对照见 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)
- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [star-vtla_arxiv_2609_12549.md](../../sources/papers/star-vtla_arxiv_2609_12549.md)
- [wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)
- [arXiv:2609.12549](https://arxiv.org/abs/2609.12549)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.12549)
- [项目页](https://stardex-web.github.io/Star/)
