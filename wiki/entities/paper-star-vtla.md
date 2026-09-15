---
type: entity
tags:
  - paper
  - tactile
  - dexterous
  - vla
  - manipulation
status: complete
updated: 2026-09-15
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

## 与其他工作对比

- **纯视觉 VLA** — 遮挡发生在指尖时，视觉就是瞎的；STAR 把触觉接进同一套 VLA 接口（视触联合预训练 + 稀疏全局触觉 token + 稀疏未来触觉预测），属 [视触融合](../concepts/visuo-tactile-fusion.md) 在策略侧的落地，而不是只做感知层融合。
- **稠密触觉序列输入** — 逐点逐帧喂触觉会把上下文长度打爆；STAR 选 **稀疏全局 token**，用少量维度概括接触分布，代价是丢掉细粒度接触位置。这个取舍是本页与一般 [触觉传感](../concepts/tactile-sensing.md) 方案最关键的分歧点。
- **[ArtManip](./paper-artmanip.md)（同批）** — 同为灵巧接触任务，路线互为镜像：ArtManip 不加传感，靠 **仿真程序化分布 + 特权蒸馏 + 课程** 求 **铰接 in-hand** 泛化（[ArtGym](https://github.com/youngcv/artgym) 已开源）；STAR 不改仿真，靠 **200 小时真机视触数据 + 模型配方** 求泛化。
- **[Gripper MagBot](./paper-gripper-magbot.md)（同批）** — MagBot 把末端自由度压到 1-DoF 求低成本，STAR 在双臂灵巧手上加模态求精细度；本期「末端该加什么、减什么」的两端。
- **[StarVLA](../methods/star-vla.md)（同名不同工作）** — 注意区分：那条线是「强 VLM 底座 + 简单 MLP 动作头」的极简 VLA 基准，与本页的视触语动作（VTLA）配方无关，只是名字相近。

- **读法：** 以上为知识库内 **路线级** 对照；与原文 baseline 的逐项定量比较以 **原文 PDF** 为准（[参考来源](#参考来源)）。开源状态为 **待发布**，数据集与权重暂不可直接复现。

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
