---
type: entity
tags:
  - paper
  - dexterous
  - manipulation
  - rl
status: complete
updated: 2026-09-14
arxiv: "2609.12498"
related:
  - ../tasks/manipulation.md
  - ./paper-star-vtla.md
  - ../overview/vla-tamp-planning-11-papers-technology-map.md
sources:
  - ../../sources/papers/artmanip_arxiv_2609_12498.md
  - ../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md
summary: "程序化物体与功能抓取生成、关节物理随机化、奖励课程与潜表示蒸馏；四类物体仿真泛化与 12 个真实物体零样本迁移。"
---

# ArtManip（arXiv:2609.12498）

**ArtManip**（[ArtManip: Category-Level Articulated In-Hand Manipulation](https://arxiv.org/abs/2609.12498)）来自 [具身智能小站 11 篇 VLA/TAMP 盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)。铰接物体内部自由度与接触动力学耦合，换同类物体易失效；ArtManip 训练类别级 in-hand 策略。

## 一句话定义

**程序化物体与功能抓取生成、关节物理随机化、奖励课程与潜表示蒸馏。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| TAMP | Task and Motion Planning | 任务与运动规划 |
| OOD | Out-of-Distribution | 分布外泛化评测 |

## 为什么重要

- 铰接物体内部自由度与接触动力学耦合，换同类物体易失效；ArtManip 训练类别级 in-hand 策略。
- 开源状态：**待发布**（步骤 2.5 核查，2026-09-14）。
- 与 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.12498](https://arxiv.org/abs/2609.12498) |
| **项目页** | https://artmanip.github.io/ |
| **开源** | **待发布** |
| **文内指标** | 四类物体仿真泛化；12 个真实物体零样本迁移。 |


## 源码运行时序图

**不适用**（截至 2026-09-14 项目页未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 要点 | 四类物体仿真泛化；12 个真实物体零样本迁移。 |

- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

- **[STAR](./paper-star-vtla.md)（同批，视触语动作）** — 同样面向灵巧接触任务，但 STAR 的抓手是 **新增触觉模态 + 200 小时真机数据**；ArtManip 不动传感配置，泛化全押在 **仿真侧的程序化物体生成与关节物理随机化** 上。
- **[SCQ](./paper-scq-rl.md)（同批，离线 RL）** — 两条都是 RL 路线，但分工不同：SCQ 改的是 **算法层**（保守 Q 学习的熵项稳定性），ArtManip 改的是 **环境与课程层**（物体/抓取生成、奖励课程、潜表示蒸馏）。
- **[UniPart](./paper-unipart.md)（同批，3D 部件分割）** — 与本页互补而非竞争：UniPart 负责「语言指到哪个部件」，ArtManip 负责「指到之后手内怎么转」；一个是感知接地，一个是接触内控制。
- **[域随机化](../concepts/domain-randomization.md) / [课程学习](../concepts/curriculum-learning.md)** — ArtManip 属这两条通用机制在 **铰接物体 in-hand** 场景的组合应用：随机化打的是关节物理参数，课程打的是接触任务的难度爬升。
- **[接触丰富操作](../concepts/contact-rich-manipulation.md)** — 该页给出接触耦合为何难的机制层解释；ArtManip 是其中「类别级泛化」这一支的具体实例。

- **读法：** 以上为知识库内 **路线级** 对照；与原文 baseline 的逐项定量比较以 **原文 PDF** 为准（[参考来源](#参考来源)）。开源状态为 **待发布**，暂无法按代码口径复现对照。

## 结论

**ArtManip 适合作为本期「待发布」边界下的快速索引页，部署前请核对项目页/仓库可运行性。**

1. 核心贡献：铰接物体内部自由度与接触动力学耦合，换同类物体易失效；ArtManip 训练类别级 in-hand 策略。
2. 开源结论：**待发布** — 以项目页实际链接为准（入库日 2026-09-14）。
3. 横向对照见 [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [11 篇技术地图](../overview/vla-tamp-planning-11-papers-technology-map.md)
- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [artmanip_arxiv_2609_12498.md](../../sources/papers/artmanip_arxiv_2609_12498.md)
- [wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md](../../sources/blogs/wechat_embodied_station_11_papers_vla_tamp_planning_2026-09-14.md)
- [arXiv:2609.12498](https://arxiv.org/abs/2609.12498)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.12498)
- [项目页](https://artmanip.github.io/)
