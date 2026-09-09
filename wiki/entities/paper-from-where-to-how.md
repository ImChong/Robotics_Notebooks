---
type: entity
tags: ['paper', 'egocentric', 'forecasting', 'assistive-robotics']
status: complete
updated: 2026-09-09
arxiv: "2609.08636"
venue: "arXiv 2026"
related:
  - ../overview/visual-focus-data-efficiency-10-papers-technology-map.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/papers/from_where_to_how_arxiv_2609_08636.md
  - ../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md
summary: "Coherent4D + HIGFlow（arXiv:2609.08636）：233K 样本连续 4D 交互预测数据集；先预测未来 3D 手部位点再条件化 residual flow matching 生成全身姿态。"
---

# Coherent4D / HIGFlow

**Coherent4D / HIGFlow**（*Continuous 4D Interaction Forecasting from Egocentric Video*，[arXiv:2609.08636](https://arxiv.org/abs/2609.08636)，[项目/代码](https://corrineqiu.github.io/from-where-to-how/)）— 详见 [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)。

## 一句话定义

辅助机器人需要同时知道「在哪互动」和「身体怎么动」——本文用共享坐标系下的连续 where-to-how 级联，而不是离散网格或独立姿态生成。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| HIGFlow | Hand-Interaction-Guided Residual Flow | 本文预测框架 |
| SMPL | Skinned Multi-Person Linear | 全身姿态参数化 |
| ADE | Average Displacement Error | 交互位置误差 mm |
| MPJPE | Mean Per Joint Position Error | 姿态误差 mm |

## 为什么重要

- Coherent4D：233,828 样本，Cooking/Health/Bike Repair 三域
- Stage1 Qwen3-VL + V-JEPA 预测连续手部位点；Stage2 flow matching 姿态

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.08636](https://arxiv.org/abs/2609.08636) |
| **开源** | **待核实** |
| **项目/代码** | [https://corrineqiu.github.io/from-where-to-how/](https://corrineqiu.github.io/from-where-to-how/) |

## 核心原理

- Coherent4D：233,828 样本，Cooking/Health/Bike Repair 三域
- Stage1 Qwen3-VL + V-JEPA 预测连续手部位点；Stage2 flow matching 姿态
- 三域 interaction location 与 pose 指标均优于 FIction 等基线

## 源码运行时序图

**不适用（官方可运行代码尚未发布或待核实）。** 截至 2026-09-09 以项目页/公众号链为准。

## 实验与评测

- 指标与设置以原文 PDF / 项目页为准；上文 Highlights 来自公众号归纳 + 项目页摘要。
- 横向对照见 [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)。

## 结论

**Coherent4D / HIGFlow 的可迁移主张已写入 Highlights；部署前以原文实验设定与开源边界为准。**

1. **真影响：** 见核心原理 bullets。
2. **次要代价：** 预印本/待开源项需独立复现验证。
3. **部署读法：** 待核实 — 先读 README 或项目页再接真机/智能体栈。

## 关联页面

- [视觉聚焦与数据效率 10 篇技术地图](../overview/visual-focus-data-efficiency-10-papers-technology-map.md)
- [模仿学习](../methods/imitation-learning.md)

## 参考来源

- [from_where_to_how_arxiv_2609_08636.md](../../sources/papers/from_where_to_how_arxiv_2609_08636.md)
- [具身智能小站 10 篇盘点（2026-09-09）](../../sources/blogs/wechat_embodied_station_visual_focus_10_papers_2026-09-09.md)
- [arXiv:2609.08636](https://arxiv.org/abs/2609.08636)

## 推荐继续阅读

- [原文 PDF](https://arxiv.org/pdf/2609.08636)
- [项目/代码](https://corrineqiu.github.io/from-where-to-how/)
