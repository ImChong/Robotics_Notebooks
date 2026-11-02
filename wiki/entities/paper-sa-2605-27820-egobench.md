---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2605.27820"
summary: "策展清单收录条目；细节以原文 PDF / 项目页为准。"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2605_27820_egobench.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoBench

**EgoBench** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 201/249** 篇，分组 **Embodied VLA Evaluation Benchmarks**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

策展清单收录条目；细节以原文 PDF / 项目页为准。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- 策展清单收录条目；细节以原文 PDF / 项目页为准。
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    S1["Stage 1：第一人称视频收集<br/>自采（智能眼镜/手机）+ Ego4D，10–30 s，物体特征鲜明且有时间顺序"] --> S2["Stage 2：工具库与数据库构建<br/>数据库含视频可见属性 + 视频不可见的补充信息（视觉-数据信息差）"]
    S2 --> S3["Stage 3：任务设计与真值标注<br/>多模态引用 → 工具检索不可见属性 → 条件分支与筛选 → 修改数据库状态<br/>真值 = 关键工具调用序列，两名标注员独立标注"]
    S3 --> ENV["评测环境：被测智能体 + 工具/数据库 + 模拟用户"]
    subgraph SU["模拟用户 (Actor-Evaluator-Summarizer)"]
        ACT["Actor 生成用户回复"] --> EVA["Evaluator 按四项二值准则检查"]
        EVA -->|"不通过：反馈修订"| ACT
        EVA -->|"通过"| SUM["Summarizer 压缩对话并记录任务进度"]
        SUM --> ACT
    end
    ENV <--> SU
    ENV --> JV["联合验证<br/>过程：真值工具调用是否被覆盖<br/>结果：最终数据库状态哈希是否等价"]
    JV --> JSR["Joint Success Rate"]
```

图依据：论文 §3.1–§3.5 / Fig. 2；本图描述基准构建与评测流程，非被测模型的方法

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 201/249 |
| 分组 | Embodied VLA Evaluation Benchmarks |
| 出处 | 见清单 / 原文 |
| 论文 | <https://arxiv.org/abs/2605.27820> |

## 核心机制（归纳）

### 策展导读要点

策展清单收录条目；细节以原文 PDF / 项目页为准。

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2605.27820) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Embodied VLA Evaluation Benchmarks** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoBench」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Embodied VLA Evaluation Benchmarks**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
- 适用边界：本页不能替代原文 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 要深读这篇，建议直接从原文入手，再回到下方关联的方法 / 任务页对照。

## 常见误区

1. 不要把 Awesome 条目的 Highlights 当成完整方法证明——它只是策展导读。
2. 若站内已有这篇的深读页，以那一页为准——本页只是清单入口，不含实验数据。

## 关联页面

- 列表实体：[Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md)
- 技术地图：[Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md)
- 方法/任务：[vla.md](../methods/vla.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/sun_awesome_ego_2605_27820_egobench.md`](../../sources/papers/sun_awesome_ego_2605_27820_egobench.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2605.27820>
- [arXiv PDF](https://arxiv.org/pdf/2605.27820)：已通读方法部分（实验与附录以略读为准）：摘要、§1、§3.1–§3.5

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2605.27820)
