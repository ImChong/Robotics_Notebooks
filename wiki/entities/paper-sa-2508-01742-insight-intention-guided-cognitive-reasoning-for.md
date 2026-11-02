---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2508.01742"
venue: "AAAI 2026"
summary: "Proposes a two-stage framework of 'HOI semantic recognition + intention reasoning' for long-term action anticipation"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2508_01742_insight-intention-guided-cognitive-reaso.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# INSIGHT

**INSIGHT: Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 155/249** 篇，分组 **Action Anticipation**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Proposes a two-stage framework of "HOI semantic recognition + intention reasoning" for long-term action anticipation

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Proposes a two-stage framework of "HOI semantic recognition + intention reasoning" for long-term action anticipation
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    A["观测视频段 Sk（每段均匀采样 4 帧）"] --> B["HOI 区域检测：100DOH 检测器 + SAM2 精细掩码"]
    A --> C["双流 EgoVideo-V：编码整帧与 HOI 区域"]
    B --> C
    C --> D["拼接后经线性 MLP 融合，输入 Transformer 识别模块"]
    D --> E["动词 / 名词两个并行分类器"]
    E --> F["动词-名词共现矩阵校正联合概率，MAP 得到观测动作 (v, n)"]
    F --> G["阶段 2：把视觉特征与观测动作写入 VLM 提示"]
    G --> H["GRPO 训练的 VLM 按 think → intention → answer 结构化输出"]
    H --> I["预测未来 Z 个动词-名词动作"]
    J["奖励：长度、标签顺序、语言一致、软超长惩罚；准确率（编辑距离）与意图奖励（Sentence-BERT 余弦）"] -.-> H
```

图依据：论文 Method（Hand-Object Semantic Action Recognition、Explicit Cognitive Reasoning for Anticipation）/ Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 155/249 |
| 分组 | Action Anticipation |
| 出处 | AAAI 2026 |
| 论文 | <https://arxiv.org/abs/2508.01742> |

## 核心机制（归纳）

### 策展导读要点

Proposes a two-stage framework of "HOI semantic recognition + intention reasoning" for long-term action anticipation

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2508.01742) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **Action Anticipation** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「INSIGHT」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **Action Anticipation**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2508_01742_insight-intention-guided-cognitive-reaso.md`](../../sources/papers/sun_awesome_ego_2508_01742_insight-intention-guided-cognitive-reaso.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2508.01742>
- [arXiv PDF](https://arxiv.org/pdf/2508.01742)：已通读方法部分（实验与附录以略读为准）：Method 全部小节（含 HOI 增强特征、共现校正、GRPO 奖励设计）

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2508.01742)
