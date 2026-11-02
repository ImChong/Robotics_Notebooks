---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2504.13621"
venue: "ICCV 2025"
summary: "Introduces the first dataset for egocentric visual intention grounding, challenging multimodal LLMs to understand and ignore unintended contextual objects and reason about uncommon object functionalities"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2504_13621_egointention-visual-intention-grounding.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoIntention

**EgoIntention: Visual Intention Grounding for Egocentric Assistants** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 056/249** 篇，分组 **25 Gaze Attention & Intention Prediction**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Introduces the first dataset for egocentric visual intention grounding, challenging multimodal LLMs to understand and ignore unintended contextual objects and reason about uncommon object functionalities

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Introduces the first dataset for egocentric visual intention grounding, challenging multimodal LLMs to understand and ignore unintended contextual objects and reason about uncommon object functionalities
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    subgraph DS["EgoIntention 数据集构建 (§3.2)"]
        A1["Ego4D 图像 + PACO 物体框"] --> A2["GPT-4 生成意图句<br/>context-aware / uncommon 两类"]
        A2 --> A3["MTurk 人工校验语义有效性<br/>过滤低质量或歧义句子"]
        A3 --> A4["MTurk 补充替代物体与边界框"]
    end
    subgraph ROG["Reason-to-Ground (RoG) 指令微调 (§4.2–4.3)"]
        B1["第一人称图像 I + 隐含意图句 Q"] --> B2["阶段一：以 reason token 提问<br/>MLLM 输出目标物体类别"]
        B2 --> B3["阶段二：以 ref token + 显式物体描述<br/>MLLM 定位目标"]
        B3 --> B4["输出边界框 (x1, y1, x2, y2)"]
    end
    A4 -->|"与 RefCOCO / RefCOCO+ / RefCOCOg 联合"| FT["LoRA 监督微调 MLLM"]
    FT --> B1
```

图依据：论文 §3.1–3.2 / Fig.3、§4.2–4.3 / Fig.4（“微调后的模型用于推理”的箭头为据 §4.3 的示意）

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 056/249 |
| 分组 | 25 Gaze Attention & Intention Prediction |
| 出处 | ICCV 2025 |
| 论文 | <https://arxiv.org/abs/2504.13621> |

## 核心机制（归纳）

### 策展导读要点

Introduces the first dataset for egocentric visual intention grounding, challenging multimodal LLMs to understand and ignore unintended contextual objects and reason about uncommon object functionalities

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2504.13621) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **25 Gaze Attention & Intention Prediction** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoIntention」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **25 Gaze Attention & Intention Prediction**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2504_13621_egointention-visual-intention-grounding.md`](../../sources/papers/sun_awesome_ego_2504_13621_egointention-visual-intention-grounding.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2504.13621>
- [arXiv PDF](https://arxiv.org/pdf/2504.13621)：已通读方法部分（实验与附录以略读为准）：摘要、§1、§3.1–3.2、§4 Method、§5.1–5.4 开头

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2504.13621)
