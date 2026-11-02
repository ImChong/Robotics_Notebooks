---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2405.17719"
venue: "ICLR 2025"
summary: "Systematically evaluates the fine-grained HOI understanding ability of egocentric VLMs for the first time, proposing the EgoHOIBench benchmark"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2405_17719_do-egocentric-video-language-models-trul.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Do Egocentric Video-Language Models Truly Understand Hand-Object Interactions?

**Do Egocentric Video-Language Models Truly Understand Hand-Object Interactions?** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 010/249** 篇，分组 **22 Hand-Object Interaction & Fine-grained Activity**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Systematically evaluates the fine-grained HOI understanding ability of egocentric VLMs for the first time, proposing the EgoHOIBench benchmark

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Systematically evaluates the fine-grained HOI understanding ability of egocentric VLMs for the first time, proposing the EgoHOIBench benchmark
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 010/249 |
| 分组 | 22 Hand-Object Interaction & Fine-grained Activity |
| 出处 | ICLR 2025 |
| 论文 | <https://arxiv.org/abs/2405.17719> |

## 流程总览

```mermaid
flowchart TD
  subgraph BM["EgoHOIBench 评测 (§2.1)"]
    B1["视频片段 + 正确 caption"] --> B2["LLM 替换 HOI 动词或名词<br/>生成困难负例 caption"]
    B2 --> B3["多选视频-文本匹配<br/>HOI-verb 与 HOI-noun 均选对才算成功"]
  end
  subgraph TRN["EgoNCE++ 继续预训练 (§3)"]
    D1["EgoClip 视频 + 文本"] --> V["视频编码器 LoRA 微调"]
    D1 --> TX["文本编码器冻结, 仅 word embedding 可训"]
    D1 --> NG["HOI 负例生成<br/>预训练词表 spaCy 替换, 或 LLM"]
    V --> LV["V2T 损失: 真实 caption 对比 batch 易负例 + 词替换难负例"]
    TX --> LV
    NG --> LV
    V --> LT["T2V 损失: 仅以同名词视频为正例, 保持 object-centric 特征空间"]
    TX --> LT
    LV --> LS["L = L_t2v + L_v2t"]
    LT --> LS
  end
  LS --> OUT["增强后的 EgoVLM (EgoVLP / EgoVLPv2 / LaViLa)"]
  OUT -.->|"零样本评测"| B3
```

图依据：论文 §2.1–2.3、§3.1–3.3 / Fig.2–4。

## 核心机制（归纳）

### 策展导读要点

Systematically evaluates the fine-grained HOI understanding ability of egocentric VLMs for the first time, proposing the EgoHOIBench benchmark

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2405.17719) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **22 Hand-Object Interaction & Fine-grained Activity** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Do Egocentric Video-Language Models Truly Understand Hand-Object Interactions?」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **22 Hand-Object Interaction & Fine-grained Activity**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2405_17719_do-egocentric-video-language-models-trul.md`](../../sources/papers/sun_awesome_ego_2405_17719_do-egocentric-video-language-models-trul.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2405.17719>
- arXiv PDF：<https://arxiv.org/pdf/2405.17719>（已通读全文：方法 §2–3、实验 §4.1）

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2405.17719)
