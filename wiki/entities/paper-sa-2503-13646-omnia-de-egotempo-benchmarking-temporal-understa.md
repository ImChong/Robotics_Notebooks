---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2503.13646"
venue: "CVPR 2025"
summary: "Benchmark for temporal reasoning ability of multimodal LLMs on egocentric procedural videos"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2503_13646_omnia-de-egotempo-benchmarking-temporal.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Omnia de Egotempo

**Omnia de Egotempo: Benchmarking Temporal Understanding of Multi-Modal LLMs in Egocentric Videos** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 143/249** 篇，分组 **293 Procedural Question Answering**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Benchmark for temporal reasoning ability of multimodal LLMs on egocentric procedural videos

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Benchmark for temporal reasoning ability of multimodal LLMs on egocentric procedural videos
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    SRC["Ego4D 视频 + 逐句 narration {(Nj, tj)}（第一人称自由文本，#C 指拍摄者）"] --> WIN["narration 时间窗 Tj：沿用 EgoVLP 思路，以 tj 为中心、宽度由相邻 narration 平均间隔 βi 与全局均值 α 决定"]
    WIN --> CLIP["合并相邻片段成 video clip：最多 120 条 narration 或 120 秒，先到者为准"]
    TAX["10 类时序能力（§3.1）：动作 5 类（Action Sequence / Action Counting / Temporal Event Ordering / Future Action Prediction / Object-Specific Actions），物体 5 类（Object Sequence / Object Counting / Spatial Relations / Locating Objects / Action-Specific Objects）"]
    CLIP --> CAP["Gemini 1.5 Pro 零样本：用 narration 生成 caption（侧重用户交互）"]
    CAP --> QAG["Gemini 1.5 Pro：依 caption 与视频、按 10 类生成 Q&A"]
    TAX --> QAG
    QAG --> REV["人工审核：问题需多帧信息、问答逻辑一致；剔除缺时序复杂度或内容无关的，以及 Gemini 单帧即可答对的"]
    REV --> BENCH["EgoTempo：500 条开放式 QA（每类 50 条），365 个 clip，来自 221 个视频"]
    BENCH --> PRED["被测 MLLM：输入视频与问题，生成开放式预测"]
    PRED --> JUDGE["LLM 评审（Gemini 1.5 Pro）：对照标准答案判对错并给理由"]
    JUDGE --> ACC["按类别与总体统计准确率"]
```

图依据：数据采集与标注流程图（数据集构建 + LLM 评审评测协议）：论文 §3.1 / §3.2（Fig. 2、Table 2）、§4.1（Evaluation Protocol）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 143/249 |
| 分组 | 293 Procedural Question Answering |
| 出处 | CVPR 2025 |
| 论文 | <https://arxiv.org/abs/2503.13646> |

## 核心机制（归纳）

### 策展导读要点

Benchmark for temporal reasoning ability of multimodal LLMs on egocentric procedural videos

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2503.13646) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **293 Procedural Question Answering** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Omnia de Egotempo」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **293 Procedural Question Answering**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2503_13646_omnia-de-egotempo-benchmarking-temporal.md`](../../sources/papers/sun_awesome_ego_2503_13646_omnia-de-egotempo-benchmarking-temporal.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2503.13646>
- [arXiv PDF](https://arxiv.org/pdf/2503.13646)：已通读方法部分（实验与附录以略读为准）：摘要、§1–§3（含 Fig. 1–3、Table 1–2）、§4.1 评测设置；§4.2 起的结果表、消融与附录未细读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2503.13646)
