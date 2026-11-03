---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2308.09126"
venue: "NeurIPS 2023"
summary: "Authoritative diagnostic benchmark for ultra-long egocentric video understanding"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2308_09126_egoschema-a-diagnostic-benchmark-for-ver.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# EgoSchema

**EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 038/249** 篇，分组 **24 Long-form Video Understanding & Structured Reasoning**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Authoritative diagnostic benchmark for ultra-long egocentric video understanding

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Authoritative diagnostic benchmark for ultra-long egocentric video understanding
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    S1["阶段 I 原始数据过滤<br/>Ego4D 视频 + 人工叙述<br/>取不重叠 3 分钟片段且至少 30 条叙述"] --> S2
    S2["阶段 II 问答生成<br/>LLM 以 Q(AW)-shot 两次链式调用<br/>先生成问题 Q 再生成正确答案 A 与错误选项 W<br/>每片段 N=3 个问题 每题 M=4 个错误选项"] --> S3
    subgraph S3["阶段 III 生成结果过滤"]
        direction TB
        R["规则过滤<br/>丢弃泄漏关键词或格式解析失败的 QAW"] --> L["LLM 盲基线过滤<br/>不看叙述仅凭问题能答对的题被剔除"]
    end
    S3 --> S4["阶段 IV 人工两轮校验<br/>核对 Q 与 A 正确 干扰项确为错误<br/>并确认 temporal certificate 至少 30 秒"]
    S4 --> DATA["EgoSchema 长视频多选问答数据集"]
```

图依据：论文 §3.1 EgoSchema Pipeline（阶段 I 至 IV）、§3.2 Temporal Certificates / Fig.4

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 038/249 |
| 分组 | 24 Long-form Video Understanding & Structured Reasoning |
| 出处 | NeurIPS 2023 |
| 论文 | <https://arxiv.org/abs/2308.09126> |

## 核心机制（归纳）

### 策展导读要点

Authoritative diagnostic benchmark for ultra-long egocentric video understanding

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2308.09126) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **24 Long-form Video Understanding & Structured Reasoning** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「EgoSchema」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **24 Long-form Video Understanding & Structured Reasoning**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2308_09126_egoschema-a-diagnostic-benchmark-for-ver.md`](../../sources/papers/sun_awesome_ego_2308_09126_egoschema-a-diagnostic-benchmark-for-ver.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2308.09126>
- [arXiv PDF](https://arxiv.org/pdf/2308.09126)：已通读方法部分（实验与附录以略读为准）：§3 EgoSchema 数据构造流程（§3.1 四阶段、§3.2 temporal certificate）；基准评测与附录仅略读

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2308.09126)
