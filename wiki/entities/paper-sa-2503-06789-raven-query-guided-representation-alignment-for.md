---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2505.17114"
venue: "arXiv 2025"
summary: "Multi-modal QA framework aligning audio, video, embedded sensors and natural language for procedural scenarios"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2503_06789_raven-query-guided-representation-alignm.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# RAVEN

**RAVEN: Query-Guided Representation Alignment for Question Answering over Audio, Video, Embedded Sensors, and Natural Language** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 144/249** 篇，分组 **293 Procedural Question Answering**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Multi-modal QA framework aligning audio, video, embedded sensors and natural language for procedural scenarios

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Multi-modal QA framework aligning audio, video, embedded sensors and natural language for procedural scenarios
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
  V["视频帧"] --> V1["SigLIP-so-400m 视觉编码器"]
  A["音频 (Kaldi-fbank)"] --> A1["BEATs 音频编码器"]
  S["IMU 传感器流"] --> S1["LIMU-BERT 传感器编码器"]
  V1 --> VP["投影层 → 共享空间"]
  A1 --> AP["投影层 → 共享空间"]
  S1 --> SP["投影层 → 共享空间"]
  VP --> Z["拼接为统一 token 矩阵 Z"]
  AP --> Z
  SP --> Z
  Q["文本问题 Q → Qwen2-7B-Instruct 词嵌入 zq"] --> QU
  Z --> QU["QuART: zq 与 Z 做多头注意力 → 输出 M"]
  QU --> R["相关性投影 WR: α = softmax(M WR)"]
  R --> C["融合上下文 C = Σ αj Zj"]
  C --> L["LLM 解码器 Π (LoRA)"]
  L --> ANS["回答"]
  subgraph T["三阶段训练"]
    direction LR
    T1["Stage I: 冻结编码器与 LLM, 仅训各模态投影头"] --> T2["Stage II: 训练 QuART + LoRA, 冻结编码器与投影"]
    T2 --> T3["Stage III: 扰动输入 (丢帧/噪声等) + 熵正则 λ=0.001"]
  end
```

图依据：论文 §4（Modality-Specific Feature Encoders、QuART）与 §5（三阶段训练）、Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 144/249 |
| 分组 | 293 Procedural Question Answering |
| 出处 | arXiv 2025 |
| 论文 | <https://arxiv.org/abs/2505.17114> |

## 核心机制（归纳）

### 策展导读要点

Multi-modal QA framework aligning audio, video, embedded sensors and natural language for procedural scenarios

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2505.17114) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **293 Procedural Question Answering** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「RAVEN」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

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

- [`sources/papers/sun_awesome_ego_2503_06789_raven-query-guided-representation-alignm.md`](../../sources/papers/sun_awesome_ego_2503_06789_raven-query-guided-representation-alignm.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2505.17114>
- [arXiv PDF](https://arxiv.org/pdf/2505.17114)：已通读方法部分（实验与附录以略读为准）：§4 RAVEN 方法（特征编码器、QuART、训练目标）与 §5 三阶段训练；§3 数据集构建（Evaluator/Critic）、§6 实验与附录略读或未读
- 编号核对（2026-11-02）：上游清单登记的 2503.06789 经 arXiv 官方记录核对为《Aharonov-Bohm and Altshuler-Aronov-Spivak oscillations in the quasi-ballistic regime in phase-pure GaAs/InAs core/shell nanowires》，与本页论文不符；已更正为 2505.17114《RAVEN: Query-Guided Representation Alignment for Question Answering over Audio, Video, Embedded Sensors, and Natural Language》。

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2505.17114)
