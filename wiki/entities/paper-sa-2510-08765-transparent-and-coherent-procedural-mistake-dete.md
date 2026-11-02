---
type: entity
tags: [paper, curated-index, awesome-egocentric-vision, sun254667-ego]
status: complete
updated: 2026-10-06
arxiv: "2510.08765"
venue: "EMNLP 2025"
summary: "Generates transparent, coherent natural language explanations for detected procedural mistakes"
related:
  - ../entities/awesome-egocentric-vision.md
  - ../overview/sun-awesome-ego-technology-map.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/papers/sun_awesome_ego_2510_08765_transparent-and-coherent-procedural-mist.md
  - ../../sources/papers/sun_awesome_ego_catalog.md
  - ../../sources/repos/awesome-egocentric-vision.md
---

# Transparent and Coherent Procedural Mistake Detection

**Transparent and Coherent Procedural Mistake Detection** 收录于 [Awesome Egocentric Vision](https://github.com/sun254667/awesome-egocentric-vision) **第 125/249** 篇，分组 **291 Procedural Error Detection**。本页是 **清单索引**：给出它在清单中的位置与原文入口，方法细节和量化结果请看原文。

## 一句话定义

Generates transparent, coherent natural language explanations for detected procedural mistakes

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Ego | Egocentric Vision | 第一人称可穿戴视角感知 |
| HOI | Hand–Object Interaction | 手–物交互理解 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言模型 |

## 为什么重要

- Generates transparent, coherent natural language explanations for detected procedural mistakes
- 想横向对照同一分组的其他工作，可以从 [Awesome Egocentric Vision 技术地图](../overview/sun-awesome-ego-technology-map.md) 逐条展开。
- 顺着列表实体 [Awesome Egocentric Vision](../entities/awesome-egocentric-vision.md) 与下方关联的方法 / 任务页，可以接回对应的学习主线。

## 流程总览

```mermaid
flowchart TD
    IN["输入：程序文本 T 与单帧视频画面 F"] --> VQG["VQG：VLM 依据 T 与对话历史生成问题（beam search 得 4 个候选）"]
    ICL["可选：in-context learning，依据人工示例问题再生成 4 个候选"] -.-> SEL
    VQG --> SEL["候选选择：默认取最可能且未问过的；或按 相关性 × 潜在信息量 重排序（NLI 模型 BART + MNLI 打分）"]
    SEL --> VQA["VQA：VLM 仅依据问题与画面作答；答案概率超过 60% 记 Yes/No，否则记 Unsure"]
    VQA --> HIST["问答追加进对话历史"]
    HIST --> CLS["成功/错误分类：依据画面与完整对话，用置信阈值 τ 判定（该步不进入历史）"]
    CLS --> STOP{"早停 ？似然变化小于 δ 连续两轮，或高度确信，或已达 10 个问题"}
    STOP -->|"否"| VQG
    STOP -->|"是"| OUT["输出：成功/错误决策 y 与理由 R = (问题序列 Q, 答案序列 A)"]
    DPO["可选：用 DPO + QLoRA 微调 VQG 适配器，正例为排序最高的问题，负例从后半抽样，推理时仅在 VQG 启用"] -.-> VQG
```

图依据：论文 §2.2 / Fig.1，§4.1–§4.2（arXiv 编号与页面不符：页面 frontmatter 的 2510.08765 对应的是无关的 UAV/IoT 定位论文，经 arXiv 检索，《Transparent and Coherent Procedural Mistake Detection》的正确编号为 2412.11927）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 125/249 |
| 分组 | 291 Procedural Error Detection |
| 出处 | EMNLP 2025 |
| 论文 | <https://arxiv.org/abs/2510.08765> |

## 核心机制（归纳）

### 策展导读要点

Generates transparent, coherent natural language explanations for detected procedural mistakes

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内相关方法页（见关联页面）。

## 评测与指标

- 本页 **没有搬运** 原文的量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2510.08765) 为准。
- 横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 同分组条目。

## 与其他工作对比

- 本页 **不做** 与具体基线的逐项数值对比；同分组的横向对照请回到 [技术地图](../overview/sun-awesome-ego-technology-map.md) 的 **291 Procedural Error Detection** 分组逐条展开。
- 如果站内已经有这篇的深读页（含机构、实验表与源码运行时序图），请以那一页为准；本页只保留清单要点。
- 与清单内相邻条目孰优孰劣，本页不下结论：Awesome Highlights 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**这一页能给你的是「Transparent and Coherent Procedural Mistake Detection」在策展清单里的坐标与要点：够你判断要不要去读原文，但不能替代原文。**

- 可确证的只有清单坐标：分组 **291 Procedural Error Detection**，以及 Highlights 点出的问题设定；本页不自行推导新结论。
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

- [`sources/papers/sun_awesome_ego_2510_08765_transparent-and-coherent-procedural-mist.md`](../../sources/papers/sun_awesome_ego_2510_08765_transparent-and-coherent-procedural-mist.md) — 本条目策展摘录
- [`sources/papers/sun_awesome_ego_catalog.md`](../../sources/papers/sun_awesome_ego_catalog.md) — 列表总表
- [`sources/repos/awesome-egocentric-vision.md`](../../sources/repos/awesome-egocentric-vision.md)
- 论文：<https://arxiv.org/abs/2510.08765>
- [arXiv PDF](https://arxiv.org/pdf/2412.11927)：已通读 §1、§2、§3、§4 与 §5 开头（附录与其余实验略读）；页面 arXiv 号 2510.08765 与该论文不符，正确编号为 2412.11927

## 推荐继续阅读

- [Awesome Egocentric Vision 仓库](https://github.com/sun254667/awesome-egocentric-vision)
- [原文](https://arxiv.org/abs/2510.08765)
