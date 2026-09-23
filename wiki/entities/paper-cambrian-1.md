---
type: entity
tags:
  - paper
  - mllm
  - vision-centric
  - spatial-reasoning
  - open-source
  - nyu
status: complete
updated: 2026-09-23
arxiv: "2406.16860"
code: https://github.com/cambrian-mllm/cambrian
related:
  - ./cv-bench-embodied.md
  - ../concepts/3d-spatial-vqa.md
  - ./molmo2-vlm.md
  - ../overview/spatial-reasoning-benchmarks-technology-map.md
sources:
  - ../../sources/papers/cambrian_1_arxiv_2406_16860.md
  - ../../sources/sites/cambrian-mllm.md
  - ../../sources/repos/cambrian-mllm-cambrian.md
summary: "Cambrian-1（arXiv:2406.16860，NYU VisionX）：视觉中心 MLLM 全开源探索；Cambrian-10M 数据 + CV-Bench 空间推理评测。"
---

# Cambrian-1：视觉中心 MLLM 开源探索

**Cambrian-1**（*A Fully Open, Vision-Centric Exploration of Multimodal LLMs*，[arXiv:2406.16860](https://arxiv.org/abs/2406.16860)，[项目页](https://cambrian-mllm.github.io/)，[代码](https://github.com/cambrian-mllm/cambrian)）以 **vision-centric** 设计系统消融 MLLM：视觉塔、连接器、数据混合，并发布 **Cambrian-10M** 与 **CV-Bench** 空间推理基准。

## 一句话定义

**Cambrian-1 用全开源栈回答：MLLM 的空间能力到底来自语言塔还是视觉塔——并附带 CV-Bench 标尺。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MLLM | Multimodal Large Language Model | 多模态大语言模型 |
| LVLM | Large Vision-Language Model | 大视觉-语言模型 |
| VQA | Visual Question Answering | 视觉问答 |

## 为什么重要

- **CV-Bench  lineage：** 站内 [CV-Bench（具身 ER 套件）](./cv-bench-embodied.md) 与 Cambrian **CV-Bench** 同名不同物——本页指 Cambrian 原版 **2D 空间推理** benchmark（HF `nyu-visionx/CV-Bench`）。
- **视觉中心方法论：** 影响后续 Molmo/Cambrian 系与 spatial MLLM 数据配方讨论。
- **Fully open：** 模型 + 10M 数据 + 训练代码全发布。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | NYU VisionX 等 |
| **数据** | Cambrian-10M |
| **基准** | CV-Bench（spatial reasoning） |
| **开源** | **已开源** 模型/数据/代码 |

## 实验与评测

- **自带标尺：** 本文随模型一并发布 **CV-Bench** 空间推理基准（HF `nyu-visionx/CV-Bench`），用于把「MLLM 的空间能力」从通用 VQA 平均分里单独拆出来评。
- **主要评测轴（vision-centric 消融）：** 视觉塔选型、视觉—语言连接器设计、指令数据混合比例三条轴分别消融，结论是 **视觉侧设计对空间类题目的边际收益被长期低估**。
- **数值口径：** 本页为 ingest 级摘要，**未复核逐项分数**；各模型在 CV-Bench 与通用 MLLM 基准上的具体数值 **以 [原文](https://arxiv.org/abs/2406.16860) 表格与 [HF 数据卡](https://huggingface.co/datasets/nyu-visionx/CV-Bench) 为准**。
- **复现边界：** 模型、Cambrian-10M 数据与训练代码均已开源，是少数可以 **端到端重跑数据配方消融** 的 MLLM 工作；复现时以 [官方仓库](https://github.com/cambrian-mllm/cambrian) 为准。

## 与其他工作对比

| 维度 | Cambrian-1（本文） | [RoboSpatial](./robospatial.md) / [EmbSpatial](./embspatial.md) 等机器人空间基准 | [CV-Bench（LightNav-ER 套件）](./cv-bench-embodied.md) |
|------|--------------------|--------------------------------------------------------------|------------------------------------------------|
| 出题视角 | 通用图像的 2D/3D 空间关系 | **机器人本体视角** 的场景空间关系 | LightNav ER 中期训练的能力锚点之一 |
| 目的 | 消融「MLLM 的空间能力来自哪」 | 评机器人可用的空间理解 | 评 ER 训练采样配比是否有效 |
| 与本文的关系 | 上游 | 下游、换场景重做 | **同名不同物**，指的是 LightNav 复用后的具身版 |

- **命名坑：** 站内两个 CV-Bench 必须分清——本页是 **Cambrian 原版**，[cv-bench-embodied](./cv-bench-embodied.md) 是 LightNav-ER 套件里的具身版；读 LightNav 博客的 CV-Bench 分数时不要回填到本文表格。
- **能力边界：** 本文是 **MLLM 侧** 的开源基线与标尺，不含动作输出；它的 CV-Bench 分数属 [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) 的 ① 认知层，**不蕴含** 下游策略成功率。
- **与 Molmo 系的关系：** [Molmo2](./molmo2-vlm.md) 等后续视觉中心 MLLM 在数据配方讨论上与本文同源；横比时先对齐视觉塔与数据规模，否则比的是数据不是方法。

## 结论

**Cambrian-1 是 spatial MLLM 的「开源基线 + CV-Bench 起源」——读 LightNav CV-Bench 脚注前先分清 Cambrian 原版与具身 ER 套件。**

- vision-centric 消融对连接器/视觉塔选型有长期参考价值
- CV-Bench 成为后续 ER/具身 benchmark 命名的上游
- 与 RoboSpatial/EmbSpatial 等 **机器人场景** benchmark 互补
- 全开源便于复现 mid-training 数据配方

## 关联页面

- [CV-Bench（LightNav-ER 套件）](./cv-bench-embodied.md)
- [3D 空间 VQA](../concepts/3d-spatial-vqa.md)
- [Molmo2 VLM](./molmo2-vlm.md)

## 参考来源

- [cambrian_1_arxiv_2406_16860.md](../../sources/papers/cambrian_1_arxiv_2406_16860.md)
- [cambrian-mllm.md](../../sources/sites/cambrian-mllm.md)
- [cambrian-mllm-cambrian.md](../../sources/repos/cambrian-mllm-cambrian.md)

## 推荐继续阅读

- [Cambrian-1 项目页](https://cambrian-mllm.github.io/)
- [CV-Bench on HF](https://huggingface.co/datasets/nyu-visionx/CV-Bench)
