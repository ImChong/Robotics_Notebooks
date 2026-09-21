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
updated: 2026-09-21
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
