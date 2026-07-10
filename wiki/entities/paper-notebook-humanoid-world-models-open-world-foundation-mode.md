---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2506.01182"
related:
  - ../overview/paper-notebook-category-11-simulation-benchmark.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_humanoid-world-models.md
summary: "人形以类人形态特别适合在为人设计的环境里交互，但让它在复杂开放世界里推理、规划、行动仍难。本文提出轻量、开源的人形世界模型：以人形控制输入为条件，预测未来的第一视角视频。作者训练两类生成模型——掩码 Transformer（Masked Transformers）与流匹配（Flow-Matching）——于 100 小时演示数据；并通过参数共享技术，在性能与视觉保真几乎无损的前提下把模型缩小 33–53%，使其能部署在1–2 张 GPU 的有限算力上，面向学术与小实验室。"
---

# Humanoid World Models

**Humanoid World Models: Open World Foundation Models for Humanoid Robotics** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：11_Simulation_Benchmark），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人形以类人形态特别适合在为人设计的环境里交互，但让它在复杂开放世界里推理、规划、行动仍难。本文提出轻量、开源的人形世界模型：以人形控制输入为条件，预测未来的第一视角视频。作者训练两类生成模型——掩码 Transformer（Masked Transformers）与流匹配（Flow-Matching）——于 100 小时演示数据；并通过参数共享技术，在性能与视觉保真几乎无损的前提下把模型缩小 33–53%，使其能部署在1–2 张 GPU 的有限算力上，面向学术与小实验室。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| World Model | 世界模型，预测未来观测 |
| Egocentric Video | 第一视角视频 |
| Masked Transformer | 掩码 Transformer 生成模型 |
| Flow-Matching | 流匹配生成模型 |
| Parameter Sharing | 参数共享，压缩模型 |
| Foundation Model | 基础模型 |

## 为什么重要

- **世界模型让人形"想象后果"**，是规划与数据增广的有力工具；
- **轻量化 + 开源**降低门槛，惠及算力受限的研究者；
- **第一视角条件预测**与 ZeroWBC/EgoHumanoid 的 egocentric 路线呼应；
- 流匹配/掩码 Transformer 是当前生成式世界模型的主流选择。

## 解决什么问题

人形要在**开放世界**推理/规划/行动，世界模型有用但： - 大模型**算力门槛高**，小实验室难用； - 缺**轻量、开源**、以**控制输入为条件**的人形世界模型。

论文要：一个**轻量、可在 1–2 GPU 上跑**的开源人形世界模型。

## 核心机制

1. **轻量开源人形世界模型**：控制条件的第一视角视频预测；
2. **两类生成架构**：掩码 Transformer 与流匹配，100h 训练；
3. **参数共享轻量化**：缩小 33–53%、性能几乎无损；
4. **低算力可部署**：1–2 GPU，面向学术/小实验室。

方法拆解（深读笔记小节）：控制条件的第一视角视频预测；两类生成架构；参数共享做轻量化；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 11_Simulation_Benchmark |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Humanoid_World_Models__Open_World_Foundation_Models_for_Humanoid_Robotics/Humanoid_World_Models__Open_World_Foundation_Models_for_Humanoid_Robotics.html> |
| arXiv | <https://arxiv.org/abs/2506.01182> |
| 作者 | Muhammad Qasim Ali、Aditya Sridhar、Shahbuland Matiana、Alex Wong、Mohammad Al-Sharman |
| 发表 | 2025 年 6 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../overview/paper-notebook-category-11-simulation-benchmark.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_humanoid-world-models.md](../../sources/papers/humanoid_pnb_humanoid-world-models.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Humanoid_World_Models__Open_World_Foundation_Models_for_Humanoid_Robotics/Humanoid_World_Models__Open_World_Foundation_Models_for_Humanoid_Robotics.html>
- 论文：<https://arxiv.org/abs/2506.01182>

## 推荐继续阅读

- [机器人论文阅读笔记：Humanoid World Models](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Humanoid_World_Models__Open_World_Foundation_Models_for_Humanoid_Robotics/Humanoid_World_Models__Open_World_Foundation_Models_for_Humanoid_Robotics.html)
