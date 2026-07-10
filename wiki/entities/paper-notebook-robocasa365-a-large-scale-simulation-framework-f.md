---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2603.04356"
related:
  - ../overview/paper-notebook-category-11-simulation-benchmark.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_robocasa365.md
summary: "RoboCasa365 是一个面向通才机器人训练与评测的大规模仿真框架，在已有 RoboCasa 基础上大幅扩展资产、环境、任务与数据集。它包含 365 个日常任务、2500 个多样厨房场景、600+ 小时人类演示与额外 1600 小时合成演示，并提供系统化基准用于训练与评测通才机器人模型。框架支持多种形态的移动操作机器人——单臂移动平台、人形机器人、带臂四足——并面向多任务学习、机器人基础模型训练、终身学习等不同问题设定提供系统化评测。"
---

# RoboCasa365

**RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：11_Simulation_Benchmark），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

RoboCasa365 是一个面向通才机器人训练与评测的大规模仿真框架，在已有 RoboCasa 基础上大幅扩展资产、环境、任务与数据集。它包含 365 个日常任务、2500 个多样厨房场景、600+ 小时人类演示与额外 1600 小时合成演示，并提供系统化基准用于训练与评测通才机器人模型。框架支持多种形态的移动操作机器人——单臂移动平台、人形机器人、带臂四足——并面向多任务学习、机器人基础模型训练、终身学习等不同问题设定提供系统化评测。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Generalist Robot | 通才机器人，一模型多任务多形态 |
| 365 Tasks | 365 个日常任务 |
| Multi-task Learning | 多任务学习 |
| Foundation Model | 机器人基础模型 |
| Lifelong Learning | 终身学习 |
| Mobile Manipulator | 移动操作机器人（含人形/四足） |

## 为什么重要

- **规模 + 多形态**是训练机器人基础模型的底座，人形是其中一等公民；
- **大量合成演示**缓解真实数据稀缺；
- **终身学习/基础模型设定**指向通用机器人智能的评测方向；
- 与本仓 11 其它基准（RoboCasa、ManiSkill-HAB、BiGym）一脉相承、规模更大。

## 解决什么问题

训练**通才机器人**缺**足够大、足够系统**的仿真与基准： - 任务/场景/数据规模不足； - 缺**跨形态**（含人形）统一平台； - 缺面向**基础模型/终身学习**的系统化评测设定。

RoboCasa365 要：一个**大规模、多形态、系统化**的训练与评测框架。

## 核心机制

1. **大规模扩展 RoboCasa**：365 任务、2500 场景、600h+1600h 演示；
2. **多形态支持**：单臂/人形/带臂四足；
3. **系统化基准**：面向多任务、基础模型、终身学习；
4. **通才机器人训练/评测平台**。

方法拆解（深读笔记小节）：在 RoboCasa 上大规模扩展；多形态支持；系统化基准与问题设定；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 11_Simulation_Benchmark |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/RoboCasa365__A_Large-Scale_Simulation_Framework_for_Training_and_Benchmarking_Generalist_Robots/RoboCasa365__A_Large-Scale_Simulation_Framework_for_Training_and_Benchmarking_Generalist_Robots.html> |
| arXiv | <https://arxiv.org/abs/2603.04356> |
| 发表 | 2026 年 3 月 |
| 会议 | ICLR（会议贡献，见原文） |
| 项目主页 | [robocasa.ai](https://robocasa.ai/) |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../overview/paper-notebook-category-11-simulation-benchmark.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_robocasa365.md](../../sources/papers/humanoid_pnb_robocasa365.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/RoboCasa365__A_Large-Scale_Simulation_Framework_for_Training_and_Benchmarking_Generalist_Robots/RoboCasa365__A_Large-Scale_Simulation_Framework_for_Training_and_Benchmarking_Generalist_Robots.html>
- 论文：<https://arxiv.org/abs/2603.04356>

## 推荐继续阅读

- [机器人论文阅读笔记：RoboCasa365](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/RoboCasa365__A_Large-Scale_Simulation_Framework_for_Training_and_Benchmarking_Generalist_Robots/RoboCasa365__A_Large-Scale_Simulation_Framework_for_Training_and_Benchmarking_Generalist_Robots.html)
