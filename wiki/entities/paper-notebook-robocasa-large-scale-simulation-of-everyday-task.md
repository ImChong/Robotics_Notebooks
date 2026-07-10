---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2406.02523"
related:
  - ../overview/paper-notebook-category-11-simulation-benchmark.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_robocasa.md
summary: "AI 的进展很大程度由规模化驱动，但机器人受限于缺乏海量机器人数据集。本文主张用逼真物理仿真来规模化机器人学习的环境、任务与数据。RoboCasa 是一个面向日常环境训练通才机器人的大规模仿真框架，以厨房为核心，提供逼真多样的场景、跨 150+ 物体类别的数千 3D 资产与数十种可交互家具家电。它用生成式 AI（文本生 3D 资产、文本生图像纹理）增强真实与多样性；设计 100 个任务用于系统评测，含由大模型引导生成的复合任务。为便于学习，提供高质量人类演示并集成自动轨迹生成以最小人力大幅扩充数据集。实验显示：用合成生成的机器人数据做大规模模仿学习有清晰的规模化趋势，且在真实任务上前景可观。"
---

# RoboCasa

**RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：11_Simulation_Benchmark），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

AI 的进展很大程度由规模化驱动，但机器人受限于缺乏海量机器人数据集。本文主张用逼真物理仿真来规模化机器人学习的环境、任务与数据。RoboCasa 是一个面向日常环境训练通才机器人的大规模仿真框架，以厨房为核心，提供逼真多样的场景、跨 150+ 物体类别的数千 3D 资产与数十种可交互家具家电。它用生成式 AI（文本生 3D 资产、文本生图像纹理）增强真实与多样性；设计 100 个任务用于系统评测，含由大模型引导生成的复合任务。为便于学习，提供高质量人类演示并集成自动轨迹生成以最小人力大幅扩充数据集。实验显示：用合成生成的机器人数据做大规模模仿学习有清晰的规模化趋势，且在真实任务上前景可观。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Generalist Robot | 通才机器人 |
| Text-to-3D / Text-to-Image | 文本生 3D / 文本生图，生成资产/纹理 |
| Composite Task | 复合任务（LLM 引导生成） |
| Trajectory Generation | 自动轨迹生成 |
| Scaling Trend | 规模化趋势 |
| 3D Asset | 3D 资产 |

## 为什么重要

- **"仿真 + 生成式 AI"是规模化数据的强力路径**，后续 RoboCasa365 进一步放大；
- **自动轨迹生成**以最小人力扩数据，是数据飞轮关键；
- **规模化趋势**为机器人基础模型提供信心；
- 厨房日常任务对人形家务落地高度相关。

## 解决什么问题

机器人学习缺**海量数据**： - 真实采集**贵、慢**； - 缺**逼真、多样、可规模化**的仿真环境/任务/数据。

RoboCasa 要：用**逼真仿真 + 生成式 AI + 自动轨迹生成**，把环境/任务/数据**规模化**。

## 核心机制

1. **大规模厨房仿真框架**：数千资产、150+ 类别、可交互家具家电；
2. **生成式 AI 增强**：文本生 3D 资产与纹理；
3. **100 任务（含 LLM 复合）+ 演示/自动轨迹生成**；
4. **规模化实证**：合成数据大规模 IL 呈清晰规模化趋势。

方法拆解（深读笔记小节）：逼真厨房场景 + 海量资产；生成式 AI 增强多样性；100 任务（含 LLM 复合任务）；演示 + 自动轨迹生成；规模化结论；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 11_Simulation_Benchmark |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/RoboCasa__Large-Scale_Simulation_of_Everyday_Tasks_for_Generalist_Robots/RoboCasa__Large-Scale_Simulation_of_Everyday_Tasks_for_Generalist_Robots.html> |
| arXiv | <https://arxiv.org/abs/2406.02523> |
| 作者 | Soroush Nasiriany、Abhiram Maddukuri、Lance Zhang、Adeet Parikh、Ajay Mandlekar、Yuke Zhu 等（UT Austin / NVIDIA） |
| 发表 | 2024 年 6 月 |
| 项目主页 | [robocasa.ai](https://robocasa.ai/) |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../overview/paper-notebook-category-11-simulation-benchmark.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_robocasa.md](../../sources/papers/humanoid_pnb_robocasa.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/RoboCasa__Large-Scale_Simulation_of_Everyday_Tasks_for_Generalist_Robots/RoboCasa__Large-Scale_Simulation_of_Everyday_Tasks_for_Generalist_Robots.html>
- 论文：<https://arxiv.org/abs/2406.02523>

## 推荐继续阅读

- [机器人论文阅读笔记：RoboCasa](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/RoboCasa__Large-Scale_Simulation_of_Everyday_Tasks_for_Generalist_Robots/RoboCasa__Large-Scale_Simulation_of_Everyday_Tasks_for_Generalist_Robots.html)
