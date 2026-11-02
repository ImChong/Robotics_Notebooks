---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2507.05906"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_feature-based-vs-gan-based-learning-from-demonst.md
summary: "Feature-Based vs. GAN-Based Learning from Demonstrations：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Feature-Based vs. GAN-Based Learning from Demonstrations

**Feature-Based vs. GAN-Based Learning from Demonstrations: When and Why** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：13_Physics-Based_Animation）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Feature-Based vs. GAN-Based Learning from Demonstrations 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；综述的组织结构见上文流程总览（依据论文 §1–§11），不含对各方法优劣的结论。

## 流程总览

```mermaid
flowchart TD
  A["§1 定义：learning from demonstrations<br/>用离线、基于状态的参考数据导出奖励，再用 RL 优化策略"] --> B["§4 参考数据的作用<br/>奖励设计转为定义或学习「相似度度量」"]
  B --> C["Feature-based 路线<br/>显式、手工特征奖励"]
  B --> D["GAN-based 路线<br/>判别器给出隐式奖励"]
  C --> E["§5 DeepMimic 式（Fig.1）<br/>逐帧特征对比，phase 变量对齐"]
  D --> F["§6 对抗奖励（Fig.2）<br/>判别器区分短转移片段"]
  F --> G["§6 隐变量条件 GAN（Fig.3）<br/>策略与判别器以运动嵌入为条件"]
  E --> H["§7 结构化表示的 feature-based（Fig.4）<br/>逐帧特征奖励 + 运动嵌入条件"]
  G --> I["§8 总结（Table 1 分类、8.1 / 8.2 优劣）"]
  H --> I
  I --> J["§9 评测指标与误解 / §10 常见看法辨析 / §11 结语"]
```

图依据：综述组织结构 / 分类体系图，论文 §1、§4–§8（Fig.1–Fig.4、Table 1 分类）、§9–§11 的章节顺序。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 13_Physics-Based_Animation |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/13_Physics-Based_Animation/feature-based-vs-gan-based-learning-from-demonst` |
| arXiv | <https://arxiv.org/abs/2507.05906> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**Feature-Based vs. GAN-Based Learning from Demonstrations 目前还没读：本页只登记了它的分类位置与原文入口，给不出可直接引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2507.05906、分类 13_Physics-Based_Animation、计划中的笔记文件夹；论文的方法机制、评测口径与实机结果在本页一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-13-physics-based-animation.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为技术选型或方法对比的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 progress.json 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_feature-based-vs-gan-based-learning-from-demonst.md](../../sources/papers/humanoid_pnb_feature-based-vs-gan-based-learning-from-demonst.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2507.05906>
- [arXiv PDF](https://arxiv.org/pdf/2507.05906)：已通读 §1–§7 的正文与 Fig.1–Fig.4 图注、§8 的 Table 1 与 §8.1 开头；§8.2、§9–§11 仅读标题与开头

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
