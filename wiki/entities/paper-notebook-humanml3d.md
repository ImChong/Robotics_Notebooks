---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-27
arxiv: "2204.09419"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./paper-molingo.md
  - ./awesome-text-to-motion-zilize.md
  - ../methods/hy-motion-1.md
sources:
  - ../../sources/papers/humanoid_pnb_humanml3d.md
summary: "HumanML3D 是目前最主流的 3D 人体运动 - 文本数据集之一，提供了近 1.5 万个动作剪辑和 4.5 万条对应的自然语言描述，是研究文本生成动作（Text-to-Motion）的基石。"
---

# Generating Diverse and Natural 3D Human Motions from Textual Descriptions

**Generating Diverse and Natural 3D Human Motions from Textual Descriptions** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：14_Human_Motion）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

HumanML3D 是目前最主流的 3D 人体运动 - 文本数据集之一，提供了近 1.5 万个动作剪辑和 4.5 万条对应的自然语言描述，是研究文本生成动作（Text-to-Motion）的基石。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| T2M | Text-to-Motion | 文本条件人体运动生成 |
| FID | Fréchet Inception Distance | 生成分布与真实分布距离（越低越好） |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。
- 后续工作如 [MoLingo](./paper-molingo.md) 仍以 HumanML3D 为训评锚点，并扩展 MARDM-67 / MS-272 / TMR-263 等多协议对照。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/HumanML3D/HumanML3D.html> |
| arXiv | <https://arxiv.org/abs/2204.09419> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**HumanML3D 的分量不在方法而在「基座」：近 1.5 万动作剪辑配 4.5 万条自然语言描述，让文本生成动作从此有了共同的训练与评测锚点。**

- 真正起作用的是规模 × 配对：动作剪辑与自然语言描述成对提供，构成 Text-to-Motion 最主流的训练素材。
- 影响力体现在被持续沿用——[MoLingo](./paper-molingo.md) 仍以 HumanML3D 为训评锚点，并在其上扩展 MARDM-67 / MS-272 / TMR-263 等多协议对照。
- 使用时注意评测协议并不唯一：跨论文比较 FID 等指标前需先对齐协议，否则数字不可直接并列。
- 检索定位上与 [Awesome Text-to-Motion（Zilize）](./awesome-text-to-motion-zilize.md) 分工明确：那边是文献索引，本页是数据集本体。
- 本页为策展索引级摘要，详细机制待从深读笔记消化，量化 benchmark 以笔记与论文 PDF 为准。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 近期 HumanML3D SOTA 方法例：[MoLingo](./paper-molingo.md)
- T2M 文献索引：[Awesome Text-to-Motion（Zilize）](./awesome-text-to-motion-zilize.md)

## 参考来源

- [humanoid_pnb_humanml3d.md](../../sources/papers/humanoid_pnb_humanml3d.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/HumanML3D/HumanML3D.html>
- 论文：<https://arxiv.org/abs/2204.09419>

## 推荐继续阅读

- [机器人论文阅读笔记：Generating Diverse and Natural 3D Human Motions from Textual Descriptions](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/14_Human_Motion/HumanML3D/HumanML3D.html)
- [MoLingo（CVPR 2026）](./paper-molingo.md) — 在 HumanML3D 协议上的语义对齐连续 latent T2M
- [Awesome Text-to-Motion](./awesome-text-to-motion-zilize.md)
