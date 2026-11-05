---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2505.19086"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_maskedmanipulator-versatile-whole-body-control-f.md
summary: "MaskedManipulator：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# MaskedManipulator

**MaskedManipulator: Versatile Whole-Body Control for Loco-Manipulation** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：13_Physics-Based_Animation）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

MaskedManipulator 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在深读笔记写出来之前，本页只保留分类位置与原文入口；方法模块与数据流见下方「流程总览」（仅依据论文方法部分，不含实验结论）。

## 流程总览

```mermaid
flowchart TD
    A["GRAB 人-物交互动捕数据（10 名受试者）"] --> B["骨架重定向：迁移局部 DoF 旋转"]
    B --> C["物体重定向：依据接触数据保持手-物接触关系"]
    C --> D["过滤不适合物理重建的序列"]
    D --> E["训练集 1007 条 / 测试集 141 条"]
    E --> F["阶段一：MimicManipulator 运动跟踪策略 π_track"]
    F --> G["PPO 训练：R_track = r_pose · r_contact · r_energy · r_interaction"]
    G --> H["提前终止 + 对失败序列过采样"]
    H --> I["训练好的教师策略 π_track"]
    I --> J["阶段二：在线师生蒸馏 (DAgger)"]
    M["随机掩码的稀疏未来目标 g_versatile（手、物体、全身姿态）"] --> N["MaskedManipulator 学生策略 π_versatile（Transformer）"]
    S["当前状态 s_t：本体感知、手-物向量、物体/桌面位姿与 BPS"] --> N
    J --> N
    N --> O["三种架构：Deterministic / C-VAE / Diffusion"]
    O --> P["动作 a_t 控制仿真角色（ProtoMotions + IsaacLab）"]
    I -.->|"给出稠密目标下的动作 a_track"| J
```

图依据：论文 §4.1 MimicManipulator、§4.2 MaskedManipulator、§5 Experiments 设置、Fig.3 / Fig.4

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 13_Physics-Based_Animation |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/13_Physics-Based_Animation/maskedmanipulator-versatile-whole-body-control-f` |
| arXiv | <https://arxiv.org/abs/2505.19086> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是尚未深读的待读条目：它保证 MaskedManipulator 在清单里有这么一条，但还不构成对这篇工作的判断。**

- 该条目已列入 Paper Notebooks 的 PROGRESS.md 待深读清单，分类为 13_Physics-Based_Animation，可从总索引与分类页检索到。
- 深读笔记尚未撰写，本页不含机制拆解、量化 benchmark 或实机指标；技术结论请直接回到 arXiv 原文。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_maskedmanipulator-versatile-whole-body-control-f.md](../../sources/papers/humanoid_pnb_maskedmanipulator-versatile-whole-body-control-f.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2505.19086>
- [arXiv PDF](https://arxiv.org/pdf/2505.19086)：已通读方法部分（实验与附录以略读为准）：§3 Preliminaries、§4.1（Task design / Data processing）、§4.2、§5 实验设置与训练细节

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
