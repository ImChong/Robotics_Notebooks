---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2505.23708"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_amor-adaptive-character-control-through-multi-ob.md
summary: "AMOR：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# AMOR

**AMOR: Adaptive Character Control through Multi-Objective Reinforcement Learning** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：13_Physics-Based_Animation）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

AMOR 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见「流程总览」（依据论文方法部分），不给实验结论。

## 流程总览

```mermaid
flowchart LR
    REF["参考动作序列 m_t"]
    VAE["VAE 编码动作窗口 → 隐变量 z_t"]
    CTX["运动上下文 c_t = (m_t, z_t)"]
    W["每个 episode 从单纯形采样奖励权重 w（Dirichlet）"]
    POL["AMOR 策略 π(a_t | s_t, c_t, w)"]
    SIM["仿真角色 / 机器人"]
    RV["7 维向量奖励：上身、下身、脚、刚体、根、速度、平滑"]
    MOPPO["MOPPO：向量价值函数 + 权重标量化优势"]
    HLP["高层策略 HLP：按 s_t, c_t 输出 w_t（softmax）"]
    DISC["判别器隐式奖励 r_D"]
    REF --> VAE --> CTX --> POL
    W --> POL
    POL -->|"动作 a_t"| SIM
    SIM --> RV
    RV --> MOPPO
    MOPPO -.->|"更新"| POL
    HLP -->|"冻结 AMOR，在线给出 w_t"| POL
    SIM --> DISC
    DISC -.->|"PPO 训练 HLP"| HLP
```

图依据：论文 §3 AMOR Overview、§4.2、§5、§6（Fig.2 / Fig.3）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 13_Physics-Based_Animation |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/13_Physics-Based_Animation/amor-adaptive-character-control-through-multi-ob` |
| arXiv | <https://arxiv.org/abs/2505.23708> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 AMOR 的待读条目节点：能被引用的只有归档事实（13_Physics-Based_Animation、待深读、arXiv 入口），技术结论一概尚未成立。**

- 可依赖的信息限于分类、计划文件夹路径与 <https://arxiv.org/abs/2505.23708>；机制与指标待深读笔记补齐。
- 适用边界：用于按分类检索，**不可**作为该论文结论的引用来源。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_amor-adaptive-character-control-through-multi-ob.md](../../sources/papers/humanoid_pnb_amor-adaptive-character-control-through-multi-ob.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2505.23708>
- [arXiv PDF](https://arxiv.org/pdf/2505.23708)：已通读方法部分（实验与附录以略读为准）：§3–§6（方法）、§7 开头的评测设定（实验结果略读）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
