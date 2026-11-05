---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2503.22249"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_flam-foundation-model-based-body-stabilization-f.md
summary: "FLAM：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# FLAM

**FLAM: Foundation Model-Based Body Stabilization for Humanoid Locomotion and Manipulation** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：14_Human_Motion）。本页 **还没有深读笔记**：除分类位置与原文入口外，仅给出依据论文方法部分的「流程总览」图，方法细节与数据请直接看原文。

## 一句话定义

FLAM 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口，不给方法结论。

## 流程总览

```mermaid
flowchart TD
    ENV["Humanoid-Bench 仿真环境（Unitree H1）"]
    POL["基础策略 TD-MPC2<br/>潜变量动力学 + 价值函数的 MPC 框架"]
    SEG["与环境交互 l_s 步，得到机器人状态轨迹片段"]
    MAP["1) 机器人-人体姿态映射 F_M<br/>对齐初始零位姿，机器人关节相对朝向 → 人体关节旋转（SMPL-X）"]
    RECON["2) 人体运动重建基础模型 RoHM（扩散模型）<br/>输入含噪人体姿态片段，输出稳定姿态片段"]
    SIM["3) 逐关节比较重建前后位置与朝向<br/>相似关节数超过阈值 t_s 则给稳定性奖励 R_S"]
    COMB["奖励合成 R = R_T + λ·(q/l_e)·R_S<br/>R_T 为基准任务奖励"]
    BUF["写入回放缓冲区"]
    UPD["从缓冲区采样批数据更新策略（重复 l_s 次）"]
    POL --> ENV --> SEG --> MAP --> RECON --> SIM --> COMB
    ENV -->|任务奖励 R_T| COMB
    COMB --> BUF --> UPD --> POL
```

图依据：论文 §III-A、§III-B（Fig. 1、Fig. 2、Eq. 1–7）

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/14_Human_Motion/flam-foundation-model-based-body-stabilization-f` |
| arXiv | <https://arxiv.org/abs/2503.22249> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**FLAM 在本库目前只是一条「已排期、未深读」的待读条目：它只标出"基础模型用于人形躯干稳定"这一线索这条线上还有这么一篇，具体结论要等读完。**

- 本页能确认的只有索引层信息——arXiv 2503.22249、分类 14_Human_Motion 与计划中的笔记文件夹；标题指向"基于基础模型的身体稳定，服务于人形运动与操作"这一命题，基础模型具体承担何种角色、如何与底层控制衔接，页内均未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-14-human-motion.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为方案对比或工程选型的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 PROGRESS.md 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_flam-foundation-model-based-body-stabilization-f.md](../../sources/papers/humanoid_pnb_flam-foundation-model-based-body-stabilization-f.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2503.22249>
- [arXiv PDF](https://arxiv.org/pdf/2503.22249)：已通读方法部分（实验与附录以略读为准）：§III 框架（A 稳定性奖励函数、B 基础策略与训练策略，含策略更新流程）；§IV 实验仅略读

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
