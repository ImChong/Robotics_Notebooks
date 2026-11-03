---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2506.23125"
venue: "2025.06"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-motion-skills-with-adaptive-assistive-c.md
summary: "Learning Motion Skills with Adaptive Assistive Curriculum Force in Humanoid Robots：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# Learning Motion Skills with Adaptive Assistive Curriculum Force in Humanoid Robots

**Learning Motion Skills with Adaptive Assistive Curriculum Force in Humanoid Robots** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Learning Motion Skills with Adaptive Assistive Curriculum Force in Humanoid Robots 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 深读笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」，仅依据论文方法部分，不含结论判断。

## 流程总览

```mermaid
flowchart TD
    OBS["观测 o_t + 历史 o_t^H"] --> VAE["VAE 压缩历史<br/>预测 o_t+1，估计线速度 v_t"]
    VAE --> LAT["潜变量 z_t + 速度估计 v_t"]
    LAT --> MP["运动策略 agent<br/>输出关节目标 q_des"]
    LAT --> AF["辅助力 agent（仅仿真）<br/>输出 6D 空间力 F_t"]
    PRIV["特权信息子集<br/>地形 / 力边界 / 线速度 / 域随机化参数"] --> AF
    CUR["超立方体力边界课程<br/>η_k 随 ‖F‖/‖η‖ 与 isSkillLearned 衰减<br/>初始边界 η_0 按任务设定"] --> AF
    AF --> MASK["随机掩码 M_t<br/>以概率 ζ 关闭力分量"]
    MASK --> PEL["作用于骨盆 base link"]
    MP --> PD["PD 控制器"]
    PD --> ENV["仿真环境"]
    PEL --> ENV
    ENV --> RW["r_motion + r_force<br/>r_force 在技能达成 80% 后引入"]
    RW --> PPO["PPO + 非对称 Actor-Critic<br/>Critic 使用完整状态"]
    PPO --> MP
    PPO --> AF
    MP --> DEP["真机部署：辅助力为零"]
```

图依据：论文 §III-B（Fig. 1、Algorithm 1）与 §III-C。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/Learning_Motion_Skills_with_Adaptive_Assistive_Curriculum_Force_in_Humanoid_Robo` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是尚未深读的待读条目：它只是让这条线不漏掉这一篇，而不是对这篇工作的判断。**

- 该条目已列入 Paper Notebooks 的 progress 待深读清单，分类为 04_Loco-Manipulation_and_WBC，可从总索引与分类页检索到。
- 深读笔记尚未撰写，本页不含机制拆解、量化 benchmark 或实机指标，不应作为技术结论的依据。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-motion-skills-with-adaptive-assistive-c.md](../../sources/papers/humanoid_pnb_learning-motion-skills-with-adaptive-assistive-c.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2506.23125)：已通读方法部分（实验与附录以略读为准）：§III-A 至 §III-D


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
