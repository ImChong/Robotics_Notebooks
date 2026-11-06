---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2508.00362"
venue: "2025.08"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_a-whole-body-motion-imitation-framework-from-hum.md
summary: "A Whole-Body Motion Imitation Framework from Human Data for Full-Size Humanoid Robot：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# A Whole-Body Motion Imitation Framework from Human Data for Full-Size Humanoid Robot

**A Whole-Body Motion Imitation Framework from Human Data for Full-Size Humanoid Robot** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

A Whole-Body Motion Imitation Framework from Human Data for Full-Size Humanoid Robot 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见上方「流程总览」（依据论文方法部分，非深读笔记）。

## 流程总览

```mermaid
flowchart TD
    M["人体动作数据 P_human<br/>（动捕 Mocap / 离线数据）"] --> R
    subgraph S1["阶段 1：Contact-aware Whole-Body Motion Retargeting"]
        R["图网络：Graph Encoder f_θ → 潜变量 Z → Graph Decoder f_φ<br/>输出机器人关节角 Q_robot，经正运动学得 P_robot"]
        R --- LS["损失 L = α·L_vec + β·L_foot + γ·L_smo + η·L_col<br/>向量相似 / 足接触 / 平滑 / 自碰撞"]
        R --> X["参考轨迹与接触计划<br/>在线：下一时刻 (x_next, u_next)<br/>离线：完整 (x_ref, u_ref) + contact schedule"]
    end
    X --> N
    subgraph S2["阶段 2：Nonlinear Centroidal MPC"]
        N["非线性质心 MPC（direct multiple shooting + SQP 实时迭代）<br/>状态 x、控制 u；足端接触等式约束与摩擦锥 / 关节限位不等式约束<br/>输出最优 x*, u*"]
    end
    N --> W
    subgraph S3["阶段 3：Whole-Body Controller"]
        W["单个 QP：低优先级任务入代价，高优先级任务作约束<br/>输出期望关节力矩 τ*"]
    end
    W --> ROBOT["全尺寸人形机器人执行"]
    ROBOT -->|"测量状态 x0"| N
```

图依据：论文 §III / Fig.2、Eq.(1)–(4)

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/A_Whole-Body_Motion_Imitation_Framework_from_Human_Data_for_Full-Size_Humanoid_R` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前提供的是「待读条目」而不是论文解读：它把这条 04_Loco-Manipulation_and_WBC 条目挂进知识库以保持检索连通，实质技术判断要等深读笔记落地。**

- 页面已确认的信息除「流程总览」外仅有分类（04_Loco-Manipulation_and_WBC）、深读状态与计划文件夹路径；量化 benchmark 与实机指标均尚未记录。
- 适用边界很窄：可用于按分类检索，不能作为该论文技术结论的引用依据。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_a-whole-body-motion-imitation-framework-from-hum.md](../../sources/papers/humanoid_pnb_a-whole-body-motion-imitation-framework-from-hum.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2508.00362)：已通读方法部分（实验与附录以略读为准）：§III Retargeting and Control（A 重定向、B 非线性 MPC、C 全身控制器）、§IV-A 实验设置

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
