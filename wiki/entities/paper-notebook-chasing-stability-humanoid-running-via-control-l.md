---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2509.19573"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_chasing-stability-humanoid-running-via-control-l.md
summary: "Chasing Stability：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Chasing Stability

**Chasing Stability: Humanoid Running via Control Lyapunov Function Guided RL** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Chasing Stability 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」，不含实验结论。

## 流程总览

```mermaid
flowchart TD
    A["多域轨迹优化 (Casadi/Pinocchio/IPOPT)<br/>单支撑 SSP → 腾空 FLT 的半周期 + 周期性约束"] --> B["步态库：1.1–3.0 m/s 多速度步态<br/>由快到慢热启动 + 站立姿态"]
    B --> C["Bézier 曲线拟合优化节点<br/>得到虚拟约束 y_d(t)（末端位姿/关节）"]
    C --> D["按指令前向速度 v_x^d 选取最近速度步态"]
    D --> E["误差 η → CLF V(η)=ηᵀPη<br/>CLF 跟踪奖励 r_v + CLF 衰减条件奖励 r_v̇"]
    E --> F["总奖励 R = r_reg + r_v + r_v̇ + r_dom<br/>（r_dom：SSP 支撑脚静止 / 腾空接触力惩罚）"]
    F --> G["PPO 训练 (IsaacLab)：Actor 观测含相位 sin/cos<br/>Critic 含特权信息；域随机化"]
    G --> H["策略部署 Unitree G1"]
    L["Fast-LIO 激光雷达里程计<br/>位置/偏航估计"] --> PD["分层 PD 控制器：横向位置 + 航向<br/>输出 v_y、v_ω 指令（按训练范围饱和）"]
    PD --> H
    H --> I["直线跑步（跑步机 / 室外）"]
```

图依据：论文 §III-A、§III-B、§III-C / Fig.2、Fig.5、式 (11)–(15)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/chasing-stability-humanoid-running-via-control-l` |
| arXiv | <https://arxiv.org/abs/2509.19573> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前是一条待读记录，而不是对 Chasing Stability 的归纳——它记录的是「这篇该读、归在哪一类、还没读」。**

- 页面能给出的确定信息只有四项：分类为 05_Locomotion、深读状态为待撰写、计划文件夹路径已预留、arXiv 入口已登记。
- 论文本身的机制、量化 benchmark 与实机结论 **本页均未涵盖**，不要把这里的任何表述当作对该工作的评价。
- 它的作用是让你能从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到它到该实体；深读笔记完成后本页应链向笔记站并补成完整摘要。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_chasing-stability-humanoid-running-via-control-l.md](../../sources/papers/humanoid_pnb_chasing-stability-humanoid-running-via-control-l.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2509.19573>
- [arXiv PDF](https://arxiv.org/pdf/2509.19573)：已通读方法部分（实验与附录以略读为准）：§III Running Controller（III-A/B/C）；§IV 结果略读

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
