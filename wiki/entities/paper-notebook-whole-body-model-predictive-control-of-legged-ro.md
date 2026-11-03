---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2503.04613"
venue: "2025.03"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_whole-body-model-predictive-control-of-legged-ro.md
summary: "Whole-Body Model-Predictive Control of Legged Robots with MuJoCo：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# Whole-Body Model-Predictive Control of Legged Robots with MuJoCo

**Whole-Body Model-Predictive Control of Legged Robots with MuJoCo** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Whole-Body Model-Predictive Control of Legged Robots with MuJoCo 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 深读笔记尚未写出；方法流程见下方「流程总览」（依据论文方法部分），其余结论以论文为准。

## 流程总览

```mermaid
flowchart TD
    GUI["MuJoCo MPC GUI：用户实时调整残差权重、目标位置、iLQR 超参数"] --> ILQR
    EST["状态估计：关节编码器 + 动捕（MoCap）融合，500 Hz 低通滤波"] --> ILQR
    subgraph ILQR["MuJoCo iLQR（50 Hz）"]
        DER["MuJoCo 有限差分求动力学 / 残差 Jacobian<br/>（前向差分；软接触模型，impratio 调到 100）"] --> RIC["Riccati 递推（每次迭代更新一次，不做收敛检查）"]
        RIC --> OUT["名义控制 ū、名义状态 x̄、时变反馈增益 Kt"]
        OUT -->|"用上一次解 warm start 下一次迭代"| DER
    end
    OUT --> TVLQR["TV-LQR 策略（300 Hz）<br/>ut = ūt + Kt(xt − x̄t)"]
    EST --> TVLQR
    TVLQR --> PD["关节 PD 控制器（20 kHz）"]
    PD --> ROBOT["Unitree Go1 四足 / H1 人形硬件"]
    ROBOT --> EST
```

图依据：论文 §III Iterative LQR with MuJoCo、§IV Implementation Details / Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/Whole-Body_Model-Predictive_Control_of_Legged_Robots_with_MuJoCo` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页目前只是待读条目：题名指向「以 MuJoCo 作为动力学后端做腿足机器人全身 MPC」这条工程路线，但页面本身尚未给出任何方法细节或结果。**

- 本页可确认的只有条目级信息：分类 04_Loco-Manipulation_and_WBC、arXiv 2503.04613、发表 2025.03、深读状态「待撰写」。
- 求解器设置、实时性预算、与学习式控制的取舍等均未涉及，不应据此页判断该路线的可行性或成本。
- 当前职责只是维持与 [分类页](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md) / [总索引](../overview/humanoid-paper-notebooks-index.md) 的交叉检索；实质结论待深读笔记完成后回填。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_whole-body-model-predictive-control-of-legged-ro.md](../../sources/papers/humanoid_pnb_whole-body-model-predictive-control-of-legged-ro.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2503.04613)：已通读方法部分（实验与附录以略读为准）：§III Iterative LQR with MuJoCo、§IV Implementation Details、Fig.2 系统图


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
