---
type: concept
tags: [control, wbc, humanoid, optimization]
status: complete
related:
  - ../tasks/locomotion.md
  - ../methods/imitation-learning.md
  - ../comparisons/wbc-vs-rl.md
  - ./sim2real.md
  - ./contact-estimation.md
  - ../queries/when-to-use-wbc-vs-rl.md
---

# Whole-Body Control (WBC)

**全身控制**：对人形机器人等复杂系统，同时协调多个肢体/关节完成全身任务的控制方法。

## 一句话定义

不单独控制每个关节，而是把整个身体当成一个整体来协调控制。

## 为什么重要

人形机器人的特点：

- 30+ 自由度
- 多个末端执行器（手、脚）
- 浮动基（身体位置和朝向不可控）
- 必须保持平衡

传统独立关节控制的局限：

- 各关节互相耦合，单独优化会冲突
- 没有全身协调意识
- 难以处理接触切换和约束

WBC 的优势：

- 通过优化或 hierarchical control 实现全身协调
- 能同时处理接触力、平衡、末端执行器任务
- 自然处理多接触场景

## 核心框架

### 1. Hierarchical WBC
多层结构：

```
任务空间控制器（末端执行器轨迹）
        ↓
全身QP优化（关节力矩分配）
        ↓
电机驱动
```

代表：Whole-Body Impedance / TSID (Task Space Inverse Dynamics)

### 2. Optimization-based WBC
用 QP 或非线性优化直接求解：

- 给定任务目标（末端位置/姿态/力）
- 满足约束（关节限位、接触力、力矩平衡）
- 最小化某个代价函数

代表框架：Openuhan / tsid / exotica

### 3. Learning-based WBC
用 RL 或 IL 学习全身策略，不依赖精确建模。

代表：DeepMimic, ASE, CALM, MimicKit

## 关键概念

### Floating Base
人形机器人在空中时，base（躯干）的位置和朝向不在控制输入的直接控制下，需要通过接触力来驱动。

### Contact Schedule
什么时候哪只脚/手在接触地面，决定了可用力和平衡策略。

### Centroidal Dynamics
用质心动力学代替全关节动力学，更高效但精度略低。

## 参考来源

- Sentis & Khatib, *Synthesis of Whole-Body Behaviors Through Hierarchical Control of Behavioral Primitives* — WBC 早期基础论文
- Del Prete et al., *Task Space Inverse Dynamics* — WBC 动力学一致控制核心工作
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md) — TSID / HQP / Crocoddyl ingest 摘要
- [TSID (Task Space Inverse Dynamics)](https://github.com/stack-of-tasks/tsid) — 开源 WBC 实现
- [Whole-Body Control 论文导航](../../references/papers/whole-body-control.md) — 论文集合

## 关联页面

- [Locomotion](../tasks/locomotion.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [WBC vs RL](../comparisons/wbc-vs-rl.md)
- [Sim2Real](./sim2real.md)
- [Contact Estimation](./contact-estimation.md) — WBC 的接触集合来自接触估计，直接影响约束矩阵
- [Query：什么时候该用 WBC，什么时候该用 RL？](../queries/when-to-use-wbc-vs-rl.md)

## 继续深挖入口

如果你想沿着 WBC 继续往下挖，建议从这里进入：

### 论文入口
- [Whole-Body Control 论文导航](../../references/papers/whole-body-control.md)

### 工具 / Repo 入口
- [Utilities / Tooling](../../references/repos/utilities.md)

## 推荐继续阅读

- [ATOM01-Train](https://github.com/Roboparty/atom01_train)
- [TSID (Task Space Inverse Dynamics)](https://github.com/stack-of-tasks/tsid)
