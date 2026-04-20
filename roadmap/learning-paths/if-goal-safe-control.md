# 学习路径：如果目标是安全控制（CLF / CBF / Safe RL）

> 如果你想让机器人在满足安全约束的前提下运动，从这里切入。

**这条路径怎么用：**
- 目标读者是有控制理论基础、想加入安全保证的工程师或研究者
- 需要有基础线性代数和微分方程直觉；RL 基础有助于后期阶段
- 每个阶段有前置知识、核心问题、推荐做什么、学完输出什么

---

## Stage 0 数学基础

### 前置知识
- 线性代数（矩阵、特征值）
- 微分方程基础（稳定性直觉）
- 一点凸优化基础（QP 是什么）

### 核心问题
- Lyapunov 稳定性是什么意思
- 为什么 Lyapunov 函数能证明系统收敛

### 推荐读什么
- [Lyapunov 稳定性形式化](../../wiki/formalizations/lyapunov.md)
- Khalil, *Nonlinear Systems* — Chapter 4（稳定性定义）

### 学完输出什么
- 能手工验证一个简单系统的 Lyapunov 稳定性
- 理解正定函数和负半定导数的含义

---

## Stage 1 CLF / CBF 基础

### 核心问题
- CLF 和 CBF 分别解决什么问题（收敛 vs. 安全边界）
- 两者如何联合放入 QP 优化

### 推荐读什么
- [Control Lyapunov Function](../../wiki/formalizations/control-lyapunov-function.md)
- [Control Barrier Function](../../wiki/concepts/control-barrier-function.md)
- [CLF vs CBF 对比](../../wiki/comparisons/clf-vs-cbf.md)
- Ames et al., *Control Barrier Function based Quadratic Programs* (2017)

### 推荐做什么
- 用 Python + CVXPY 实现一个 CBF-QP，保证 2D 小车不越过边界

### 学完输出什么
- 能解释 CLF 和 CBF 的数学定义和功能区别
- 能写出 CBF-QP 的标准形式

---

## Stage 2 CLF+CBF 在 WBC/MPC 中的应用

### 核心问题
- 如何把 CLF 和 CBF 约束嵌入 WBC 的 QP 层
- Safety filter 和 MPC safety constraint 有什么区别

### 推荐读什么
- [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
- [Query：CLF+CBF 在 WBC/MPC 中联合使用](../../wiki/queries/clf-cbf-in-wbc.md)
- Zeng et al., *Safety-Critical Model Predictive Control* (2021)

### 推荐做什么
- 在一个简单 locomotion 环境中加入 CBF safety filter，观察对步态的影响

### 学完输出什么
- 能描述 safety filter 的工作方式
- 能识别 WBC 中哪些约束层可以加 CLF/CBF 项

---

## Stage 3 Safe RL

### 核心问题
- Constrained MDP（CMDP）和标准 MDP 的区别
- 如何用 Lagrangian 方法或 barrier 方法训练安全策略

### 推荐读什么
- Garcia & Fernandez, *A Comprehensive Survey on Safe RL* (2015)
- [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 推荐做什么
- 用 Safety Gym（OpenAI）或 safe-control-gym 跑一个 constrained RL 实验

### 学完输出什么
- 能解释 CMDP 的形式化定义
- 能对比 model-based 安全保证 vs. model-free 安全奖励塑形的优劣

---

## 关联页面

- [Lyapunov 稳定性](../../wiki/formalizations/lyapunov.md)
- [Control Lyapunov Function](../../wiki/formalizations/control-lyapunov-function.md)
- [Control Barrier Function](../../wiki/concepts/control-barrier-function.md)
- [CLF vs CBF 对比](../../wiki/comparisons/clf-vs-cbf.md)
- [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
- [Query：CLF+CBF 在 WBC/MPC 中联合使用](../../wiki/queries/clf-cbf-in-wbc.md)
