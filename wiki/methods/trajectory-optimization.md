---
type: method
tags: [control, optimization, motion-planning, trajectory-optimization, mpc, humanoid]
status: complete
updated: 2026-04-20
related:
  - ./model-predictive-control.md
  - ../concepts/optimal-control.md
  - ../concepts/whole-body-control.md
  - ../formalizations/zmp-lip.md
  - ../comparisons/trajectory-opt-vs-rl.md
sources:
  - ../../sources/papers/optimal_control.md
  - ../../sources/papers/mpc.md
summary: "轨迹优化（Trajectory Optimization）通过数值优化方法求解开环最优控制序列，是 MPC 的理论核心和离线运动规划的主力工具。"
---

# Trajectory Optimization（轨迹优化）

**轨迹优化（Trajectory Optimization）** 是一种通过数值方法求解最优控制问题（OCP）的技术。它的目标是找到一条最优的状态轨迹 $\mathbf{x}(t)$ 和控制序列 $\mathbf{u}(t)$，使得在满足系统动力学、物理约束和任务约束的前提下，最小化给定的代价函数。

## 核心思想

将连续时间的控制问题转化为有限维度的非线性规划（NLP）问题。

- **输入**：系统模型（动力学）、初值、目标状态、代价函数、硬约束。
- **输出**：一条完整的时空一致的运动轨迹。

## 与 MPC 的关系

轨迹优化是 **MPC（模型预测控制）** 的算法内核。
- **轨迹优化**：通常关注于长时域的开环求解（离线规划）。
- **MPC**：在每个控制周期内运行一个短时域的轨迹优化，并仅执行第一步动作，利用反馈消除误差。

## 主要技术路线

### 1. 直接法 (Direct Methods)
将状态和控制离散化，直接作为优化变量。
- **直接打靶法 (Direct Shooting)**：仅优化控制变量 $\mathbf{u}$，状态由前向仿真得到。
- **直接配点法 (Direct Collocation)**：同时优化状态 $\mathbf{x}$ 和控制 $\mathbf{u}$，将动力学方程作为等式约束施加在相邻点之间。适合处理高维、多约束问题（如人形机器人接触规划）。

### 2. 差分动态规划 (DDP / iLQR)
利用 Bellman 最优性原理，通过二阶（DDP）或一阶（iLQR）泰勒展开局部逼近代价函数和动力学。
- **优点**：计算效率极高，复杂度随时间步长线性增长。
- **代表框架**：Crocoddyl。

## 在机器人中的典型应用

1. **人形机器人离线动捕模仿**：将人类动捕数据作为参考，通过 TO 产生符合物理规律的平衡步态。
2. **高动态动作发现**：如翻跟头、跳跃、跑酷。这类动作涉及复杂的非线性动力学和接触切换。
3. **接触隐式优化 (Contact-Implicit Optimization)**：同时优化接触发生的时刻和力度，而不需要预设步态模式。

## 方法局限性

- **局部最优**：TO 通常只能找到局部最优解，对初始值的依赖非常强。
- **实时性挑战**：全动力学的轨迹优化计算量巨大，往往难以达到 1kHz 的闭环控制需求（因此通常需要简化模型或结合 WBC）。
- **模型依赖**：高度依赖精确的刚体动力学模型和接触参数。

## 关联页面
- [Model Predictive Control](./model-predictive-control.md)
- [Optimal Control (OCP) 概念](../concepts/optimal-control.md)
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [ZMP + LIP 形式化](../formalizations/zmp-lip.md)
- [对比：轨迹优化 vs RL](../comparisons/trajectory-opt-vs-rl.md)

## 参考来源
- Betts, J. T. (2010). *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*.
- Posa, M., et al. (2014). *A direct method for trajectory optimization of rigid bodies through contact*.
- [sources/papers/optimal_control.md](../../sources/papers/optimal_control.md)
