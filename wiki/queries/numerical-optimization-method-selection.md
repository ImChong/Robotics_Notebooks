---
type: query
tags: [optimization, trajectory-optimization, optimal-control, constrained-optimization, method-selection]
status: complete
updated: 2026-06-24
related:
  - ../methods/lqr-ilqr.md
  - ../methods/quasi-newton-bfgs.md
  - ../methods/penalty-barrier-augmented-lagrangian.md
  - ../methods/convex-relaxation-robotics.md
  - ./optimization-software-selection.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "机器人数值优化方法选型：按问题结构（含时序动力学 / 一般 NLP / 约束处理 / 非凸几何）在 LQR·iLQR、拟牛顿 BFGS、罚/障碍/增广拉格朗日、凸松弛四类算法间分层组合。"
---

# Numerical Optimization Method Selection（机器人数值优化方法选型）

> **Query 产物**：本页由以下问题触发：「机器人里那么多数值优化方法——LQR/iLQR、拟牛顿、罚函数/增广拉格朗日、凸松弛——到底该按什么标准选、又怎么组合？」
> 综合来源：[LQR/iLQR](../methods/lqr-ilqr.md)、[拟牛顿 BFGS](../methods/quasi-newton-bfgs.md)、[罚/障碍/增广拉格朗日](../methods/penalty-barrier-augmented-lagrangian.md)、[凸松弛](../methods/convex-relaxation-robotics.md)。与 [优化软件选型](./optimization-software-selection.md) 互补（本页选**算法**，那页选**实现/求解器**）。

## 一句话结论

**这四类不是互斥选项，而是叠在一起的四层**：先看问题有没有「沿时间的动力学」——有就用 [LQR/iLQR](../methods/lqr-ilqr.md) 这类**结构利用型**求解器；没有结构、就是一般无约束/弱约束 NLP，用 [拟牛顿 BFGS/L-BFGS](../methods/quasi-newton-bfgs.md) 当默认下降引擎；**约束怎么进目标**由 [罚/障碍/增广拉格朗日](../methods/penalty-barrier-augmented-lagrangian.md) 决定（包在前两者外面）；而当问题**本质非凸且要全局保证或好初值**时，先用 [凸松弛](../methods/convex-relaxation-robotics.md) 求下界/近似解再收紧。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OCP | Optimal Control Problem | 含动力学约束的轨迹优化 |
| NLP | Nonlinear Programming | 一般非线性规划 |
| DDP | Differential Dynamic Programming | iLQR 的二阶/带约束推广 |
| L-BFGS | Limited-memory BFGS | 高维 TrajOpt 默认下降 |
| ALM | Augmented Lagrangian Method | 增广拉格朗日（PHR） |
| IPM | Interior-Point Method | 内点/障碍法 |
| SDP | Semidefinite Program | 凸松弛常用形式 |
| GNC | Graduated Non-Convexity | 凸松弛中逐步收紧非凸 |

## 选型决策树

```
问题有沿时间的动力学约束（OCP / TrajOpt）？
├─ 是
│   ├─ 系统近线性、二次代价 → LQR（一步 Riccati 解析反馈）
│   ├─ 非线性动力学、要轨迹 → iLQR / DDP（Crocoddyl）
│   └─ 有硬路径约束（接触力、关节限位）
│         → iLQR/DDP 外套 增广拉格朗日 / 障碍法
└─ 否（一般 NLP，无显式时序结构）
    ├─ 无约束 / 罚到无约束、维度高 → 拟牛顿 L-BFGS（cuRobo 默认）
    ├─ 有等式/不等式约束
    │     ├─ 想要内点 → 障碍法 / IPM
    │     └─ 想避免每步解硬 KKT → 增广拉格朗日（PHR 乘子更新）
    └─ 本质非凸（旋转、对应关系、组合）且要全局性 / 好初值
          → 凸松弛（SDP/SOCP）求下界，再 GNC / 局部细化收紧
```

## 四类方法对照

| 方法族 | 解决的核心问题 | 典型机器人场景 | 何时**不**用 |
|--------|---------------|---------------|-------------|
| [LQR / iLQR / DDP](../methods/lqr-ilqr.md) | 利用时序+动力学结构的 OCP | 足式 MPC、全身 TrajOpt（Crocoddyl） | 问题无动力学/时序结构时是杀鸡用牛刀 |
| [拟牛顿 BFGS / L-BFGS](../methods/quasi-newton-bfgs.md) | 一般高维 NLP 的快速下降 | GPU 运动生成（cuRobo）、打靶后 NLP | 强非凸需全局性时易陷局部极小 |
| [罚 / 障碍 / 增广拉格朗日](../methods/penalty-barrier-augmented-lagrangian.md) | 把约束「推进」目标的序列无约束化 | NMPC 约束处理、锥规划、接触约束 | 约束本就凸且求解器原生支持时多此一举 |
| [凸松弛](../methods/convex-relaxation-robotics.md) | 非凸问题的可解近似 / 全局下界 | 位姿估计、点云配准、抓取规划 | 实时回路里 SDP 太慢，仅作离线/初值 |

## 分层组合的常见配方

- **足式 MPC**：凸 MPC（QP）或 iLQR/DDP 为内核，硬约束用 [增广拉格朗日](../methods/penalty-barrier-augmented-lagrangian.md) 软化 → 实时 SQP-RTI。
- **机械臂运动生成**：[凸松弛](../methods/convex-relaxation-robotics.md) 或采样给初值 → [L-BFGS](../methods/quasi-newton-bfgs.md) 在 GPU 上批量细化（cuRobo 范式）。
- **位姿/配准类感知**：[凸松弛](../methods/convex-relaxation-robotics.md) 求全局下界 → GNC 抗外点 → 局部 [拟牛顿](../methods/quasi-newton-bfgs.md) 精修。
- **离线全身 TrajOpt**：[iLQR/DDP](../methods/lqr-ilqr.md) 主循环 + 障碍/AL 处理接触与限位。

## 常见误区

- **把四类当成「四选一」**：它们在不同层（重构 / 约束处理 / 下降 / 结构利用），实际是叠加的。
- **对有动力学结构的 OCP 用通用 NLP**：丢掉 Riccati 的稀疏结构，慢且数值差——优先 [iLQR/DDP](../methods/lqr-ilqr.md)。
- **纯外点罚函数硬调权重**：病态且难收敛，改用 [增广拉格朗日](../methods/penalty-barrier-augmented-lagrangian.md) 的乘子更新。
- **把凸松弛当最终解**：松弛通常给的是下界/近似，需 GNC 或局部细化恢复非凸可行解。

## 关联页面

- [LQR / iLQR](../methods/lqr-ilqr.md) — 结构利用型最优控制内核（Riccati / DDP）
- [拟牛顿 BFGS / L-BFGS](../methods/quasi-newton-bfgs.md) — 高维 NLP 默认下降引擎
- [罚 / 障碍 / 增广拉格朗日](../methods/penalty-barrier-augmented-lagrangian.md) — 约束处理与序列无约束化
- [凸松弛](../methods/convex-relaxation-robotics.md) — 非凸问题的可解近似与全局下界
- [优化软件选型](./optimization-software-selection.md) — 选定算法后选求解器/框架
- [MPC 求解器选型](./mpc-solver-selection.md) — 凸 MPC / WBC 的 QP 求解器专项

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 数值优化基础课程（约束化、拟牛顿、最优控制、凸松弛各章）
- Crocoddyl / cuRobo 官方文档（iLQR/DDP 与 GPU L-BFGS 工程实现）
