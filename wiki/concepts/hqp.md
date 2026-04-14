---
type: concept
tags: [wbc, tsid, optimization, qp, humanoid, control]
status: complete
---

# HQP（Hierarchical QP）

**分层二次规划（Hierarchical Quadratic Programming，HQP）**：全身控制（WBC）中处理多任务优先级冲突的优化框架，通过将任务按优先级分层求解，确保高优先级任务精确满足，低优先级任务在不影响高优先级的前提下尽量完成。

## 一句话定义

> HQP 回答的是："当机器人同时有多个目标，而且这些目标互相冲突时，应该优先满足哪个？" — 答案是分层，先保命（平衡），再走路，再优雅（手臂姿态）。

## 为什么重要

人形机器人全身控制的核心挑战是**任务冲突**：

- 躯干稳定 vs. 摆腿速度
- 精确轨迹跟踪 vs. 关节力矩限制
- 保持接触 vs. 全身动作幅度

单层加权 QP（把所有任务加权相加）无法保证高优先级任务被**精确满足**——权重调整只能做软性平衡。HQP 提供了**硬性优先级**的解决方案，是 TSID、全身控制框架的核心求解机制。

## 问题形式化

### 单层加权 QP（对比用）

$$\min_z \sum_i w_i \| A_i z - b_i \|^2 \quad \text{s.t.} \quad C z \leq d$$

问题：权重比值再大，也不能保证任务 $i$ 精确满足。

### HQP：分层求解

设有 $L$ 个优先级层次，层 $l$ 的任务为 $A_l z = b_l$：

**层 1（最高优先级）**：

$$z_1^* = \arg\min_z \| A_1 z - b_1 \|^2 \quad \text{s.t.} \quad C z \leq d$$

**层 2**（在不影响层 1 的前提下）：

$$z_2^* = \arg\min_z \| A_2 z - b_2 \|^2 \quad \text{s.t.} \quad A_1 z = A_1 z_1^*, \; C z \leq d$$

即：层 2 的解在层 1 的**零空间（null space）**中求最优。

**层 $l$**（一般形式）：

$$z_l^* = \arg\min_z \| A_l z - b_l \|^2 \quad \text{s.t.} \quad A_k z = A_k z_{l-1}^* \; \forall k < l, \; C z \leq d$$

## 人形控制中的典型优先级结构

```text
Priority 1（最高）：动力学约束 + 接触约束
    → M(q)q̈ + h = Sᵀτ + Jcᵀf
    → Jc q̈ + J̇c q̇ = 0

Priority 2：CoM 跟踪 + 躯干姿态稳定
    → 保持平衡，防止摔倒

Priority 3：摆动腿末端轨迹跟踪
    → 精确控制落脚点

Priority 4：手臂/上身动作
    → 在不影响平衡和步态的前提下做动作

Priority 5（最低）：关节正则化 / 零空间姿态
    → 避免奇异、关节限位保护
```

**一句话记忆**：先保命，再走路，再优雅。

## HQP vs. 加权 QP

| 维度 | 加权 QP | HQP |
|------|---------|-----|
| 优先级表达 | 软性（权重） | 硬性（分层） |
| 高优先级保证 | ❌（只是权重大） | ✅（精确满足） |
| 计算复杂度 | 低（单个 QP） | 高（多个 QP） |
| 实现难度 | 简单 | 较复杂 |
| 适用场景 | 任务不严格冲突 | 严格优先级需求 |

## 实时求解挑战

高自由度人形（30+ DoF）+ 高频控制（500-1000 Hz）下，HQP 的求解压力不小：

- 每层都是一个 QP，多层叠加
- 矩阵规模随 DoF 增大
- 接触切换时矩阵维度变化

**常见优化方案**：

1. **稀疏求解器**：利用系统矩阵稀疏结构（Acados、qpOASES、eiquadprog）
2. **热启动**：用上步解作为下步初值
3. **任务层数限制**：一般 3-5 层已足够，不做过深分层
4. **近似 HQP**：用加权方案做一层 QP，但权重设计成近似层次效果（折中方案）

## 和 TSID 的关系

TSID（Task Space Inverse Dynamics）是典型的 HQP 实现框架：

- TSID 把任务目标写成加速度层约束
- 求解一个（或分层的）QP 得到 $[\ddot{q}, \tau, f]$
- stack-of-tasks 库实现了完整的 HQP 控制器

见：[TSID](./tsid.md)

## 典型开源实现

| 库 | 说明 |
|----|------|
| [TSID](https://github.com/stack-of-tasks/tsid) | 基于 Pinocchio 的 TSID + HQP 实现 |
| [Crocoddyl](https://github.com/loco-3d/crocoddyl) | DDP-based，不是严格 HQP 但有优先级结构 |
| [mc_rtc](https://jrl-umi3218.github.io/mc_rtc/) | 工业级全身控制框架，内置 HQP |

## 关联页面

- [TSID](./tsid.md) — HQP 是 TSID 的核心求解机制
- [Whole-Body Control](./whole-body-control.md) — HQP 是 WBC 任务冲突处理的主流方案
- [MPC 与 WBC 集成](./mpc-wbc-integration.md) — 典型控制架构中 WBC 层用 HQP 求解
- [LQR / iLQR](../formalizations/lqr.md) — LQR 是单任务最优控制解析解；HQP 是多任务优先级控制的数值求解
- [Crocoddyl](../entities/crocoddyl.md) — 相关最优控制求解框架

## 参考来源

- Kanoun et al., *Kinematic Control of Redundant Manipulators: Generalizing the Task-Priority Framework to Inequality Task* (2011) — HQP 优先级框架理论基础
- Del Prete et al., *Prioritized motion-force control of constrained fully-actuated robots: "Task Space Inverse Dynamics"* — TSID + HQP 的核心实现论文
- Escande et al., *Hierarchical Quadratic Programming: Fast Online Humanoid-Robot Motion Generation* (2014) — 人形机器人 HQP 实时求解
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md) — HQP 相关来源整理

## 推荐继续阅读

- [TSID](./tsid.md)
- [Whole-Body Control](./whole-body-control.md)
- TSID library: <https://github.com/stack-of-tasks/tsid>

## 一句话记忆

> HQP 是全身控制的"优先级调度器"——确保关键约束（平衡、接触）被精确满足，再用剩余的自由度去完成次要任务，是人形机器人多任务冲突处理的主流框架。
