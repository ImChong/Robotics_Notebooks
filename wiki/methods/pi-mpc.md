---

type: method
tags: [mpc, solver, admm, nmpc, gpu, parallel-computing, tsinghua, horizon-robotics]
status: complete
updated: 2026-06-10
related:
  - ./model-predictive-control.md
  - ../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md
  - ../queries/mpc-solver-selection.md
  - ./trajectory-optimization.md
  - ../concepts/optimal-control.md
  - ../formalizations/lqr.md
sources:
  - ../../sources/papers/pi_mpc_arxiv_2601_14414.md
  - ../../sources/papers/mpc_rl_arxiv_2606_05687.md
summary: "π MPC（arXiv:2601.14414）：parallel-in-horizon、construction-free 的 ADMM NMPC；velocity-form + 变量分裂实现时域逐步闭式并行更新；πⁿ MPC 扩展支撑 MPC-RL 大规模 GPU 批训练。"
---

# π MPC（Parallel-in-horizon、Construction-free NMPC）

**π MPC**（*π MPC: A Parallel-in-horizon and Construction-free NMPC Solver*，arXiv:2601.14414，JHU · Tsinghua · Caltech）是一种面向 MPC 的 **ADMM 求解算法**：通过 **新变量分裂** 与 **velocity-form（输入增量）** 表示，使每个预测步的 ADMM 子问题 **闭式可解**，并在时域索引 $k=0,\ldots,N-1$ 上 **并行执行**；**无需** 将 MPC 在线组装为大型稀疏 QP。

## 一句话定义

**把 MPC 的 ADMM 迭代拆成「逐步可并行、逐步可闭式」——长时域不再被 Riccati 串行或 QP 构造拖慢。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| π MPC | pi-MPC | 本文 parallel-in-horizon ADMM NMPC 算法 |
| πⁿ MPC | pi-n-MPC | 时变动力学矩阵的批 GPU 扩展（MPC-RL） |
| ADMM | Alternating Direction Method of Multipliers | 交替方向乘子法，本文求解框架 |
| NMPC | Nonlinear Model Predictive Control | 非线性 MPC；π MPC 作用于线性化/RTI 子问题 |
| RTI | Real-Time Iteration | 实时迭代 NMPC，每步线性化后解 QP 型子问题 |
| GPU | Graphics Processing Unit | πⁿ MPC 批并行执行设备 |

## 为什么重要

- **突破 Riccati 串行瓶颈：** 结构 exploit 型 ADMM 虽 **construction-free** 且 **线性于 $N$**，但 Riccati 递推 **难以并行**；π MPC 使 **主计算** 在时域上并行。
- **避免 MPC→QP 在线构造：** 一般 ADMM 解稀疏 QP 时，$(H+\rho E^\top E)^{-1}$ 对时变 NMPC **立方于时域**；π MPC 直接操作 $\{A_{t,k},B_{t,k},e_{t,k}\}$，适合 **RTI / 线性时变** 子问题。
- **参数选择简单：** ADMM 对任意 $\rho>0$ 收敛，无需像梯度法估计 Lipschitz 常数——利于嵌入式与 **时变 Hessian** 场景。
- **支撑 RL 内嵌 MPC：** [MPC-RL](../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md) 的 **πⁿ MPC**（PyTorch/JAX）在 **4096 环境 × 长 horizon** 下仍可批解，相对 qpth / CusADi 显著降低 **VRAM 与预编译** 成本。

## 主要技术路线

### 1. 变量分裂（parallel-in-horizon 的关键）

在标准 LTV MPC 上引入：

- $x_{k+1}-z_{k+1}=0$（状态副本）
- $B_{t,k}u_k-v_k=0$（**$B u$ 副本**，而非 $u$ 副本——分裂设计要点）
- $z_{k+1}-A_{t,k}x_k-v_k-e_{t,k}=0$（动力学写入 $z$ 链）

使 **(8a)(8b)(8c)** 三类更新均可在 $k$ 上并行。

### 2. Velocity-form 闭式更新

提升状态 $\bar{x}_k=[x_k;u_{k-1}]$，优化 **输入增量** $\Delta u_k$：

$$\bar{x}_{k+1}=\bar{A}_{t,k}\bar{x}_k+\bar{B}_{t,k}\Delta u_k+\bar{e}_{t,k}$$

配合 Theorem 2，**所有 ADMM 子步闭式**（含 $(\bar{B}^\top\bar{B})^{-1}$ 与 horizon-local $H_k$）。

### 3. 加速与实现

- **Nesterov 加速 ADMM + restart**（Goldstein et al.）
- 原始/对偶残差停止准则
- **πⁿ MPC：** 时变 $\{A_k,B_k,e_k\}$ 沿 horizon 存储；PyTorch/JAX **batch 维 = 环境 × 时域**

### 算法数据流（单步 ADMM）

```mermaid
flowchart LR
  subgraph parallel_k [并行于 k=0..N-1]
    DU[闭式更新 Δu_k]
    XBAR[闭式更新 x̄_{k+1}]
    ZV[闭式更新 z_{k+1}, v_k\nProj + 平均]
    DUAL[更新 θ, β, λ]
  end
  DU --> XBAR --> ZV --> DUAL
  DUAL -->|下一 ADMM 迭代| DU
```

## 与常见求解器对比（选型速览）

| 求解器 | 并行于时域 | Construction-free | 典型瓶颈 |
|--------|------------|-------------------|----------|
| **π MPC / πⁿ MPC** | ✅ | ✅ | ADMM 迭代次数 |
| Riccati ADMM | ❌ 串行 | ✅ | 长 horizon 延迟 |
| 一般稀疏 QP ADMM | 部分 | ❌ 需组装 $E,H$ | 立方逆、VRAM |
| CusADi | 环境并行 | ❌ 需符号编译 | 编译时间与内存 |
| qpth / qpax | 环境并行 | ❌ | 长 horizon OOM |

详见 [MPC 求解器选型](../queries/mpc-solver-selection.md)。

## 常见误区或局限

- **误区：** π MPC 等于「任意非线性全身 MPC 的万能解」；实际用于 **凸 LTV / RTI 线性化** 子问题（如质心 MPC）。
- **误区：** 并行等于「比 Riccati 总时间一定更短」；短 horizon 或少量环境时，ADMM 迭代开销可能仍需 profile。
- **局限：** 论文为 technical note 级算法贡献；真机大规模验证主要通过 [MPC-RL](../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md) 训练栈体现。

## 关联页面

- [MPC-RL 论文实体](../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md) — πⁿ MPC 的主要应用与批 GPU 实现
- [Model Predictive Control](./model-predictive-control.md) — MPC 问题 formulation 背景
- [Optimal Control (OCP)](../concepts/optimal-control.md) — MPC 所求解的有限时域 OCP 背景
- [LQR](../formalizations/lqr.md) — Riccati 递推对照路线
- [MPC vs RL](../comparisons/mpc-vs-rl.md) — 训练期 MPC 指导范式
- [MPC 求解器选型](../queries/mpc-solver-selection.md) — OSQP / Acados / Crocoddyl 等对照

## 参考来源

- [π MPC 论文摘录（arXiv:2601.14414）](../../sources/papers/pi_mpc_arxiv_2601_14414.md)
- [MPC-RL 论文摘录（arXiv:2606.05687）](../../sources/papers/mpc_rl_arxiv_2606_05687.md)

## 推荐继续阅读

- [π MPC 论文（arXiv:2601.14414）](https://arxiv.org/abs/2601.14414)
- [MPC-RL 论文与代码](https://arxiv.org/abs/2606.05687) · [github.com/junhengl/mpc-rl](https://github.com/junhengl/mpc-rl)
- OSQP / Acados 文档 — 对照传统 QP/NLP MPC 后端
