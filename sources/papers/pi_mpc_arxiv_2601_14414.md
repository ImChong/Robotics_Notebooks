# π MPC（arXiv:2601.14414）

> 来源归档（ingest）

- **标题：** π MPC: A Parallel-in-horizon and Construction-free NMPC Solver
- **缩写：** **π MPC**（读作 pi-MPC）
- **类型：** paper / mpc / solver / admm / nmpc
- **arXiv：** <https://arxiv.org/abs/2601.14414>
- **PDF：** <https://arxiv.org/pdf/2601.14414>
- **作者：** Liang Wu†, Bo Yang†, Junheng Li†, Xu Yang, Yilin Mo, Yang Shi, Aaron D. Ames, Ján Drgoňa（† 共同一作）
- **机构：** Johns Hopkins University；Tsinghua University；Caltech；University of Victoria
- **入库日期：** 2026-06-10
- **一句话说明：** 基于 ADMM 的 **parallel-in-horizon、construction-free** NMPC：新变量分裂 + **velocity-form** 表示使各时域步 **闭式并行更新**；避免 Riccati 递推的串行瓶颈与 MPC→稀疏 QP 在线组装；含 Nesterov 加速与 restart；为 [MPC-RL](https://arxiv.org/abs/2606.05687) 的 **πⁿ MPC** GPU 批实现提供算法基础。

## 核心论文摘录（MVP）

### 1) 问题与动机（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2601.14414>
- **核心贡献：** ADMM 在嵌入式 MPC 中代码简单、$\rho>0$ 即收敛；但既有 ADMM 要么解一般 QP，要么用 Riccati 结构 exploit 稀疏 MPC-QP——后者 **线性于时域但串行**，长时域难并行。π MPC 目标：**construction-free**（直接作用于 $\{A_{t,k},B_{t,k},e_{t,k}\}$）且 **parallel-in-horizon**。
- **对 wiki 的映射：**
  - [π MPC 方法页](../../wiki/methods/pi-mpc.md)
  - [MPC 求解器选型](../../wiki/queries/mpc-solver-selection.md)

### 2) 变量分裂与 parallel-in-horizon ADMM（§IV）

- **核心贡献：** 引入辅助变量 $z_{k+1}$、$v_k=B_{t,k}u_k$ 的三路分裂；动力学约束写入 $z_{k+1}-\bar{A}_{t,k}\bar{x}_k-v_k-\bar{e}_{t,k}=0$；**Theorem 1** 给出 $(z,v)$ 闭式更新；**Theorem 2** 用 **velocity-form**（优化 $\Delta u_k$，提升状态 $\bar{x}_k=[x_k;u_{k-1}]$）使 $(U,X)$ 更新亦闭式；每步 ADMM 在 $k=0,\ldots,N-1$ **并行**。
- **对 wiki 的映射：**
  - [Model Predictive Control](../../wiki/methods/model-predictive-control.md)
  - [Trajectory Optimization](../../wiki/methods/trajectory-optimization.md)

### 3) 加速与停止准则（§IV-B）

- **核心贡献：** 采用带 restart 的 **Nesterov 加速 ADMM**；基于原始/对偶残差定义停止条件。
- **对 wiki 的映射：**
  - [π MPC 方法页](../../wiki/methods/pi-mpc.md)

### 4) 与 MPC-RL 的关系（后续工作 arXiv:2606.05687）

- **核心贡献：** [MPC-RL](https://arxiv.org/abs/2606.05687) 将 π MPC 扩展为 **πⁿ MPC**：时变质心动力学矩阵沿时域存储、PyTorch/JAX 批 GPU 求解，嵌入 4096 环境 PPO 训练。
- **对 wiki 的映射：**
  - [MPC-RL 论文实体](../../wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md)
  - [junhengl/mpc-rl 仓库](../../sources/repos/junhengl_mpc_rl.md)

## 对 wiki 的映射（汇总）

- 沉淀方法页：[`wiki/methods/pi-mpc.md`](../../wiki/methods/pi-mpc.md)
- 应用实体：[`wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md`](../../wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md)
