# MPC-RL（arXiv:2606.05687）

> 来源归档（ingest）

- **标题：** Accelerating and Scaling MPC-Guided Reinforcement Learning for Humanoid Locomotion and Manipulation
- **缩写：** **MPC-RL**
- **类型：** paper / humanoid / locomotion / loco-manipulation / mpc-guided-rl
- **arXiv：** <https://arxiv.org/abs/2606.05687>
- **PDF：** <https://arxiv.org/pdf/2606.05687>
- **视频：** <https://youtu.be/PrcbXkA1kYg>
- **代码：** <https://github.com/junhengl/mpc-rl>
- **作者：** Junheng Li†, Liang Wu†, Sergio A. Esteban, Lizhi Yang, Ján Drgoňa, Aaron D. Ames（† 共同一作）
- **机构：** Caltech；Johns Hopkins University
- **入库日期：** 2026-06-10
- **一句话说明：** 训练期 **质心 MPC（CD-MPC）** 为大规模并行 PPO 提供 **预测地标奖励**；配套 **πⁿ MPC** GPU 批求解器实现 **无构造、并行于时域** 的长时域 MPC；部署时 **纯 RL 策略**（无 MPC 在线）；Themis 真机验证行走、推恢复、负重与 **290 kg 推车** loco-manipulation。

## 核心论文摘录（MVP）

### 1) 动机与总架构（Abstract / §I–II）

- **链接：** <https://arxiv.org/abs/2606.05687>
- **核心贡献：** MPC 提供物理预测与约束处理，RL 通过大规模仿真获得鲁棒全身技能；但训练期嵌入 MPC 往往 **问题构造耗时** 或 **训练开销过大**。MPC-RL 在 **训练时** 用质心动力学 MPC 生成预测参考，转为 **landmark guidance reward** 监督 PPO；**部署时** 仅运行学得的关节位置增量策略（无 MPC 闭环）。
- **对 wiki 的映射：**
  - [MPC-RL 论文实体](../../wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md)
  - [MPC vs RL](../../wiki/comparisons/mpc-vs-rl.md) — 训练期 MPC 指导 + 推理期纯 RL 的混合范式
  - [Centroidal Dynamics](../../wiki/concepts/centroidal-dynamics.md)

### 2) CD-MPC 与预测置信度加权地标奖励（§II）

- **核心贡献：** 质心状态 $\xi=[c^\top, l_G^\top, k_G^\top]^\top$；离散质心动力学 + 接触力/力矩约束（CWC、摩擦锥）；从 MPC 全轨迹抽取 **$N_L$ 个预测地标** $\delta_\ell$，用 **prediction-confidence weighting** $\alpha_0>\cdots>\alpha_{N_L-1}$ 融合为指数奖励 $r_t^{\mathrm{mpc},c}$；近端地标权重大、远端降权，兼顾长时域结构与模型失配鲁棒性。
- **对 wiki 的映射：**
  - [Model Predictive Control](../../wiki/methods/model-predictive-control.md)
  - [MPC 与 WBC 集成](../../wiki/concepts/mpc-wbc-integration.md)

### 3) πⁿ MPC：并行于时域、无构造批求解器（§III）

- **核心贡献：** 在 [π MPC](https://arxiv.org/abs/2601.14414) 的 **parallel-in-horizon ADMM** 上扩展至 **时变** 腿足动力学矩阵；PyTorch / JAX 实现 **πⁿ MPC**；预测时域作为 batch 维 GPU 并行，避免稀疏 QP 组装与 CusADi 式预编译；支撑 4096 环境 × 长时域 CD-MPC 在 RL 训练内可行。
- **对 wiki 的映射：**
  - [π MPC 方法页](../../wiki/methods/pi-mpc.md)
  - [MPC 求解器选型](../../wiki/queries/mpc-solver-selection.md)

### 4) 训练栈与奖励设计（§IV）

- **核心贡献：** **rsl-rl + mjlab**，4096 并行环境，200 Hz 物理 / 50 Hz 控制；非对称 actor–critic（actor 仅本体感知 + 摇杆命令；critic 特权含 MPC 预测 CoM / 角动量）；MPC 奖励块替代 base-RL 速度跟踪，含 `mpc_com`、`mpc_lin_vel`、`mpc_ang_mom`、`mpc_foot`、`mpc_grf`；**有效惯量 PD 增益**（$K_p=J_{\mathrm{eff}}\omega_n^2$）适配 Themis 高惯量低减速比执行器。
- **对 wiki 的映射：**
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 5) 实验与真机（§V）

- **仿真：** 相对纯 RL，MPC-RL 在时变速度跟踪、推恢复、约束满足上更优；消融 horizon 长度、更新率、奖励结构。
- **真机（Themis）：** 跑步机行走、未知 10 kg 背心/载荷行走、推恢复、**290 kg（829% 体重）推车** box-pushing。
- **对 wiki 的映射：**
  - [MPC-RL 代码仓库](../../sources/repos/junhengl_mpc_rl.md)

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md`](../../wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md)
- 求解器方法页：[`wiki/methods/pi-mpc.md`](../../wiki/methods/pi-mpc.md)
- 互链参考：[π MPC 论文摘录](./pi_mpc_arxiv_2601_14414.md)、[MPC vs RL](../../wiki/comparisons/mpc-vs-rl.md)、[Centroidal Dynamics](../../wiki/concepts/centroidal-dynamics.md)
