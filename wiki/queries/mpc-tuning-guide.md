---
type: query
tags: [mpc, control, optimization, tuning, locomotion]
status: complete
updated: 2026-11-07
related:
  - ../methods/model-predictive-control.md
  - ../methods/trajectory-optimization.md
  - ../concepts/optimal-control.md
sources:
  - ../../sources/papers/feedback_mpc_anymal_arxiv_1905_06144.md
  - ../../sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md
  - ../../sources/papers/crocoddyl_mastalli_icra_2020.md
  - ../../sources/papers/acados_arxiv_1910_13753.md
  - ../../sources/papers/mpc.md
  - ../../sources/papers/pi_mpc_arxiv_2601_14414.md
original:
  - ../../sources/papers/feedback_mpc_anymal_arxiv_1905_06144.md
  - ../../sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md
  - ../../sources/papers/crocoddyl_mastalli_icra_2020.md
  - ../../sources/papers/acados_arxiv_1910_13753.md
summary: "MPC 调参指南：系统梳理了预测时域、代价函数权重 Q/R 矩阵、更新率与反馈层、摩擦锥约束对机器人稳定性的影响，并结合 ANYmal / Mini Cheetah / Crocoddyl / acados 原文数据点给出调试顺序。"
---

# Model Predictive Control (MPC) 调参指南

> **Query 产物**：本页由以下问题触发：「我的线性 MPC 跟踪很不稳，Base 晃动严重，应该先调哪个参数？Q 和 R 怎么给值？」
> 综合来源：[MPC 核心页](../methods/model-predictive-control.md)、[Optimal Control](../concepts/optimal-control.md)

> **读法（2026-11-07 对照原文后）**：本页把"已被原文数据支持的结论"和"经验性说法"分开标注。经验性说法统一写作"（推测）"，不能当作定论引用。上一版引用的 Di Carlo 等 2018 (Cheetah 3) 本次**没有取到全文**，其中的权重排序与时域数值因此降级为经验说法。

调节 MPC 控制器本质上是在**跟踪性能**、**控制平滑度**与**计算开销**之间寻找平衡。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| SLQ | Sequential Linear Quadratic | ETH 的 DDP 型实时 MPC 算法，输出前馈与 Riccati 反馈增益 |
| RTI | Real-Time Iteration | 每个采样时刻只做一次完整 SQP 迭代的实时 NMPC 方案 |
| FDDP | Feasibility-driven Differential Dynamic Programming | Crocoddyl 的可行性驱动 DDP，接受不可行初值 |
| WBIC | Whole-Body Impulse Control | MIT 的全身冲量控制，跟踪 MPC 给出的地面反力 |

## 1. 权重矩阵的设计 (Q & R)

MPC 的目标是最小化 $J = \sum (x^T Q x + u^T R u)$（MIT 凸 MPC 的写法是 $\sum \|x(k+1)-x_{ref}(k+1)\|_Q + \|f(k)\|_R$，ETH 反馈 MPC 实验里"所有实验用对角代价"）。

- **状态权重 Q (State Penalty)**：
  - **Base Orientation (姿态)**：一般给较高权重——足式机器人姿态一旦偏离，重力补偿会失效（经验说法，推测）。
  - **Velocity (速度)**：中等权重（推测）。
  - **Position (位置)**：通常较低（推测）。
  - 原文没有给出这些权重的具体数值：Kim 2019 与 Grandia 2019 都只说明用对角代价，没有公布 Q/R 数值。
- **动作权重 R (Control Penalty)**：
  - 代表你对电机"出力"（或接触力）的厌恶程度。
  - R 过小：动作剧烈，容易引起高频震荡和硬件损耗；
  - R 过大：动作迟缓，像是在"泥潭"里行走（两条均为经验说法，推测）。
- **有原文支持的另一条旋钮：频率整形代价（Grandia 2019）**。对输入加高通滤波器 $r_i(\omega)=(1+\beta_i j\omega)/(1+\alpha_i j\omega)$（$\beta_i>\alpha_i$），让高频输入变贵、直接状态到接触力的反馈增益下降约一个数量级；实验取 $\alpha=0.01$，接触力 $\beta=0.2$，关节速度 $\beta=0.1$。作者报告：直接把 SLQ-MPC 的 LQR 增益用在真机上，系统即使在全支撑相也会失稳，加频率整形后才部署成功。

## 2. 预测时域 (Horizon) 与步长 (dt)

- **时域 (T)**：预测未来多久的运动。已核对的数据点：
  - Grandia 2019（ANYmal 动态步行）：**1.0 s** 滚动时域，MPC 更新约 **15 Hz**；
  - Mastalli 2020（Crocoddyl，离线 / 预定接触序列的机动）：积分步长通常 $10^{-2}$ s（双足行走 $3\times10^{-2}$ s），节点 60–115 个，对应时域 **0.6–3 s**；
  - 上一版"1–2 个步态周期 (0.4–0.8 s)"没有原文支持（推测）。
  - 时域越长，全局观越好（能预见接触切换）；但**计算开销并非"三次方增长"**：Crocoddyl 报告单次迭代耗时随节点数**线性**增长；acados 论文说明凝聚法的复杂度对时域长度是**二次**的，结构化（Riccati / 内点法）求解对时域为线性（π MPC 引言同样指出 Riccati 递推对时域线性但串行）。
- **步长 (dt)**：**MPC 的离散步长、MPC 更新率与底层控制频率是三个不同的量**，不要默认一致：
  - MIT WBIC：MPC 约 30–40 Hz，WBIC 500 Hz，关节 PD 40 kHz；
  - ETH：底层控制 400 Hz，MPC 异步运行在 15–20 Hz；
  - Crocoddyl：离散步长 10 ms 量级（与更新率无关）。
  - 上一版"步长与控制周期一致（10–30 ms）"只在部分系统成立（推测）。

## 3. 更新率不够时：用反馈而不是只用前馈

Grandia 2019 的核心实验：在 10% 质量模型误差的仿真里，前馈 MPC 更新率从 400 Hz 降到 20 Hz 时，每次更新处的期望加速度出现明显不连续（累积偏差被重置）；改用 SLQ 的 **Riccati 反馈策略**后，不连续显著减小，轨迹接近 400 Hz 的效果。真机上放 5.7 kg（约总质量 15%）负载，前馈 + 常规跟踪器出现力矩跳变，反馈策略则没有。MIT 的思路是另一条路：低频 MPC 给出反力，外挂 500 Hz 的 WBIC 做高频反馈（见 [MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md)）。

## 4. 约束（摩擦锥）怎么放

- 不加锥约束时，最优解可能在后腿给出**负接触力**，以便尽早到达期望的身体位置（Grandia Fig.5）；
- 做法一：**松弛障碍函数**把二阶锥并入代价（固定 $\mu=0.5$、$\delta=0.1$ 时约束均严格满足；用有限 $\mu$ 而不是趋于零，可用少许最优性换取更大稳定裕度）；
- 做法二：MIT 凸 MPC 用金字塔近似 $|f_x|\le\mu f_z,|f_y|\le\mu f_z,f_z>0$；并通过**删掉非接触足的力变量**使问题缩小，原文称提速超过 10 倍；
- Crocoddyl 的基准**刻意忽略**摩擦锥与力矩限，作者称可用二次惩罚加入——所以那些毫秒量级的求解时间不含约束开销。

## 5. 推荐调试顺序（经验流程，推测）

1. **先调静态平衡**：将期望速度设为 0，调高姿态权重 Q，直到机器人能稳稳对抗推力。
2. **增加控制平滑度**：逐渐增大 R（或加频率整形），直到足端接地力不再发生剧烈跳变。
3. **开启动态跟踪**：引入速度指令，调优速度跟踪项的权重。
4. **最后核对更新率与反馈层**：若 MPC 更新 ≤ 20 Hz，优先考虑反馈策略或外挂高频跟踪器，而不是继续堆权重。

## 关联页面
- [Model Predictive Control (MPC)](../methods/model-predictive-control.md)
- [Trajectory Optimization (轨迹优化)](../methods/trajectory-optimization.md)
- [MPC 与 WBC 集成](../concepts/mpc-wbc-integration.md)
- [MPC 求解器选型指南](./mpc-solver-selection.md)

## 参考来源
- [Grandia et al. 2019 — Feedback MPC 通读归档](../../sources/papers/feedback_mpc_anymal_arxiv_1905_06144.md)
- [Kim et al. 2019 — WBIC + MPC（Mini Cheetah）通读归档](../../sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md)
- [Mastalli et al. 2020 — Crocoddyl 通读归档](../../sources/papers/crocoddyl_mastalli_icra_2020.md)
- [Verschueren et al. — acados 通读归档](../../sources/papers/acados_arxiv_1910_13753.md)
- [sources/papers/mpc.md](../../sources/papers/mpc.md)（二手整理；其中 Di Carlo 等 2018 本次未取到全文）

## 推荐继续阅读
- Grandia et al., *Feedback MPC for Torque-Controlled Legged Robots*：<https://arxiv.org/abs/1905.06144>
- Kim et al., *Highly Dynamic Quadruped Locomotion via WBIC and MPC*：<https://arxiv.org/abs/1909.06586>
- Di Carlo et al., *Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control*（IROS 2018，本次未读全文）
