---
type: query
tags: [wbc, control, optimization, tuning, humanoid, qp]
status: complete
updated: 2026-11-07
related:
  - ../concepts/whole-body-control.md
  - ../concepts/tsid.md
  - ./wbc-implementation-guide.md
  - ../formalizations/friction-cone.md
sources:
  - ../../sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md
  - ../../sources/papers/delprete_tsid_arxiv_1410_3863.md
  - ../../sources/papers/kanoun_task_priority_inequality_tro_2011.md
  - ../../sources/papers/escande_hqp_ijrr_2014.md
  - ../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md
  - ../../sources/papers/osqp_admm_qp_arxiv_1711_08013.md
  - ../../sources/repos/tsid.md
  - ../../sources/papers/whole_body_control.md
original:
  - ../../sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md
  - ../../sources/papers/delprete_tsid_arxiv_1410_3863.md
  - ../../sources/papers/kanoun_task_priority_inequality_tro_2011.md
  - ../../sources/papers/escande_hqp_ijrr_2014.md
  - ../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md
  - ../../sources/papers/osqp_admm_qp_arxiv_1711_08013.md
  - ../../sources/repos/tsid.md
summary: "WBC 调参指南：通过合理的任务权重分配、约束松弛、正则化技巧与求解器热启动配置，实现高性能全身控制的实践经验。"
---

# WBC 调参指南：从公式到实机表现

> **Query 产物**：本页由以下问题触发：「WBC QP 求解器怎么调参？权重矩阵、约束松弛、热启动有哪些技巧？」
> 综合来源：[Whole-Body Control](../concepts/whole-body-control.md)、[TSID](../concepts/tsid.md)、[WBC Implementation Guide](./wbc-implementation-guide.md)

---

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学求解关节力矩的 WBC 实现 |
| CoM | Center of Mass | 质心，平衡与 locomotion 规划的核心状态量 |
| PD | Proportional–Derivative | 关节位置/阻抗底层控制，策略输出常为其 setpoint |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 1. 权重矩阵的设计 (The "Art of Weights")

WBC 通常是一个多目标优化问题 $\min \|Ax - b\|_W^2$。权重 $W$ 决定了任务的优先级。

### 调参顺序
1. **主优先级任务 (High Priority)**：
   - **躯干姿态 (Floating Base Orientation)**：这是平衡的核心。如果不稳，机器人会迅速侧翻。
   - **支撑足加速度 (Support Foot Consistency)**：必须设为极高权重或作为硬等式约束，否则脚会滑移。
2. **次优先级任务**：
   - **质心轨迹 (CoM Tracking)**：决定了运动的总体走向。
   - **摆动足轨迹 (Swing Foot Tracking)**：确保迈步准确。
3. **低优先级任务 (Regularization)**：
   - **关节姿态正则化 (Joint Posture)**：给出一个倾向的“基础姿势”（如微蹲），防止奇异点。

### 权重比例技巧
- 先按单位与容许误差归一化各任务残差，再比较相对权重；米、弧度与力矩的原始误差不能直接比较。
- `1000 : 1000 : 100 : 100 : 1` 仅可作说明性示例，不是通用调参比例。加权和也不能保证严格优先级；需要严格优先级时应采用分层优化或硬约束。

---

## 2. 约束松弛与惩罚 (Relaxation)

硬约束（如 $Ax=b$）在实际物理碰撞中容易导致 QP 无解（Infeasible）。

### 处理技巧
- **等式约束转化为惩罚项**：将 $Ax=b$ 变为 $\min \lambda \|Ax - b\|^2$。
- **引入松弛变量 (Slack Variables)**：优先对允许暂时偏离的跟踪任务引入松弛；保留非穿透、摩擦与执行器限制等物理约束。放松摩擦锥只能用于诊断约束冲突，不应把超出摩擦锥的解当作可实现的真机命令。
- **权重策略**：对任务松弛设置经归一化的线性或二次惩罚，并监控松弛量与条件数；盲目增大权重会恶化数值条件。

---

## 3. 正则化 (Regularization)

为了提高数值稳定性，尤其是当雅可比矩阵 $J$ 接近奇异时。

- **力矩正则化**：惩罚 $\|\tau\|^2$。防止 QP 求解器为了微小的跟踪精度提升而输出巨大的瞬时力矩。
- **阻尼项**：在 $A^T A$ 中加入 $\epsilon I$。这能保证 Hessian 矩阵始终正定。

---

## 4. 求解器热启动 (Hot Start)

对于 1kHz 的控制环，每一微秒都很珍贵。

- **重复利用上一时刻解**：将 $t-1$ 时刻的解 $x^*$ 作为 $t$ 时刻的初始猜测。
- **求解器策略**：
  - **OSQP**：开启 `warm_starting=True`，能显著减少 ADMM 迭代次数。
  - **qpOASES**：非常适合热启动，因为它基于 Active Set 策略，相邻帧的有效约束集往往变化很小。

---

## 5. 实机调试排障

| 症状 | 可能原因 | 解决对策 |
|------|----------|----------|
| **关节高频震荡** | PD 增益太高或力矩正则化太弱 | 增加 $\|\tau\|^2$ 权重，减小任务空间 $K_p$ |
| **支撑脚打滑** | 摩擦系数高估、接触状态错误或实际接触力超限 | 保守估计 $\mu$、检查接触估计，保留摩擦可行性约束并降低任务需求 |
| **Base 姿态下沉** | CoM 权重太低，无法克服重力 | 增加 CoM 任务权重，检查动力学 $g(q)$ 补偿是否正确 |
| **QP 频繁无解** | 任务冲突严重或硬约束太多 | 检查是否有相互排斥的等式约束，增加松弛变量 |

## 6. 原文里的参数与现象（已对照原文）

| 来源 | 内容 |
|------|------|
| Kim 2019（MIT WBIC，Mini Cheetah） | WBIC 500 Hz；任务（身体姿态 / 位置 / 足端）PD 增益 $k_p=100\,s^{-2}$、$k_d=10\,s^{-1}$；QP 权重：反力项 $Q_1=1$、浮动基座松弛项 $Q_2=0.1$；关节 PD $k_p=3$ N·m/rad、$k_d=0.3$ N·m·s/rad（外展关节 $k_d=1$），40 kHz；**同一组参数用于全部步态与环境**。对浮动基座加速度加松弛，让腾空相基座可偏离指令。 |
| Del Prete 2014（TSID 原论文，仿真） | 任务 $K_p=10\,s^{-2}$、$K_d=5\,s^{-1}$；阻尼伪逆 $\lambda=0.02$（最大增益 25）；总要加最低优先级姿态任务 $\ddot q_p^*=K_p(q_p-q)-K_d\dot q$，使解与权重矩阵无关并稳定零空间。阻尼会给代价加 $\lambda\|\tau\|_W^2$ 的正则项：远离奇异时影响可忽略，接近奇异时以牺牲任务误差换取力矩有界。 |
| Kanoun 2011（HRP-2 仿真） | 任务冲突时需把正则参数 $\rho$ 调到约 0.01 量级以防不稳定；把"视线"与"伸手"放同一优先级只能得到折中姿态，降低视线优先级后手才能到达目标。 |
| Escande 2014 | 严格层级相对权重法的区别：权重 $Q$ 是"人为平衡"，层级保证低优先级不破坏高优先级；带不等式时有效集迭代次数不定，可用 warm start 与迭代上限换取确定耗时（代价少数周期次优）。 |
| Saab 2013 | "法向力非负"层不可行（强接触不稳定）时松弛变量非零、解动力学不一致，应改变接触模式而不是继续下发。 |
| OSQP 论文 | `warm_starting` 用上一次解初始化；仅向量 $q,l,u$ 变化时 KKT 分解可复用；默认精度中等（官方默认 `eps_abs=1e-3`、`max_iter=4000`），要高精度可开 polishing（需额外一次分解）。 |
| `tsid` 库 | `addMotionTask(task, weight, priorityLevel)`：优先级 0 为硬约束层（该层权重无意义），其余层带权重；默认求解器至多 2 层；`Contact6d` 以线性化摩擦锥 + 法向力界约束接触力，`setFrictionCoefficient` 可调 $\mu$。 |

> 以上数值都是**特定机器人 / 仿真下的取值**，不是通用推荐；其余章节里没有注明来源的比例与现象属于经验性说法（推测）。

---

## 关联页面
- [Whole-Body Control 概念](../concepts/whole-body-control.md)
- [TSID 方法](../concepts/tsid.md)
- [WBC 实现指南](./wbc-implementation-guide.md)
- [Friction Cone 形式化](../formalizations/friction-cone.md)

## 参考来源
- [Kim et al. 2019 — WBIC + MPC 通读归档](../../sources/papers/wbic_mpc_mini_cheetah_arxiv_1909_06586.md)（*Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control*，arXiv:1909.06586；旧版此处写作 "Quadrupedal ... Whole-Body Impulse Control"，已按原文标题订正）
- [Del Prete et al. 2014 — TSID 原论文通读归档](../../sources/papers/delprete_tsid_arxiv_1410_3863.md)
- [Kanoun et al. 2011](../../sources/papers/kanoun_task_priority_inequality_tro_2011.md)、[Escande et al. 2014](../../sources/papers/escande_hqp_ijrr_2014.md)、[Saab et al. 2013](../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md)
- [OSQP 论文通读归档](../../sources/papers/osqp_admm_qp_arxiv_1711_08013.md)
- [TSID 仓库归档](../../sources/repos/tsid.md)
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md)

## 推荐继续阅读

- Kim et al., *Highly Dynamic Quadruped Locomotion via WBIC and MPC*：<https://arxiv.org/abs/1909.06586>
- Del Prete et al., *Task Space Inverse Dynamics*：<https://arxiv.org/abs/1410.3863>
- Escande et al., *Hierarchical QP*：<https://hal.science/hal-00751924/document>
