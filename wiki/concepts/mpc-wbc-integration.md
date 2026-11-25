---
type: concept
status: complete
original: ../../sources/papers/mpc.md
sources:
  - ../../sources/papers/mpc.md
  - ../../sources/papers/whole_body_control.md
  - ../../sources/papers/wensing_opt_control_tro_arxiv_2211_11644.md
summary: "MPC-WBC integration 说明高层预测规划与低层全身控制如何分层协同完成稳定运动。"
updated: 2026-11-21
---

# MPC 与 WBC 集成：人形机器人 locomotion 的典型控制架构

> **Query op 产物** · 2026-04-13
> 问题：MPC 和 WBC 在人形机器人 locomotion 里是怎么配合工作的？
> 综合来源：MPC page + WBC page + locomotion page + optimal-control page

## 一句话定义（一句话总结）

**MPC 负责"大尺度规划"（质心往哪走、落脚点放哪），WBC 负责"全身执行"（怎么协调关节力矩来跟踪 MPC 发出的指令）**——两者分层配合，组成当前人形机器人 locomotion 最主流的控制架构。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| LIP | Linear Inverted Pendulum | 线性倒立摆，质心动力学的常用简化模型 |
| ANYmal | ANYbotics Quadruped | ANYbotics 的四足机器人研究平台 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |
| OCP | Optimal Control Problem | MPC 每步求解的有限时域最优控制问题 |

## 问题背景

人形 locomotion 面临两个层次的控制问题：

| 层次 | 核心问题 | 时间尺度 | 典型方法 |
|------|---------|---------|---------|
| **高层规划** | 接下来几步怎么走、落脚点在哪 | 100ms ~ 1s | MPC |
| **底层执行** | 每个关节发多少力矩才能跟踪规划 | 1ms | WBC (QP) |

如果只用 MPC 直接控制关节——计算量太大（30+ 自由度），实时不可行。
如果只用 WBC 不用 MPC——没有前瞻规划，只能被动响应，无法处理复杂地形和扰动。

## 为什么重要

- **跨时域预测与瞬时全阶动力学的职责分工**：简化模型 MPC（SRBD / Centroidal）可在数十毫秒周期内预见未来 $0.3\text{--}1.0\text{ s}$ 的落脚与接触反力（Kim et al. 2019 步态站立期更新约 4 次）；全阶 WBC-QP 在 $500\text{ Hz}\text{--}1\text{ kHz}$ 瞬时频率下把接触力命令和摆动腿跟踪落地为关节力矩（Wensing et al., T-RO 2024 § VI）。

## 核心机制：分层控制架构

```
感知层（IMU、编码器、视觉等）
    ↓
┌─────────────────────────────────────┐
│  高层：MPC（模型预测控制）              │
│  - 基于 centroidal dynamics / LIP 模型 │
│  - 预测未来 N 步（100ms~1s）          │
│  - 输出：下一步质心位置 / 落脚点计划    │
└─────────────────────────────────────┘
    ↓  task-space 指令
┌─────────────────────────────────────┐
│  低层：WBC（全身控制 / QP 优化）        │
│  - 接收 MPC 发来的任务空间目标          │
│  - 求解关节力矩分配 QP                  │
│  - 输出：各关节力矩指令                │
└─────────────────────────────────────┘
    ↓
电机驱动 → 机器人执行
```

## 典型实例：Walking Control

### Phase 1：MPC 生成支撑多边形和质心计划

MPC 以 **Centroidal Dynamics** 或 **LIP** 为预测模型：

```
输入：
  - 当前状态（位置、速度、姿态）
  - 参考步态（步长、步频）
  - 约束（摩擦锥、关节限位）

求解：
  - 未来 N 步的质心轨迹
  - 未来 N 步的落脚点序列（Contact Force Plan）

输出：
  - 质心参考轨迹 (CoM position / velocity)
  - 落脚点位置 + 期望接触力
```

### Phase 2：WBC 跟踪 MPC 指令

WBC 接收 MPC 的输出，将其转化为关节力矩：

```
任务层（Task Space）：
  - 跟踪质心轨迹  → 质心任务
  - 跟踪落脚点    → 足端位置任务
  - 跟踪姿态      → 姿态任务

优先级调度：
  1. 平衡任务（最高优先级）
  2. 落脚点跟踪
  3. 姿态跟踪
  4. 关节限位（硬约束）

QP 优化层：
  - 最小化任务跟踪误差
  - 满足关节力矩/速度限位
  - 满足摩擦锥约束

输出：
  - 各关节力矩指令（1kHz）
```

### 关键设计点

**1. MPC 和 WBC 的模型一致性**
- 如果 MPC 用 Centroidal Dynamics，WBC 需要把 centroidal 目标映射到关节空间
- 常用方法：先在质心空间跟踪，再在关节空间用 WBC 补偿
- **常见坑**：两层模型不一致会导致跟踪不稳定

**2. 时域分离**
- MPC 以 10-50Hz 运行（每步计算量太大，无法 kHz 级别）
- WBC 以 1kHz+ 运行（纯 QP，计算快）
- MPC 的输出作为 WBC 的"参考轨迹"输入

**3. 接触序列决定控制模式**
- 双脚支撑期：两个接触点，MPC 分配地面反力
- 单脚支撑期：一个接触点，WBC 处理力矩平衡
- **扰动恢复**：MPC 重新规划落脚点，WBC 快速响应

**4. 真实部署例子**

| 机器人 | MPC 层 | WBC 层 |
|--------|--------|--------|
| MIT Mini Cheetah | Convex MPC（约 30–40 Hz） | WBIC（500 Hz，关节 PD 40 kHz；Kim et al. 2019） |
| Unitree H1 | NMPC | WBC (Hierarchical QP) |
| ANYmal | Hierarchical NMPC | Whole-body QP |
| Atlas (Boston Dynamics) | MPC (Model Predictive Planning) | WBC |

## 为什么不用 RL 替代 MPC？

| 维度 | MPC + WBC | RL 策略 |
|------|-----------|---------|
| 实时性 | ✅ 确定性，毫秒级响应 | ⚠️ 需要推理速度足够快 |
| 扰动恢复 | ✅ MPC 重新规划，适应性很强 | ⚠️ 取决于训练分布 |
| 样本效率 | ✅ 无需训练 | ❌ 需要大量训练 |
| 约束处理 | ✅ 硬约束自然表达 | ⚠️ 需要 reward shaping |
| 泛化到新地形 | ⚠️ 需要重新规划 | ✅ 有一定泛化能力 |

**主流趋势**：MPC + WBC 作为 baseline，RL 作为提升层（学更好的策略 prior 或补偿误差）。

另有一条与「部署期 MPC+WBC」正交的路线：[MPC-RL](../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md) 在 **训练期** 用 CD-MPC 地标奖励指导 PPO，**部署期** 不再运行 MPC，由学得的低层策略直接跟踪关节目标。

## 新兴趋势：反向层级架构 (MPC-over-RL)

在 2025-2026 年，出现了一种与上述“传统分层”相反的架构，用于处理极其复杂的全身移动操作：

- **底层 (RL Policy)**：一个预训练的、强鲁棒的通用全身控制（WBC）策略。它处理高频的稳定与接触，并将原本非线性、不稳定的机器人动力学抽象为一个“命令空间”。
- **高层 (Sample-based MPC)**：在线规划器。它在底层策略的“命令空间”内进行并行轨迹采样（如 CEM）。
- **代表工作**：[sumo](../methods/sumo.md) (RAI Institute, 2026) 证明了这种架构能让机器人（Spot/G1）操纵超重、超大物体，且具备零样本泛化能力。

## 工程实践

- **WBIC 中的浮动基松弛技巧（Kim et al., arXiv:1909.06586 § IV）**：在 Mini-Cheetah 以 `3.7 m/s` 奔跑时，由于空中相与瞬时碰撞下质心状态不可直接稳态控制，WBIC 对浮动基动力学和 MPC 反力命令同时引入松弛变量 $\delta_f, \delta_{f_r}$，防止腾空相强制闭环干扰摆动腿加速度。
- **双环频率配比**：MPC 典型运行在 `30–100 Hz`（输出 $f_r^{\text{MPC}}$ 与质心参考轨迹），WBC QP 运行在 `500 Hz`（`2 ms`），关节力矩 $\tau_j$ 由全阶动力学式 $\tau_j = (A\ddot{q} + b + g - J_c^T f_r)_{\text{joints}}$ 直接解出。

## 局限与常见误区

- **上下层模型失配与约束矛盾**：上层 SRBD 忽略摆动腿动量或腕/肩惯量时，给出的 $f_r^{\text{MPC}}$ 与参考高度轨迹在下层全阶动力学 QP 中可能无法同时满足，需合理设定松弛权重比（如 $Q_1=1$ 对反力 vs $Q_2=0.1$ 对浮动基加速度）。
- **固定接触时序瓶颈**：经典凸 MPC 需预定义触地相位与摆动时长；面对剧烈碰撞打滑或大推扰恢复时，切换时序固定会限制步态自适应能力（Sleiman et al. 2021 / Wensing 2024 § VII）。

## 参考来源

- [mpc.md](../../sources/papers/mpc.md) — Convex MPC / Whole-Body NMPC / WBIC（arXiv:1909.06586、2103.00946）
- [whole_body_control.md](../../sources/papers/whole_body_control.md) — QP-WBC 与 HQP 档案
- [wensing_opt_control_tro_arxiv_2211_11644.md](../../sources/papers/wensing_opt_control_tro_arxiv_2211_11644.md) — Wensing et al., T-RO 2024 § V–VI 分层闭环分析
- [未知载荷下的自适应全身 MPC](../../sources/papers/adaptive_whole_body_mpc_2610_12972.md) — 质心 MPC、模糊平衡补偿与姿态规划
- Di Carlo et al., *Dynamic Locomotion in the MIT Cheetah 3* (2018) — MPC+WBC 两层架构代表作
- [MIT Mini Cheetah 平台页](../entities/mit-mini-cheetah.md) — 官方 `Cheetah-Software` 与开源边界
- Bellicoso et al., *Advances in Real-world Applications for Legged Systems* (2018) — ANYmal MPC-WBC 实践
- Kim et al., *Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control* (2019) — WBC 全身冲量控制；实体页 [paper-wbic-mpc-mini-cheetah](../entities/paper-wbic-mpc-mini-cheetah.md)
- [MIT Mini Cheetah](../entities/mit-mini-cheetah.md) — 平台与论文生态

## 推荐继续阅读

- [SRBD + 凸 MPC + WBC](./srbd-convex-mpc-wbc.md) — 从简单模板模型到瞬时力矩执行的完整公式拆解
- [Quadratic Programming](../formalizations/quadratic-programming.md) — 下层 WBC QP 的凸优化标准型
- [Contact Estimation](./contact-estimation.md) — 双环闭环依赖的接触与基座速度感知前提

## 关联页面

- [未知载荷下的自适应全身 MPC](../entities/paper-adaptive-whole-body-mpc-payload.md) — 搬运中的载荷失配补偿

- [具身三层控制架构](./embodied-three-layer-control-architecture.md) — MPC+WBC 所在「小脑层」与大脑/反射边界
- [Model Predictive Control (MPC)](../methods/model-predictive-control.md) — MPC 层
- [Whole-Body Control (WBC)](./whole-body-control.md) — WBC 层
- [Centroidal Dynamics](./centroidal-dynamics.md) — MPC 常用的人形简化动力学模型
- [MIT Mini Cheetah](../entities/mit-mini-cheetah.md) — Convex MPC 工程祖师爷平台
- [Locomotion](../tasks/locomotion.md) — locomotion 任务层
- [Optimal Control (OCP)](./optimal-control.md) — 理论层，MPC 是 OCP 的在线求解
- [Query：控制架构选型对比](../queries/control-architecture-comparison.md)
- [MPC-RL](../entities/paper-mpc-rl-humanoid-locomotion-manipulation.md) — 训练期 CD-MPC 指导、部署期纯 RL
