# 路线（纵深）：如果目标是安全控制（CLF / CBF / Safe RL / 真机安全部署）

**摘要**：面向"在满足安全约束的前提下控制机器人"的纵深路线，从 Lyapunov 稳定性到 CBF-QP、再到 Safe RL，最后落到真机上的分层安全部署（急停 / 安全状态机 / 跌倒减损 / 上机流程），按 Stage 0–4 串通；本路线是 [运动控制主路线](motion-control.md) 的一条分支。

## 路线一览

```mermaid
flowchart LR
  S0["<b>Stage 0</b><br/>数学基础<br/><em>Lyapunov 稳定性</em>"]
  S1["<b>Stage 1</b><br/>CLF / CBF 基础<br/><em>CBF-QP 实现</em>"]
  S2["<b>Stage 2</b><br/>嵌入 WBC / MPC<br/><em>Safety Filter</em>"]
  S3["<b>Stage 3</b><br/>Safe RL<br/><em>CMDP / Lagrangian</em>"]
  S4["<b>Stage 4</b><br/>真机安全部署<br/><em>分层安全壳 / 安全 FSM</em>"]

  S0 --> S1 --> S2 --> S3 --> S4

  classDef stage fill:#142a3a,stroke:#e67e22,stroke-width:2px,color:#fff
  class S0,S1,S2,S3,S4 stage
```

## 这条路径怎么用

- 目标读者是有控制理论基础、想加入安全保证的工程师或研究者
- 需要有基础线性代数和微分方程直觉；RL 基础有助于后期阶段
- 每个阶段有前置知识、核心问题、推荐做什么、学完输出什么

**和主路线的关系：**
- 安全控制可以作为 WBC / MPC 的安全约束层，也可以作为 RL 的奖励整形层
- 如果你做主路线 L4 时希望加上安全保证，看本路线
- 如果你做 RL 时希望策略可证明地不越界，看本路线 Stage 3
- 如果你要把策略搬上真机、担心摔机 / 失控 / 故障伤人，看本路线 Stage 4（也可先读，作为上机前的安全底线）

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
- [Lyapunov 稳定性形式化](../wiki/formalizations/lyapunov.md)
- Khalil, *Nonlinear Systems* — Chapter 4（稳定性定义）

### 推荐做什么
- 选一个一阶或二阶简单系统，手写一个 Lyapunov 函数并验证导数负半定
- 对照 Khalil *Nonlinear Systems* Chapter 4，把稳定性定义和自己的证明对上

### 学完输出什么
- 能手工验证一个简单系统的 Lyapunov 稳定性
- 理解正定函数和负半定导数的含义

---

## Stage 1 CLF / CBF 基础

### 核心问题
- CLF 和 CBF 分别解决什么问题（收敛 vs. 安全边界）
- 两者如何联合放入 QP 优化

### 推荐读什么
- [Control Lyapunov Function](../wiki/formalizations/control-lyapunov-function.md)
- [Control Barrier Function](../wiki/concepts/control-barrier-function.md)
- [CLF vs CBF 对比](../wiki/comparisons/clf-vs-cbf.md)
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
- [Safety Filter](../wiki/concepts/safety-filter.md)
- [Whole-Body Control](../wiki/concepts/whole-body-control.md)
- [Query：CLF+CBF 在 WBC/MPC 中联合使用](../wiki/queries/clf-cbf-in-wbc.md)
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
- [Safe RL](../wiki/methods/safe-rl.md)
- [CMDP 形式化](../wiki/formalizations/cmdp.md)
- [安全的真机 RL 微调](../wiki/concepts/safe-real-world-rl-fine-tuning.md)
- Garcia & Fernandez, *A Comprehensive Survey on Safe RL* (2015)
- [Reinforcement Learning](../wiki/methods/reinforcement-learning.md)

### 推荐做什么
- 用 Safety Gym（OpenAI）或 safe-control-gym 跑一个 constrained RL 实验

### 学完输出什么
- 能解释 CMDP 的形式化定义
- 能对比 model-based 安全保证 vs. model-free 安全奖励塑形的优劣

---

## Stage 4 真机安全部署：分层安全壳

### 前置知识
- Stage 2 的 safety filter 与 Stage 3 的 Safe RL（知道算法层能管什么）
- 有一台可上机的策略（RL / WBC 均可），读过 [Sim2Real](../wiki/concepts/sim2real.md) 基本概念

### 核心问题
- 为什么 CBF / Safe RL 的保证不够：模型误差、驱动器报错、总线超时、估计发散都在算法假设之外——**安全必须独立于策略**，分布在算法 / 控制板 / 驱动 / 机械多层
- 硬故障时切到什么安全态：确定性 FSM（看门狗、总线超时 → 阻尼 / 无力矩 / 冻结），以及「无力矩 = 摔倒」的 **fail-passive gap**
- 急停怎样才不摔：先判断当前状态是否还「可停」，再选停止策略或阻尼防摔
- 跌倒不可避免时如何减损；上机前如何逐级验证（SIL → HIL → 吊架 → 渐进测试）

### 推荐读什么
- [Sim2Real 闭环误差分层工程](../wiki/queries/sim2real-closed-loop-engineering.md) — §6「分层安全：独立于策略」
- [Sim2Real 工程 Checklist](../wiki/queries/sim2real-checklist.md) — 阶段 4「真机初次部署」：吊绳 / 急停 / 力矩上限 / 站立先行
- [机器人安全状态机](../wiki/concepts/robot-safety-state-machine.md) 与 [wbc_fsm](../wiki/entities/wbc-fsm.md)（Passive / Loco / WBC 模式切换实例）
- [Fail-Passive Gap](../wiki/entities/paper-fail-passive-gap.md) — ISO 13849 断电即安全 vs 双足主动平衡；外部链可评 PL，机侧反应链评不了
- [Safe-Stop](../wiki/entities/paper-safe-stop-humanoid.md) — 急停建模为 reach-avoid，双 stoppability 估计一致才停，否则 damping fallback
- [ResSafe](../wiki/entities/paper-ressafe.md) / [SafeWBC](../wiki/entities/paper-motion-cerebellum-safewbc.md) — 学习型 / CBF 型安全层接在策略或 WBC 输出之后
- [SafeFall](../wiki/entities/paper-hrl-stack-41-safefall.md) 与 [Balance Recovery](../wiki/tasks/balance-recovery.md) — 摔倒预测 + 减损策略，平时 dormant
- [Hardware-in-the-Loop](../wiki/concepts/hardware-in-the-loop.md) / [人形测试流程 L0–L5](../wiki/concepts/humanoid-testing-workflow.md) — 上机前的逐级验证
- [SLowRL](../wiki/entities/paper-slowrl-safe-lora-locomotion-sim2real.md) — 真机微调期用 Recovery 安全滤波限制探索（衔接 Stage 3）
- [模型版本管理与 OTA](../wiki/concepts/model-versioning-ota.md) — 仅 Passive / Safe 态允许换权重
- [RL 策略真机调试 Playbook](../wiki/queries/robot-policy-debug-playbook.md)

### 推荐做什么
- 在仿真里给自己的控制栈加一个 `Init → Passive → Active → Fault → Safe` 状态机，注入总线超时、deadline miss、IMU 失效，检查每种故障在一个控制周期内进入安全态、且不会自动弹回 Active
- 按 [Sim2Real Checklist](../wiki/queries/sim2real-checklist.md) 阶段 4 的顺序做一次上机：吊架保护 + 硬件急停 + 力矩上限 → 站立 ≥ 30 s → 原地踏步 → 极慢速直走
- 对一台人形，写清楚「Safe 态到底是什么」：无力矩 / 阻尼站立 / 跪倒脚本，并说明它在哪些状态下会导致摔倒

### 学完输出什么
- 一张本机的分层安全表：每层（机械 / 驱动 / 控制板 / 算法）的故障源 → 检测 → 安全动作
- 能解释 safety filter（约束正常控制）与安全 FSM（处理硬故障）的分工，以及 fail-passive gap 为什么是人形认证的缺口
- 一份可复用的上机前 checklist

---

## 快速入口汇总

| 阶段 | 核心问题 | 知识页入口 |
|------|---------|-----------|
| Stage 0 | Lyapunov 稳定性 | [Lyapunov 稳定性](../wiki/formalizations/lyapunov.md) |
| Stage 1 | CLF / CBF 与 CBF-QP | [Control Barrier Function](../wiki/concepts/control-barrier-function.md) |
| Stage 2 | 嵌入 WBC / MPC | [Safety Filter](../wiki/concepts/safety-filter.md) |
| Stage 3 | Safe RL / CMDP | [Safe RL](../wiki/methods/safe-rl.md) |
| Stage 4 | 真机分层安全 / 安全 FSM / 急停 | [机器人安全状态机](../wiki/concepts/robot-safety-state-machine.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制算法工程师成长路线](motion-control.md)
- 相关纵深路线（按主题邻近，完整目录见 [路线总览](motion-control.md#depth-optional-index)）：
  - [传统模型控制](depth-classical-control.md) — Safety Filter / CBF 约束嵌进 WBC/MPC
  - [Sim2Real](depth-sim2real.md) — Stage 4 真机安全部署的迁移侧（辨识、DR、上机流程）
  - [RL 运动控制](depth-rl-locomotion.md) — Stage 3 Safe RL 的策略训练侧
  - [接触操作](depth-contact-manipulation.md) — 接触力边界是另一类安全约束
- 关联知识页：
  - [Lyapunov 稳定性](../wiki/formalizations/lyapunov.md)
  - [Control Lyapunov Function](../wiki/formalizations/control-lyapunov-function.md)
  - [Control Barrier Function](../wiki/concepts/control-barrier-function.md)
  - [CLF vs CBF 对比](../wiki/comparisons/clf-vs-cbf.md)
  - [Safety Filter](../wiki/concepts/safety-filter.md)
  - [Safe RL](../wiki/methods/safe-rl.md) 与 [CMDP 形式化](../wiki/formalizations/cmdp.md)
  - [Whole-Body Control](../wiki/concepts/whole-body-control.md)
  - [Query：CLF+CBF 在 WBC/MPC 中联合使用](../wiki/queries/clf-cbf-in-wbc.md)
  - [机器人安全状态机](../wiki/concepts/robot-safety-state-machine.md) 与 [Sim2Real 闭环误差分层工程](../wiki/queries/sim2real-closed-loop-engineering.md)
  - [真机安全微调知识链](../wiki/overview/hub-safe-fine-tuning.md)

## 参考来源

- [Lyapunov 稳定性](../wiki/formalizations/lyapunov.md)
- [Control Lyapunov Function](../wiki/formalizations/control-lyapunov-function.md)
- [Control Barrier Function](../wiki/concepts/control-barrier-function.md)
- [CLF vs CBF 对比](../wiki/comparisons/clf-vs-cbf.md)
- Ames et al., *Control Barrier Function based Quadratic Programs* (2017)
- Khalil, *Nonlinear Systems*
- [Sim2Real 闭环误差分层工程](../wiki/queries/sim2real-closed-loop-engineering.md)、[Sim2Real 工程 Checklist](../wiki/queries/sim2real-checklist.md)
- [机器人安全状态机](../wiki/concepts/robot-safety-state-machine.md)、[Fail-Passive Gap（arXiv:2608.02809）](../wiki/entities/paper-fail-passive-gap.md)、[Safe-Stop（arXiv:2609.02358）](../wiki/entities/paper-safe-stop-humanoid.md)
