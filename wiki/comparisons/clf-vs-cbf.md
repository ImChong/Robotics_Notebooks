---
type: comparison
tags: [control, safety, clf, cbf, qp, stability, optimization, humanoid]
status: stub
summary: "CLF 负责稳定性（驱动系统到达目标），CBF 负责安全性（阻止系统进入危险区域），两者互补，联合构成 CLF-CBF-QP 实时安全控制框架。"
sources:
  - ../../sources/papers/optimal_control_theory.md
related:
  - ../concepts/control-barrier-function.md
  - ../formalizations/control-lyapunov-function.md
  - ../concepts/whole-body-control.md
  - ../formalizations/lqr.md
---

# CLF vs CBF：稳定性与安全性的对偶工具

## 比较对象

| 工具 | 全称 | 核心角色 |
|------|------|---------|
| **CLF** | Control Lyapunov Function（控制李雅普诺夫函数） | 驱动系统到达目标状态（**稳定性**） |
| **CBF** | Control Barrier Function（控制屏障函数） | 阻止系统进入危险区域（**安全性**） |

两者均将各自的条件转化为对控制输入的线性约束，嵌入 QP 实时求解——但解决的是完全不同的问题。

## 一句话定义对比

- **CLF**：找一个"能量函数" $V(x)$，要求控制输入使其持续下降，直到系统到达目标
- **CBF**：找一个"边界函数" $h(x)$，要求控制输入使其保持非负，从而系统永不越过安全边界

## 核心差异

### 数学条件对比

| 维度 | CLF | CBF |
|------|-----|-----|
| 函数类型 | 正定函数 $V(x) > 0$（$V(0) = 0$） | 符号函数 $h(x)$（$h \geq 0$ 为安全） |
| 约束方向 | $\dot{V}(x, u) \leq -\alpha(V(x))$ | $\dot{h}(x, u) \geq -\gamma h(x)$ |
| 约束语义 | $V$ 必须衰减（收敛到 0） | $h$ 不能从正变负（保持安全集） |
| 控制目标 | 驱动状态到达目标 | 阻止状态离开安全区 |
| 强制执行方式 | 软约束（通常带松弛 $\delta$） | 硬约束（不允许违反） |
| 失效时的处理 | 允许适当放松（牺牲收敛速度） | 不允许放松（安全性绝对优先） |

### 系统行为对比

```
CLF 视角（稳定性）：
  状态 x(t) ---CLF驱动---> 目标 x* = 0
  V(x) 单调递减，指数收敛

CBF 视角（安全性）：
  状态 x(t) ---CBF保持---> 始终在 {x | h(x)≥0} 内
  h(x) 保持非负，不越过安全边界
```

### 约束冲突时的行为

当 CLF 条件（要求快速靠近目标）和 CBF 条件（要求远离危险边界）发生冲突时：

- **CBF 优先**：系统会减慢靠近目标的速度（CLF 松弛），而不是穿越安全边界
- **结果**：系统可能"绕路"抵达目标，或在安全边界附近暂时停留

这种优先级设计反映了一个工程哲学：**安全是硬约束，性能是软目标**。

## 各自适用场景

### 只用 CLF 的场景

**适用条件**：
- 任务空间中没有障碍物或安全约束
- 重点是保证跟踪精度和收敛速度
- 系统工作在安全范围内远离边界

**典型应用**：
- 机械臂末端轨迹跟踪（无碰撞担忧）
- 闭环步态稳定（期望步态在吸引子附近）
- 基于 Energy Shaping 的平衡控制

### 只用 CBF 的场景

**适用条件**：
- 已有成熟的名义控制律，只需要在其上叠加安全过滤
- 安全约束是主要设计目标，收敛性由名义控制律保证
- 需要快速验证一个"黑盒"控制器的安全性

**典型应用**：
- 在 RL 策略上叠加碰撞避免约束（Safe RL）
- 在 WBC 力分配中强制接触力锥约束
- 关节限位保护（保护硬件不受损坏）

### CLF-CBF-QP 联合使用的场景

**适用条件**：
- 需要同时保证到达目标（稳定性）和避免危险（安全性）
- 约束的冲突需要被系统地解决（QP 自动处理优先级）
- 要求可证明的稳定性与安全性，而非仅靠启发式调参

**典型应用**：
- 人形机器人在有障碍物的环境中导航到目标位置
- 双足行走时的足端安全区约束 + 步态稳定
- 机械臂操作中的工作空间限制 + 末端轨迹跟踪
- 与 WBC 集成：CLF 驱动任务完成，CBF 保证接触力和关节安全

## CLF-CBF-QP 联合框架

$$\min_{u, \delta} \quad \frac{1}{2} \|u - u_{nom}\|^2 + p \delta^2$$

$$\text{s.t.} \quad \dot{V}(x, u) \leq -c \cdot V(x) + \delta \quad \text{（CLF 软约束）}$$
$$\qquad\quad \dot{h}(x, u) \geq -\gamma h(x) \quad \text{（CBF 硬约束）}$$
$$\qquad\quad u \in \mathcal{U}, \quad \delta \geq 0$$

**变量说明**：
- $u_{nom}$：来自高层控制器（MPC / WBC / RL）的名义控制输入
- $\delta$：CLF 松弛变量，$p$ 是惩罚权重
- $c, \gamma$：分别控制收敛速率和安全衰减速率的调参数
- $\mathcal{U}$：物理约束集（关节限位、力矩上限等）

## 选型决策树

```
你的主要控制问题是什么？
│
├── 主要是"让系统到达目标"？
│   ├── 有安全约束需要强制执行？ → 用 CLF-CBF-QP
│   └── 无安全约束？ → 用 CLF-QP（或直接 LQR/MPC）
│
├── 主要是"保证安全"，已有名义控制律？
│   ├── 需要理论保证？ → 用 CBF-QP（安全过滤器）
│   └── 只需工程近似？ → 用势能场 / 惩罚项
│
└── 既要到达目标，又要保证安全？ → 用 CLF-CBF-QP
```

## 常见误判

### 误判1：CLF 能保证安全

CLF 只保证系统**收敛**，不阻止系统经过危险区域。在靠近目标的过程中，系统完全可能穿越障碍物（如果没有 CBF 约束）。

### 误判2：CBF 能保证到达目标

CBF 保证系统**留在安全集内**，但安全集内的行为完全取决于名义控制律。如果名义控制律不稳定，CBF 无法让系统到达目标——反而可能在安全边界附近徘徊。

### 误判3：CLF-CBF-QP 总是有可行解

当 CBF 约束非常紧（系统靠近多个安全边界的交叉点），且 CLF 约束与之冲突时，QP 可能无可行解。需要：
- 软化 CLF 约束（引入松弛）
- 调整 $\gamma$ 参数使约束不那么保守
- 设计 CBF 使安全集的交集非空

### 误判4：两者都适合离散时间系统

连续时间 CLF/CBF 理论不能直接应用于离散时间系统。离散时间需要专门的 DCLF/DCBF，约束形式是差分而非微分。

## 结论

| 问题类型 | 推荐工具 |
|---------|---------|
| 只需稳定性，无安全约束 | CLF-QP 或 LQR |
| 只需安全过滤，名义控制器已有 | CBF-QP |
| 同时需要稳定性 + 安全性 | CLF-CBF-QP |
| 系统已用 MPC，只需安全过滤 | 在 MPC 约束中加入 CBF 约束 |
| RL 策略的安全保证 | CBF-QP 作为后处理层 |

在人形机器人实际工程中，最常见的模式是：**WBC/MPC 提供名义控制律，CBF-QP 作为安全过滤器**，而 CLF 则被"内化"到 WBC 的任务空间跟踪误差代价函数中（隐式 CLF）。

## 参考来源

- Ames et al., *Control Barrier Function Based Quadratic Programs for Safety Critical Systems* (IEEE TAC, 2017) — CLF-CBF-QP 奠基论文
- Sontag, *A universal construction of Artstein's theorem on nonlinear stabilization* (1989) — CLF 理论基础
- [sources/papers/optimal_control_theory.md](../../sources/papers/optimal_control_theory.md) — 最优控制与 Lyapunov 方法背景

## 关联页面

- [Control Barrier Function（CBF）](../concepts/control-barrier-function.md) — CBF 详细介绍
- [Control Lyapunov Function（CLF）](../formalizations/control-lyapunov-function.md) — CLF 详细介绍
- [Whole-Body Control](../concepts/whole-body-control.md) — CLF-CBF-QP 在全身控制中的应用
- [LQR](../formalizations/lqr.md) — CLF 的线性系统特例

## 推荐继续阅读

- Ames et al., [*Control Barrier Functions: Theory and Applications*](https://arxiv.org/abs/1903.11199)
- [CLF 详解](../formalizations/control-lyapunov-function.md)
- [CBF 详解](../concepts/control-barrier-function.md)

## 一句话记忆

> CLF 是油门（推向目标），CBF 是刹车（不越安全边界）；CLF-CBF-QP 是同时踩油门和刹车——让系统在最快到达目标的同时，永远不越过安全红线。
