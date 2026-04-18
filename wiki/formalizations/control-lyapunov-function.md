---
type: formalization
tags: [control, lyapunov, clf, stability, qp, optimization, safety]
status: stub
summary: "Control Lyapunov Function（控制李雅普诺夫函数）通过构造满足 V̇(x)≤-αV(x) 的标量函数，为控制系统提供可证明的渐近稳定性保证，是 CLF-CBF-QP 框架的稳定性核心。"
sources:
  - ../../sources/papers/optimal_control_theory.md
related:
  - ../concepts/control-barrier-function.md
  - ../comparisons/clf-vs-cbf.md
  - ./lyapunov.md
  - ./lqr.md
  - ../concepts/whole-body-control.md
---

# Control Lyapunov Function（控制李雅普诺夫函数）

**Control Lyapunov Function（CLF）**：一种用于设计使系统渐近稳定的控制律的数学工具。通过找到一个正定标量函数 $V(x)$，使得在某个控制输入下 $\dot{V}(x)$ 严格为负，从而保证系统状态最终收敛到目标状态。

## 一句话定义

> CLF 回答的是：**"存在哪个控制输入，能让系统'能量'（Lyapunov 函数）单调递减，直到状态到达目标？"** 它将稳定性设计问题转化为寻找一个满足特定衰减条件的控制律，并可嵌入 QP 实时在线求解。

## 为什么重要

Lyapunov 稳定性分析是控制理论的核心工具，但传统 Lyapunov 方法主要用于**分析**给定控制律的稳定性，而非**综合**使系统稳定的控制律。CLF 弥补了这一鸿沟：

1. **从分析到综合**：CLF 不仅分析稳定性，还直接给出使系统稳定的控制条件（CLF 条件），进而通过 QP 求解满足该条件的控制输入
2. **与 CBF 的互补性**：CLF 负责驱动系统到达目标（稳定性/性能），CBF 负责保持系统在安全集内（安全性）——两者组合构成 CLF-CBF-QP 框架
3. **实时控制中的应用**：CLF 条件对控制输入是线性的，可以直接作为 QP 约束，适合嵌入高频控制回路
4. **理论严谨性**：CLF 提供渐近稳定性的严格数学保证，比单纯的代价函数优化更可靠

## 核心定义

### Lyapunov 稳定性回顾

对于自治系统 $\dot{x} = f(x)$，若存在正定函数 $V(x)$（$V(0) = 0$，$V(x) > 0 \; \forall x \neq 0$）满足：

$$\dot{V}(x) = \frac{\partial V}{\partial x} f(x) < 0 \quad \forall x \neq 0$$

则原点全局渐近稳定（GAS）。

### Control Lyapunov Function 定义

对于仿射控制系统 $\dot{x} = f(x) + g(x)u$，若存在正定函数 $V(x)$ 和扩展类 $\mathcal{K}$ 函数 $\alpha$，使得对所有 $x \neq 0$，**存在**控制输入 $u$，满足：

$$\dot{V}(x, u) = \frac{\partial V}{\partial x}[f(x) + g(x)u] \leq -\alpha(V(x))$$

则称 $V$ 为系统的 **Control Lyapunov Function**。

常见选择 $\alpha(V) = c \cdot V$（线性衰减），则条件变为：

$$\dot{V}(x) \leq -c \cdot V(x), \quad c > 0$$

这保证 $V(x(t)) \leq V(x(0)) e^{-ct}$，即指数收敛。

### 存在性与充分条件

CLF 的关键性质：**存在 CLF 当且仅当系统可以被渐近稳定化**（在连续时间、非线性系统中，这由 Artstein 定理保证）。因此，CLF 不仅是稳定性工具，也是系统是否能被稳定的等价刻画。

## CLF 条件作为 QP 约束

### 标准 CLF-QP

将 CLF 条件写为对控制输入 $u$ 的线性约束，构建 QP：

$$\min_{u} \quad \frac{1}{2} \|u - u_{nom}\|^2 + \frac{p}{2} \delta^2$$

$$\text{s.t.} \quad \underbrace{\frac{\partial V}{\partial x} g(x)}_{\text{行向量}} \cdot u \leq -\frac{\partial V}{\partial x} f(x) - c \cdot V(x) + \delta$$

其中 $u_{nom}$ 是名义控制律（如 PD 控制），$\delta \geq 0$ 是松弛变量，$p$ 是惩罚权重。

注意：CLF 约束要求 $\dot{V}$ **足够小**（即约束是 $\leq$ 方向），而 CBF 约束要求 $\dot{h}$ **足够大**（$\geq$ 方向）——方向相反。

### CLF 约束的线性性

关键观察：$\dot{V} = \frac{\partial V}{\partial x}[f(x) + g(x)u]$ 对 $u$ 是仿射的，所以 CLF 约束对 $u$ 是线性的，QP 可高效求解。

## CLF-CBF-QP：联合框架

在实际机器人控制中，往往需要同时保证稳定性（到达目标）和安全性（避免危险区域）。CLF-CBF-QP 将两个约束纳入统一的 QP：

$$\min_{u, \delta} \quad \frac{1}{2} \|u - u_{nom}\|^2 + p \delta^2$$

$$\text{s.t.} \quad \underbrace{\dot{V}(x, u) \leq -c \cdot V(x) + \delta}_{\text{CLF 约束（软）}}$$

$$\quad\quad\quad \underbrace{\dot{h}(x, u) \geq -\gamma h(x)}_{\text{CBF 约束（硬）}}$$

$$\quad\quad\quad u \in \mathcal{U}, \quad \delta \geq 0$$

**设计决策说明：**
- CLF 约束通常做**软约束**（引入松弛 $\delta$）：允许在安全约束紧张时适当牺牲收敛速度
- CBF 约束通常做**硬约束**：安全性必须绝对保证，不允许松弛
- 当 CLF 与 CBF 约束发生冲突时，CBF 优先（系统会减速靠近目标，而不是穿越安全边界）

### 可行性分析

CLF-CBF-QP 的可行性取决于：
1. CBF 约束集 $\mathcal{C}$ 内存在满足 CBF 条件的控制输入（CBF 本身的条件）
2. CLF 约束与 CBF 约束不完全冲突（在某些边界情况下可能发生）

当 CLF 约束被软化（引入松弛）时，只要 CBF 约束可行，整个 QP 一定有可行解。

## CLF 在机器人控制中的应用

### 1. 平衡控制中的 CLF 设计

对人形机器人的质心（CoM）高度控制，定义：

$$V(x) = \frac{1}{2}(h_{CoM} - h_{ref})^2 + \frac{1}{2}\dot{h}_{CoM}^2$$

这是一个关于 CoM 位置偏差和速度的二次 Lyapunov 函数，CLF 条件驱动 CoM 回到参考高度。

### 2. 步态稳定中的 CLF

对于双足行走的步态稳定，可以定义 CLF 基于轨道能量（orbital energy）或步态误差，使步态在扰动后能收敛回稳定周期轨道。

### 3. 与 WBC 的集成

在 Whole-Body Control 框架中，CLF 可以作为任务空间轨迹跟踪的稳定性保证：

- 上层：轨迹生成（给出参考状态 $x_{ref}(t)$）
- 中层：CLF-CBF-QP（求解满足稳定性和安全性的关节加速度）
- 下层：逆动力学（将加速度转化为关节力矩）

## CLF 与 LQR 的关系

LQR 的代价函数 $V(x) = x^T P x$（其中 $P$ 是 Riccati 方程的解）本身就是一个 CLF：

$$\dot{V}(x) = x^T (A^T P + PA + Q - PBR^{-1}B^T P) x = -x^T Q x \leq -\lambda_{min}(Q) \|x\|^2$$

因此 LQR 是 CLF 理论在线性系统上的特例，LQR 的最优代价函数自然满足 CLF 条件。

## 常见误区

1. **CLF 不保证安全性**：CLF 只驱动系统到达目标，不阻止系统经过危险区域。需要 CBF 共同工作才能同时保证稳定性和安全性。

2. **CLF 的存在不等于好找**：虽然 CLF 存在性与可稳定性等价，但对于复杂非线性系统，构造一个有效的 CLF 仍然是开放问题。常用方法：Energy-based CLF、SOS（平方和）优化、学习 CLF。

3. **指数衰减 vs 渐近衰减**：$\dot{V} \leq -c V$ 保证指数收敛（更强），而 $\dot{V} < 0$ 只保证渐近收敛。实际中通常选指数衰减以获得更快的响应和更好的扰动抑制。

4. **离散时间系统**：连续时间 CLF 的理论在离散时间需要修改（条件变为 $V(x_{t+1}) - V(x_t) \leq -c V(x_t)$），不能直接套用。

## 参考来源

- Sontag, *A universal construction of Artstein's theorem on nonlinear stabilization* (1989) — CLF 奠基理论（Artstein-Sontag 公式）
- Ames et al., *Control Barrier Function Based Quadratic Programs for Safety Critical Systems* (IEEE TAC, 2017) — CLF-CBF-QP 联合框架
- [sources/papers/optimal_control_theory.md](../../sources/papers/optimal_control_theory.md) — 最优控制与 Lyapunov 方法背景

## 关联页面

- [Control Barrier Function（CBF）](../concepts/control-barrier-function.md) — 安全性对偶工具，CLF 的搭档
- [CLF vs CBF](../comparisons/clf-vs-cbf.md) — 两者详细对比与联合使用场景
- [Lyapunov 稳定性](./lyapunov.md) — CLF 的理论基础
- [LQR](./lqr.md) — LQR 最优代价函数是线性系统上的 CLF 特例
- [Whole-Body Control](../concepts/whole-body-control.md) — CLF-CBF-QP 在 WBC 中的实际应用

## 推荐继续阅读

- Ames et al., [*Control Barrier Functions: Theory and Applications*](https://arxiv.org/abs/1903.11199)
- Galloway et al., *Torque Saturation in Bipedal Robotic Walking through Control Lyapunov Function Based Quadratic Programs*
- [Control Barrier Function（CBF）](../concepts/control-barrier-function.md)

## 一句话记忆

> CLF 把"系统稳定性"转化为"Lyapunov 函数需以速率 $c$ 指数衰减"的控制约束，与 CBF 安全约束一起嵌入 QP，构成可证明安全且稳定的实时控制框架。
