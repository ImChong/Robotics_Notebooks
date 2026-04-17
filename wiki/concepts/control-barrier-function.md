---
type: concept
tags: [control, safety, cbf, qp, optimization, humanoid, whole-body-control]
status: stub
summary: "Control Barrier Function（控制屏障函数）通过维持 h(x)≥0 来为机器人系统提供可证明的安全保证，常与 QP 结合嵌入实时控制回路。"
sources:
  - ../../sources/papers/optimal_control_theory.md
related:
  - ./whole-body-control.md
  - ./contact-dynamics.md
  - ../formalizations/lqr.md
  - ../formalizations/control-lyapunov-function.md
  - ../comparisons/clf-vs-cbf.md
---

# Control Barrier Function（控制屏障函数）

**控制屏障函数（Control Barrier Function，CBF）**：一种将系统安全约束转化为可在控制层实时强制执行的数学工具。通过定义一个标量函数 $h(x)$，使得 $h(x) \geq 0$ 对应"安全集"，并构造使系统始终保持在安全集内的控制条件。

## 一句话定义

CBF 回答的是：**"给定一个当前控制律，如何在最小修改的前提下，保证系统状态永远不进入危险区域？"** 其核心是把安全性转化为对控制输入的线性约束，进而通过 QP（二次规划）实时求解。

## 为什么重要

在机器人控制中，稳定性与安全性是两个不同的问题：

- **稳定性**：系统能否收敛到目标状态（CLF 负责）
- **安全性**：系统是否避免了某些危险配置（CBF 负责）

传统方法把安全约束编码为惩罚项（软约束），无法提供可证明保证。CBF 的优势在于：

1. **可证明的安全性**：在 CBF 条件满足时，安全集的不变性可以用 Lyapunov 理论严格证明
2. **计算高效**：安全约束被线性化为 QP 约束，求解速度足以满足实时控制（1kHz+）
3. **可与任意上层控制器结合**：作为"安全过滤器"叠加在 WBC、MPC 或学习策略之上
4. **与物理约束天然对接**：接触力锥约束、关节限位、碰撞避免都可建模为 CBF

## 核心定义

### 安全集与屏障函数

给定一个连续时间仿射控制系统：

$$\dot{x} = f(x) + g(x)u$$

定义**安全集** $\mathcal{C}$ 为：

$$\mathcal{C} = \{x \in \mathbb{R}^n \mid h(x) \geq 0\}$$

其中 $h: \mathbb{R}^n \to \mathbb{R}$ 是一个连续可微函数。$h(x) \geq 0$ 对应安全区域，$h(x) < 0$ 对应危险区域。

### CBF 条件（一阶 CBF）

$h$ 是一个**控制屏障函数**，当且仅当存在一个扩展类 $\mathcal{K}$ 函数 $\alpha$，使得对所有 $x \in \mathcal{C}$ 存在控制输入 $u$，满足：

$$\dot{h}(x, u) = \frac{\partial h}{\partial x} [f(x) + g(x)u] \geq -\alpha(h(x))$$

直觉解释：$h$ 的时间导数下界由 $-\alpha(h(x))$ 控制。当 $h(x)$ 接近 0（接近边界）时，$-\alpha(h(x))$ 也接近 0，强迫系统减慢"撞墙"的速度。一旦 $h(x) > 0$（深处），$-\alpha(h(x))$ 可以较大，允许系统自由运动。

常见选择：$\alpha(h) = \gamma h$（线性类 K 函数），则条件变为：

$$\dot{h}(x, u) \geq -\gamma h(x)$$

### 集合不变性证明

若上述 CBF 条件在所有时刻成立，则 $\mathcal{C}$ 是前向不变集（forward invariant set）：若初始状态 $x(0) \in \mathcal{C}$，则对所有 $t \geq 0$ 有 $x(t) \in \mathcal{C}$。这由比较引理（Comparison Lemma）直接给出。

## CBF-QP：实时安全过滤器

### 基本形式

给定一个"名义控制律" $u_{nom}$（来自 WBC、MPC 或学习策略），CBF-QP 对其做最小修改，使安全约束满足：

$$\min_{u} \quad \frac{1}{2} \|u - u_{nom}\|^2$$
$$\text{s.t.} \quad \frac{\partial h}{\partial x} g(x) u \geq -\frac{\partial h}{\partial x} f(x) - \gamma h(x)$$

这是一个含线性约束的 QP，可在微秒级内求解。CBF 约束对控制输入 $u$ 是线性的（因为系统对 $u$ 是仿射的），这是 CBF 方法能实时化的关键。

### 多约束扩展

实际系统往往有多个安全约束（关节限位、碰撞避免、接触力锥等），每个约束对应一个 $h_i(x)$：

$$\min_{u} \quad \frac{1}{2} \|u - u_{nom}\|^2 + \lambda \xi$$
$$\text{s.t.} \quad \dot{h}_i(x, u) \geq -\gamma_i h_i(x) - \xi, \quad \forall i$$

引入松弛变量 $\xi \geq 0$ 处理约束冲突（feasibility relaxation），代价函数中用权重 $\lambda$ 惩罚松弛量。

## CBF 在人形机器人中的应用

### 接触力安全约束

脚与地面的接触力必须在摩擦锥内，否则脚会打滑：

$$h_{friction}(f) = \mu f_z - \sqrt{f_x^2 + f_y^2} \geq 0$$

其中 $\mu$ 是摩擦系数，$f_z$ 是法向力，$f_x, f_y$ 是切向力。用 CBF-QP 在 WBC 力分配时强制这一约束。

### 关节限位约束

关节角度 $q_i$ 必须保持在 $[q_{min}, q_{max}]$ 内：

$$h_{joint}^+(q_i) = q_{max} - q_i \geq 0$$
$$h_{joint}^-(q_i) = q_i - q_{min} \geq 0$$

### 碰撞避免

定义机器人与障碍物之间的最小距离为 $d(x)$，令：

$$h_{coll}(x) = d(x) - d_{safe}$$

当 $d(x) \to d_{safe}$ 时，CBF 自动减小接近速度，实现软减速。

## 高阶 CBF（HOCBF）

当系统的相对阶（relative degree）大于 1 时（即控制输入需要经过多次微分才能出现在 $\dot{h}$ 中），需要用**高阶 CBF**。对于机器人动力学（位置约束的相对阶通常为 2），定义中间量：

$$\psi_0(x) = h(x)$$
$$\psi_1(x) = \dot{\psi}_0 + \alpha_1(\psi_0)$$

约束条件变为 $\dot{\psi}_1(x, u) \geq -\alpha_2(\psi_1(x))$，其对控制输入 $u$ 仍然是线性的。

## 常见误区

1. **CBF 不等于稳定性**：CBF 只保证安全集的不变性，不保证系统收敛到目标状态——这是 CLF 的工作。两者需要联合使用（CLF-CBF-QP）。

2. **CBF 条件可能无解**：当多个 CBF 约束互相冲突时，QP 可能无可行解（infeasible）。需要引入松弛变量或调整约束优先级。

3. **$h(x)$ 的设计是关键**：$h$ 需要满足一定的正则性条件，且 $\partial h / \partial x \cdot g(x) \neq 0$（否则约束退化）。设计不当的 $h$ 会导致保守性过高或约束失效。

4. **离散时间系统需要特殊处理**：上述分析基于连续时间，离散时间版本需要用离散 CBF（DCBF），约束形式有所不同。

## 参考来源

- Ames et al., *Control Barrier Function Based Quadratic Programs for Safety Critical Systems* (IEEE TAC, 2017) — CBF-QP 奠基论文
- Ames et al., *Control Barrier Functions: Theory and Applications* (ECC, 2019) — 综述
- [sources/papers/optimal_control_theory.md](../../sources/papers/optimal_control_theory.md) — 最优控制与安全约束背景

## 关联页面

- [Whole-Body Control](./whole-body-control.md) — CBF 作为安全过滤器叠加在 WBC 之上
- [Contact Dynamics](./contact-dynamics.md) — 接触力锥约束是 CBF 在机器人中的主要应用场景
- [LQR](../formalizations/lqr.md) — CBF-QP 与 LQR 都是基于 QP 的实时优化，可结合使用
- [Control Lyapunov Function](../formalizations/control-lyapunov-function.md) — CLF 负责稳定性，CBF 负责安全性，联合形成 CLF-CBF-QP
- [CLF vs CBF](../comparisons/clf-vs-cbf.md) — 两者的详细对比与选型指南

## 推荐继续阅读

- Ames et al., [*Control Barrier Functions: Theory and Applications*](https://arxiv.org/abs/1903.11199)
- Wei et al., *Safe Control with Learning-Based CBF Estimation* — 学习 CBF 方向
- [CLF vs CBF 对比](../comparisons/clf-vs-cbf.md)

## 一句话记忆

> CBF 把安全约束变成 QP 里的线性不等式，让机器人控制器在保持名义性能的同时，以可证明的方式永远不进入危险区域。
