---
type: formalization
tags: [contact, lcp, complementarity, simulation, optimization, contact-dynamics]
status: stable
---

# Contact Complementarity（接触互补约束）

**接触互补约束（Contact Complementarity Conditions）** 是描述刚体与环境接触物理行为的数学框架：**接触力与接触间隙必须互补为零**——要么接触面有力（接触中），要么接触面有间隙（不接触），二者不能同时非零。这个约束是 LCP（线性互补问题）建模的核心，也是仿真器接触求解的数学基础。

## 一句话定义

> 接触互补约束说的是"要么在接触（有力无间隙），要么不接触（有间隙无力）"——这个非线性互补条件是刚体接触动力学的根本约束，驱动了从仿真到 MPC 的一整套接触建模方法。

---

## 基本数学形式

### Signorini 条件（法向接触互补）

对接触点 $i$，定义：
- $\phi_i \geq 0$：接触间隙（gap，正值表示不接触）
- $f_i^n \geq 0$：法向接触力（正值表示压力）

互补条件：

$$
\phi_i \geq 0, \quad f_i^n \geq 0, \quad \phi_i \cdot f_i^n = 0
$$

等价写法（互补对）：$0 \leq \phi_i \perp f_i^n \geq 0$

**物理含义**：
- $\phi_i > 0$（空中）→ $f_i^n = 0$（无接触力）
- $\phi_i = 0$（接触）→ $f_i^n \geq 0$（可以有压力，不能是拉力）

### Coulomb 摩擦互补

切向接触力 $\mathbf{f}^t$ 受 Coulomb 锥约束：$\|\mathbf{f}^t\| \leq \mu f^n$

引入滑动速度 $\mathbf{v}^t$ 后，静/动摩擦也可以写成互补形式（但非线性，通常线性化处理）。

---

## LCP 问题形式

将接触约束写成矩阵形式，对所有接触点联立：

$$
\mathbf{w} = \mathbf{M} \mathbf{z} + \mathbf{q}, \quad \mathbf{w} \geq 0, \quad \mathbf{z} \geq 0, \quad \mathbf{w}^T \mathbf{z} = 0
$$

- $\mathbf{z}$：接触力（待求）
- $\mathbf{w}$：接触间隙或速度（待求）
- $\mathbf{M}$：Delassus 矩阵（接触点间的耦合）
- $\mathbf{q}$：自由运动项

LCP 求解算法：Lemke's method、Dantzig's method、pivot 算法

**复杂度**：一般是 NP-hard，但稀疏接触结构下实际可高效求解。

---

## 在仿真器中的实现对比

| 仿真器 | 接触模型 | 互补处理方式 |
|--------|---------|------------|
| **MuJoCo** | Soft contact（弹性） | 优化（convex QP），绕开 LCP |
| **Bullet / ODE** | Hard contact | LCP / 投影 Gauss-Seidel |
| **Drake** | LCP 或 SAP | 精确 LCP 或近似 SAP |
| **RBDL** | 基于冲量的 LCP | Lemke 算法 |
| **IsaacGym** | Soft contact | 类 MuJoCo 优化 |

MuJoCo 的 soft contact 模型**绕开了 LCP**，用连续弹性接触力替代硬接触 + 互补约束，在机器人学习场景下更稳定但物理精确度略低。

---

## 在最优控制 / MPC 中的挑战

接触互补约束带来**混合整数非线性规划（MINLP）**的困难：

- 接触是否激活是离散决策
- 力和间隙的互补是非线性约束
- 不能直接用梯度方法求解

**常见工程绕路方案**：

1. **预设接触序列**：MPC 固定接触时序，只优化连续变量（避开整数）
2. **互补松弛**（Complementarity Relaxation）：$\phi \cdot f \leq \epsilon$，允许微小违反
3. **Contact-Implicit TO**：直接将互补约束写入 NLP，用 IPOPT 等求解（Posa et al. 2014）
4. **可微分仿真**（Differentiable Simulation）：近似互补条件使梯度可传播（Drake SAP，Genesis）

---

## Contact-Implicit Trajectory Optimization

Contact-Implicit TO 是将接触互补条件直接嵌入轨迹优化的方法，无需预设接触序列：

$$
\min_{\mathbf{x}, \mathbf{u}, \boldsymbol{\lambda}} \quad J(\mathbf{x}, \mathbf{u})
$$

$$
\text{s.t.} \quad \mathbf{f}(\mathbf{x}, \dot{\mathbf{x}}, \ddot{\mathbf{x}}, \boldsymbol{\lambda}) = 0 \quad \text{(动力学)}
$$

$$
\quad 0 \leq \phi(\mathbf{x}) \perp \boldsymbol{\lambda} \geq 0 \quad \text{(互补约束)}
$$

代表工具：**Drake**（DirectCollocation + 接触约束），**ContactImplicit DDP**

---

## 参考来源

- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md) — ingest 档案（Stewart LCP 2000 / Todorov MuJoCo 2012）
- Stewart & Trinkle, *An Implicit Time-Stepping Scheme for Rigid Body Dynamics with Inelastic Collisions and Coulomb Friction* (2000)
- Posa et al., *A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact* (IJRR 2014) — Contact-Implicit TO 开创性工作

---

## 关联页面

- [Contact Dynamics](../concepts/contact-dynamics.md) — 互补约束是接触动力学的数学核心
- [Whole-Body Control](../concepts/whole-body-control.md) — WBC 中的接触约束建模依赖互补理论
- [Trajectory Optimization](../methods/trajectory-optimization.md) — Contact-Implicit TO 直接使用互补约束
- [Drake](../entities/drake.md) — Drake 提供完整 LCP 和 SAP 接触求解器

---

## 推荐继续阅读

- Posa et al., *A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact*
- Todorov, *Convex and analytically-invertible dynamics with contacts and constraints* (ICRA 2014)

## 一句话记忆

> 接触互补约束是刚体接触物理的核心数学规则：要么接触有力，要么不接触有间隙，非此即彼——它驱动了仿真器的接触求解、MPC 的接触规划和 Contact-Implicit 轨迹优化。
