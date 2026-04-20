---
type: query
tags: [control, clf, cbf, wbc, mpc, qp, stability, safety, optimization, humanoid]
status: complete
summary: "CLF 与 CBF 如何在 WBC/MPC 中联合使用：通过 CLF-CBF-QP 框架同时保证系统收敛到目标（CLF）和维持安全集（CBF），本页梳理公式、冲突处理策略与实际工程实现要点。"
sources:
  - ../../sources/papers/optimal_control.md
related:
  - ../formalizations/control-lyapunov-function.md
  - ../concepts/control-barrier-function.md
  - ../concepts/whole-body-control.md
  - ../comparisons/clf-vs-cbf.md
---

# Query：CLF 与 CBF 在 WBC/MPC 中的联合使用

> **Query 产物**：本页由以下问题触发：「CLF 和 CBF 如何在 WBC/MPC 中联合使用保证稳定性与安全性？」
> 综合来源：[Control Lyapunov Function](../formalizations/control-lyapunov-function.md)、[Control Barrier Function](../concepts/control-barrier-function.md)、[Whole-Body Control](../concepts/whole-body-control.md)、[CLF vs CBF 对比](../comparisons/clf-vs-cbf.md)

## TL;DR 核心结论

| 工具 | 角色 | 约束方向 | 优先级 |
|------|------|---------|--------|
| **CLF**（控制李雅普诺夫函数） | 驱动系统收敛到目标状态（**稳定性**） | $\dot{V} \leq -cV + \delta$（软约束，允许松弛） | 次优先 |
| **CBF**（控制屏障函数） | 维持系统在安全集内（**安全性**） | $\dot{h} \geq -\gamma h$（硬约束，不允许违反） | 最优先 |
| **CLF-CBF-QP** | 两者组合成统一 QP，实时求解使系统既趋向目标又不越安全边界的控制输入 | — | — |

**决策规则**：当 CLF 约束（快速靠近目标）与 CBF 约束（远离危险边界）发生冲突时，CBF 优先——系统会减慢收敛速度甚至绕路，但绝不越安全边界。

---

## 详细内容

### 1. CLF-CBF-QP 标准公式

#### 1.1 问题设定

考虑仿射控制系统：

$$\dot{x} = f(x) + g(x)u$$

- **CLF** $V(x)$：正定函数，$V(0)=0$，$V(x)>0 \; \forall x\neq 0$，满足 CLF 条件时系统指数收敛
- **CBF** $h(x)$：$h(x)\geq 0$ 对应安全集 $\mathcal{C}$，CBF 条件维持集合不变性

#### 1.2 联合 QP 公式

$$\min_{u,\,\delta} \quad \frac{1}{2}\|u - u_{\text{ref}}\|^2 + p\,\delta^2$$

$$\text{s.t.} \quad \underbrace{\dot{V}(x,u) \leq -c\,V(x) + \delta}_{\text{CLF 软约束（稳定性）}}$$

$$\qquad\quad \underbrace{\dot{h}(x,u) \geq -\gamma\,h(x)}_{\text{CBF 硬约束（安全性）}}$$

$$\qquad\quad u \in \mathcal{U}, \quad \delta \geq 0$$

**变量说明**：
- $u_{\text{ref}}$：来自高层控制器（WBC / MPC / RL 策略）的名义控制输入
- $\delta$：CLF 松弛变量，当 CLF 与 CBF 冲突时允许 CLF 条件被适当放宽
- $p$：CLF 松弛惩罚权重，越大则越要求快速收敛（但与 CBF 冲突越激烈）
- $c$：CLF 收敛速率参数，通常取 $c \in [1, 10]$（Hz 量级）
- $\gamma$：CBF 安全衰减速率参数，越大则安全边界越"硬"（CBF 越保守）

#### 1.3 约束的线性性——为何能实时求解

对于仿射控制系统，展开 $\dot{V}$ 和 $\dot{h}$：

$$\dot{V}(x,u) = \underbrace{\frac{\partial V}{\partial x} f(x)}_{L_f V} + \underbrace{\frac{\partial V}{\partial x} g(x)}_{L_g V} u$$

$$\dot{h}(x,u) = \underbrace{\frac{\partial h}{\partial x} f(x)}_{L_f h} + \underbrace{\frac{\partial h}{\partial x} g(x)}_{L_g h} u$$

两个约束对控制输入 $u$ 均为**线性**，整个问题是标准 QP（二次目标 + 线性约束），可在微秒级内用现成求解器（OSQP、qpOASES 等）求解。

---

### 2. 冲突处理：CBF 优先策略

#### 2.1 冲突发生的场景

CLF 和 CBF 约束发生冲突，通常出现在以下情况：

| 场景 | CLF 意图 | CBF 意图 | 冲突表现 |
|------|---------|---------|---------|
| 系统靠近安全边界 | 快速向目标移动（穿越边界） | 减速/停止以维持 $h \geq 0$ | CLF 要求加速，CBF 要求减速 |
| 多个安全约束交叉 | 向目标前进 | 同时受多个 CBF 约束限制 | 可行域变小甚至为空 |
| 目标位于安全集边界附近 | 收敛到 $h(x)\approx 0$ 处 | 维持 $h \geq 0$ | 两者在边界争夺控制权 |

#### 2.2 松弛 CLF 的机制

CLF 约束引入松弛变量 $\delta$ 后，惩罚权重 $p$ 控制"愿意牺牲多少收敛速度来满足安全约束"：

```
p 较小 → 允许 CLF 条件被大幅松弛（收敛慢，但 QP 更稳定）
p 较大 → CLF 条件被强制执行（收敛快，但与 CBF 冲突更激烈）
```

工程经验：取 $p \in [10^3, 10^5]$ 通常能在收敛性和可行性之间取得平衡。

#### 2.3 CBF 无可行解时的处理

当多个 CBF 约束同时激活导致 QP 无可行解时，处理方案：

1. **软化 CBF 约束**：为 CBF 也引入松弛变量，但惩罚权重设置极大（数值上仍保持强约束效果）
2. **调小 $\gamma$**：减小 CBF 约束的激进程度，在代价是安全集边界保守性稍差
3. **优先级排序**：对多个 CBF 构造 HQP（分层 QP），高优先级安全约束不松弛，低优先级的允许松弛

---

### 3. CLF-CBF-QP 在 WBC 中的位置

#### 3.1 WBC 的层次结构与 CLF-CBF-QP 的嵌入点

```
高层规划（MPC / 步态生成）
         ↓ 参考轨迹 x_ref(t)
CLF-CBF-QP 安全过滤层
  - CLF：驱动末端效应器/CoM 跟踪参考轨迹
  - CBF：关节限位 / 接触力锥 / 碰撞避免
         ↓ 过滤后的任务空间加速度 ä_des
TSID / HQP 全身力矩分配
  - 逆动力学 + 接触力分配
  - 关节力矩约束
         ↓ 关节力矩命令 τ
电机驱动器
```

CLF-CBF-QP 通常作为**任务空间约束层**，叠加在 TSID/HQP 之上。它负责在任务空间决定哪些加速度命令是"稳定且安全"的，再将过滤后的命令传递给逆动力学求解器。

#### 3.2 与 TSID/HQP 的对比

| 维度 | TSID/HQP | CLF-CBF-QP |
|------|---------|-----------|
| 处理内容 | 多任务优先级 + 全身力矩分配 | 稳定性条件 + 安全集维持 |
| 约束来源 | 接触力、关节限位、任务目标 | CLF（$\dot{V}$）、CBF（$\dot{h}$）函数值 |
| 求解变量 | 关节加速度 + 接触力 | 控制输入（任务空间或关节空间） |
| 可证明性 | 无显式 Lyapunov 保证 | 有可证明的稳定性与安全性 |
| 组合方式 | CLF-CBF 作为约束叠加到 HQP | CLF-CBF-QP 前置过滤，TSID 执行 |

#### 3.3 在 MPC 中的集成方式

在 MPC 框架中使用 CBF 约束，有两种典型方式：

**方式一：在 MPC 预测时域内每步加入 CBF 约束**

```
min_{u_0,...,u_{N-1}} Σ cost(x_k, u_k)
s.t.  x_{k+1} = f(x_k, u_k)           # 动力学
      h(x_k) + γ·Δt·h(x_k) ≥ 0        # 离散 CBF（每步）
      V(x_N) ≤ ε                        # 终端 CLF 约束
```

**方式二：MPC 输出名义指令，CBF-QP 做在线安全过滤**

```
MPC → u_nom → CBF-QP → u_safe → 执行器
```

方式二更适合已有 MPC 框架、只需叠加安全保证的工程场景。

---

### 4. 典型 CBF 设计举例（人形机器人）

#### 4.1 关节限位 CBF

$$h_{q,i}^+(q) = q_{i,\max} - q_i \geq 0, \quad h_{q,i}^-(q) = q_i - q_{i,\min} \geq 0$$

自动在关节接近限位时限制关节速度，不需要手动 clamp。

#### 4.2 接触力摩擦锥 CBF

$$h_{\text{fric}}(f) = \mu f_z - \sqrt{f_x^2 + f_y^2} \geq 0$$

在 WBC 力分配中强制接触力落在摩擦锥内，避免脚底打滑。

#### 4.3 CoM 高度安全约束

$$h_{\text{CoM}}(x) = z_{\text{CoM}} - z_{\min} \geq 0$$

防止 CoM 高度低于安全阈值（即将跌倒），可结合 CLF 加速 CoM 恢复。

#### 4.4 碰撞避免 CBF

$$h_{\text{coll}}(x) = \|p_{\text{ee}}(q) - p_{\text{obs}}\| - d_{\text{safe}} \geq 0$$

末端执行器与障碍物距离保持在安全距离以上，CBF 自动产生排斥力效果。

---

### 5. 数值实现注意事项

#### 5.1 QP 规模与实时性

典型人形机器人 WBC 的 CLF-CBF-QP 规模：
- **变量数**：控制自由度（30~50 个关节加速度）+ 松弛变量（$\delta$）
- **约束数**：CLF 约束 1 条 + CBF 约束若干条（关节限位 2×N + 接触力锥 4/摩擦椎体面 + 碰撞约束）
- **求解时间目标**：< 1 ms（以支持 1 kHz 控制频率）

推荐求解器：
- **OSQP**：开源，稀疏 QP，1 kHz 下通常可行
- **qpOASES**：active-set 方法，warm start 效果好，适合约束频繁激活的场景
- **HPIPM**：结构化 QP（适合 MPC），效率极高

#### 5.2 CBF 参数 $\gamma$ 的选取

| $\gamma$ 取值 | 效果 | 适用场景 |
|-------------|------|---------|
| $\gamma$ 较小（如 0.5） | CBF 约束宽松，系统可以"慢慢靠近"安全边界 | 任务需要精确接触，安全边界宽松 |
| $\gamma$ 较大（如 5.0） | CBF 约束激进，远离安全边界时也会产生约束 | 高安全要求，保守操作 |
| 自适应 $\gamma(x)$ | 随状态调整保守程度 | 高动态环境，需平衡安全与性能 |

#### 5.3 高阶 CBF（HOCBF）

当系统相对阶大于 1（位置约束需要经过两次微分才能出现控制输入），需要使用高阶 CBF：

```
ψ_0(x) = h(x)
ψ_1(x) = ψ̇_0 + α_1(ψ_0)
约束: ψ̇_1(x, u) ≥ -α_2(ψ_1(x))
```

机器人关节位置约束（相对阶 = 2）通常需要 HOCBF，或者直接在速度层设计 CBF（相对阶 = 1）。

#### 5.4 离散时间注意事项

连续时间 CLF/CBF 理论不能直接用于离散时间系统。在控制频率有限（如 50 Hz，步长 20 ms）时，应使用离散 CBF（DCBF）：

$$h(x_{k+1}) \geq (1 - \gamma \Delta t) h(x_k)$$

否则在低频控制下，连续时间条件的离散近似会引入数值误差，导致安全约束实际上被违反。

---

## 参考来源

- Ames et al., *Control Barrier Function Based Quadratic Programs for Safety Critical Systems* (IEEE TAC, 2017) — CLF-CBF-QP 奠基论文，完整公式推导
- Ames et al., *Control Barrier Functions: Theory and Applications* (ECC, 2019) — 综述，覆盖 HOCBF 与机器人应用
- Galloway et al., *Torque Saturation in Bipedal Robotic Walking through Control Lyapunov Function Based Quadratic Programs* — 双足行走中的 CLF-QP 实现
- [sources/papers/optimal_control.md](../../sources/papers/optimal_control.md) — 最优控制与 Lyapunov 方法背景

## 关联页面

- [Control Lyapunov Function（CLF）](../formalizations/control-lyapunov-function.md) — CLF 完整定义与 QP 推导
- [Control Barrier Function（CBF）](../concepts/control-barrier-function.md) — CBF 定义、摩擦锥约束、HOCBF
- [Whole-Body Control](../concepts/whole-body-control.md) — CLF-CBF-QP 在 WBC 中的嵌入位置
- [CLF vs CBF 对比](../comparisons/clf-vs-cbf.md) — 两者核心差异、选型决策树与常见误判
