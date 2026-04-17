---
type: formalization
tags: [control, stability, locomotion, nonlinear-systems, optimal-control]
status: complete
summary: Lyapunov 稳定性分析用能量函数/候选函数判断平衡点附近误差是否收敛，是分析人形机器人闭环稳定性的基础工具。
related:
  - ../formalizations/lqr.md
  - ../concepts/whole-body-control.md
  - ../tasks/locomotion.md
  - ../concepts/centroidal-dynamics.md
sources:
  - ../../sources/papers/optimal_control.md
  - ../../sources/papers/optimal_control_theory.md
---

# Lyapunov 稳定性

**Lyapunov 稳定性**：通过构造一个随系统状态变化的标量函数 $V(x)$ 来判断平衡点附近的误差是否收敛。对机器人控制来说，它回答的是："这个控制器不仅能把误差压小，而且能持续保持稳定吗？"

## 一句话定义

> Lyapunov 方法把稳定性问题变成“能量函数是否持续下降”的问题，是分析运动控制、姿态稳定和闭环收敛性的通用语言。

## 为什么重要

- 人形机器人 locomotion、平衡恢复、姿态控制都需要讨论稳定性，而不仅是“能不能跟踪轨迹”
- 许多控制器（LQR、MPC 终端设计、阻抗控制、WBC 局部线性化）最终都要回到稳定性分析
- 在接触切换和非线性动力学场景里，Lyapunov 语言比单纯线性系统极点分析更通用

## 基本形式

考虑平衡点 $x^*=0$ 的非线性系统：

$$\dot{x} = f(x)$$

若存在一个标量函数 $V(x)$ 满足：

$$V(0)=0, \quad V(x) > 0 \; (x \neq 0)$$

且沿系统轨迹的导数满足：

$$\dot{V}(x) = \nabla V(x)^T f(x) \le 0$$

则平衡点至少是 **Lyapunov 稳定** 的；如果进一步有：

$$\dot{V}(x) < 0 \; (x \neq 0)$$

则平衡点是 **渐近稳定** 的，误差会随时间衰减到 0。

## 常见稳定性层级

| 层级 | 条件 | 含义 |
|------|------|------|
| Lyapunov 稳定 | $\dot{V}(x) \le 0$ | 扰动后状态不发散 |
| 渐近稳定 | $\dot{V}(x) < 0$ | 状态最终回到平衡点 |
| 指数稳定 | $\dot{V}(x) \le -\alpha V(x)$ | 误差按指数速度衰减 |

## 在机器人控制中的典型用法

### 1. 线性反馈控制
LQR 的二次型代价函数和 Riccati 方程解本质上就给出了一个 Lyapunov 函数，因此能解释为什么 LQR 在线性化工作点附近具有稳定性保证。

### 2. 腿足 / 人形平衡控制
在步态稳定、质心控制、Capture Point / DCM 分析中，常把误差动力学写成可验证收敛的形式，再用 Lyapunov 函数说明闭环不会发散。

### 3. 非线性控制与阻抗控制
阻抗控制、滑模控制、backstepping 等方法经常需要手工构造 Lyapunov 候选函数，证明关节空间或任务空间误差收敛。

## 与机器人学习的关系

- 在强化学习里，Lyapunov 视角常用于分析安全约束、稳定性奖励设计和安全 RL
- 在 sim2real 中，一个策略即使奖励高，也可能没有稳定性裕度；Lyapunov 分析能帮助区分“会走”和“稳定可部署”
- 在 VLA / 端到端策略场景中，通常缺少显式 Lyapunov 证明，因此部署时更依赖外层安全控制器兜底

## 常见误区

- **误区 1：稳定 = 轨迹误差很小。** 误差小只说明当前状态好，不代表受到扰动后仍能保持稳定。
- **误区 2：有 Lyapunov 函数就一定全局稳定。** 还要看函数定义域和负定条件，很多机器人结论只是局部稳定。
- **误区 3：学习策略不需要稳定性分析。** 真机部署时，稳定性边界和安全约束仍然很关键。

## 关联页面

- [LQR / iLQR](./lqr.md) — LQR 的二次型价值函数可视作 Lyapunov 函数
- [Whole-Body Control](../concepts/whole-body-control.md) — WBC 闭环设计常需讨论稳定性与约束一致性
- [Locomotion](../tasks/locomotion.md) — 步态稳定、外界扰动恢复都离不开稳定性分析
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md) — 质心动力学稳定性分析的常用建模层
- [Capture Point / DCM](../concepts/capture-point-dcm.md) — 步行平衡中的误差发散/收敛分析

## 参考来源

- [optimal_control](../../sources/papers/optimal_control.md) — LQR / iLQR / 最优控制主线
- [最优控制理论（Optimal Control Theory）](../../sources/papers/optimal_control_theory.md) — 轨迹优化与稳定性分析背景
- Khalil, *Nonlinear Systems* — 非线性系统稳定性分析经典教材

## 推荐继续阅读

- [LQR / iLQR](./lqr.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Locomotion](../tasks/locomotion.md)
- [Capture Point / DCM](../concepts/capture-point-dcm.md)

## 一句话记忆

> Lyapunov 方法的核心不是“算出一个控制器”，而是证明“这个控制器会不会让系统越控越稳”。
