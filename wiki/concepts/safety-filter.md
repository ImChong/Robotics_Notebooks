---
type: concept
tags: [safety, control, cbf, safe-rl, wbc, deployment]
status: complete
updated: 2026-04-20
summary: "Safety Filter 指位于高层策略与低层执行器之间的安全过滤层，用最小修改把名义动作投影回可执行安全集。"
related:
  - ./control-barrier-function.md
  - ./whole-body-control.md
  - ../formalizations/control-lyapunov-function.md
  - ../queries/clf-cbf-in-wbc.md
  - ../queries/robot-policy-debug-playbook.md
sources:
  - ../../sources/papers/optimal_control.md
  - ../../sources/papers/sim2real.md
---

# Safety Filter（安全过滤器）

**Safety Filter**：位于高层策略和低层控制器之间的一层在线修正模块。它接收一个“名义动作”或“候选控制输入”，在尽量少改动原动作的前提下，强制满足安全约束，例如关节限位、碰撞距离、接触力边界和速度上限。

## 一句话定义

安全过滤器的目标不是重新规划整个任务，而是在最后一层把“不安全动作”改成“最接近原意的安全动作”。

## 为什么重要

在机器人系统里，很多高层策略都不自带可证明安全性：

- RL / IL / VLA 输出可能抖动或越界
- MPC 的名义解可能因为模型误差在真机上失效
- WBC 在约束切换时可能出现瞬时不可行

这时安全过滤器的价值在于：

1. **把安全与智能解耦**：高层专注做任务，过滤器专注做底线保护。
2. **部署代价低**：无需重训策略，就能为现有控制栈加一层安全壳。
3. **便于调试**：当系统失败时，可以区分“策略错”还是“保护层没接住”。

## 常见实现方式

### 1. CBF-QP 安全过滤

最典型做法是把名义控制输入 $u_{nom}$ 投影到满足 CBF 约束的可行域：

$$
\min_u \frac{1}{2}\|u-u_{nom}\|^2
$$

$$
\text{s.t.}\ \dot{h}(x,u)\ge -\gamma h(x)
$$

这种形式的优点是：改动最小、实时可解、可给出安全集不变性的理论保证。

### 2. 几何 / 规则式过滤

在工程里也常见更简单的版本：

- 关节速度、加速度、力矩 clamp
- workspace 边界裁剪
- 碰撞距离阈值触发减速或停机
- 动作差分限幅（rate limiter）

这类方法理论保证较弱，但实现简单、算力需求低。

### 3. 分层安全过滤

对复杂系统，常把过滤拆成两层：

- **语义层过滤**：检查任务是否合法，例如禁止机械臂进入人类工作区
- **执行层过滤**：检查数值约束，例如关节、速度、接触力

## 在机器人控制栈中的位置

```text
高层策略（RL / IL / VLA / MPC）
        ↓ 名义动作
Safety Filter
  - 约束检查
  - 最小修改
  - fallback / hold / retract
        ↓ 安全动作
低层控制器（PD / impedance / WBC）
        ↓
执行器
```

## 典型应用

- **Safe RL**：策略输出先过安全过滤，再发给执行器
- **WBC / MPC**：作为额外安全层，处理关节限位、碰撞避免、接触力锥
- **VLA 部署**：对大模型输出的动作块做限幅、裁剪、回退

## 常见误区

- **误区 1：有 safety filter 就不需要改策略。**  
  过滤器能兜底，但不能替代策略本身的质量；若长期大量修正，说明上层策略本身有问题。
- **误区 2：安全过滤器一定会让动作变保守。**  
  它只在接近危险边界时显著介入；设计得好时，正常区域内几乎不影响性能。
- **误区 3：只有 RL 需要 safety filter。**  
  任何存在模型误差、噪声、延迟或黑盒模块的控制栈都需要它。

## 参考来源

- [sources/papers/optimal_control.md](../../sources/papers/optimal_control.md) — QP 约束控制与安全约束背景
- [sources/papers/sim2real.md](../../sources/papers/sim2real.md) — 真机部署中的安全与调试经验
- Ames et al., *Control Barrier Function Based Quadratic Programs for Safety Critical Systems*

## 关联页面

- [Control Barrier Function](./control-barrier-function.md)
- [Whole-Body Control](./whole-body-control.md)
- [Control Lyapunov Function](../formalizations/control-lyapunov-function.md)
- [Query：CLF 与 CBF 在 WBC/MPC 中的联合使用](../queries/clf-cbf-in-wbc.md)
- [Query：RL 策略真机调试 Playbook](../queries/robot-policy-debug-playbook.md)
