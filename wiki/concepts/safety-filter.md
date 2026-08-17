---
type: concept
tags: [safety, control, cbf, safe-rl, wbc, deployment]
status: complete
updated: 2026-08-17
summary: "Safety Filter 指位于高层策略与低层执行器之间的安全过滤层，用最小修改把名义动作投影回可执行安全集。"
related:
  - ./control-barrier-function.md
  - ./whole-body-control.md
  - ../formalizations/control-lyapunov-function.md
  - ../queries/clf-cbf-in-wbc.md
  - ../queries/robot-policy-debug-playbook.md
  - ../entities/paper-importance-sampling-pca-av-failures.md
  - ../entities/paper-pac-man-perceptive-cbf-rl.md
  - ../entities/paper-fail-passive-gap.md
sources:
  - ../../sources/papers/optimal_control.md
  - ../../sources/papers/sim2real.md
  - ../../sources/papers/importance_sampling_pca_av_failures_arxiv_2607_18106.md
  - ../../sources/papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md
  - ../../sources/papers/fail_passive_gap_arxiv_2608_02809.md
---

# Safety Filter（安全过滤器）

**Safety Filter**：位于高层策略和低层控制器之间的一层在线修正模块。它接收一个“名义动作”或“候选控制输入”，在尽量少改动原动作的前提下，强制满足安全约束，例如关节限位、碰撞距离、接触力边界和速度上限。

## 一句话定义

安全过滤器的目标不是重新规划整个任务，而是在最后一层把“不安全动作”改成“最接近原意的安全动作”。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| IL | Imitation Learning | 从专家演示学习策略，奖励难定义时的主路线 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| MPC | Model Predictive Control | 滚动时域内优化控制序列的预测控制 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| CBF | Control Barrier Function | 用前向不变集保证安全约束的控制屏障函数 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| CLF | Control Lyapunov Function | 以能量函数衰减保证稳定性的控制李雅普诺夫函数 |

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
- **对照：训练期屏障、部署无滤波** — [PAC-MAN](../entities/paper-pac-man-perceptive-cbf-rl.md) 把 Joint-CBF 投影仅作仿真 `+filter` 上限；真机部署的 Link-CBF 策略**不**走运行时安全层，靠训练内化避碰
- **对照：认证功能安全** — 过滤器给的是控制层约束，不是 ISO 13849 的 PFHD/PL。工业人形保护停还卡在机侧反应链，见 [Fail-Passive Gap](../entities/paper-fail-passive-gap.md)

## 常见误区

- **误区 1：有 safety filter 就不需要改策略。**  
  过滤器能兜底，但不能替代策略本身的质量；若长期大量修正，说明上层策略本身有问题。
- **误区 2：安全过滤器一定会让动作变保守。**  
  它只在接近危险边界时显著介入；设计得好时，正常区域内几乎不影响性能。
- **误区 3：只有 RL 需要 safety filter。**  
  任何存在模型误差、噪声、延迟或黑盒模块的控制栈都需要它。
- **误区 4：更强的在线 CBF 投影总能搬到真机。**  
  若滤波器要读真值球态/完整威胁几何，而机载感知给不出，则只能当仿真上限（见 PAC-MAN Joint-CBF +filter）。

## 参考来源

- [sources/papers/optimal_control.md](../../sources/papers/optimal_control.md) — QP 约束控制与安全约束背景
- [sources/papers/sim2real.md](../../sources/papers/sim2real.md) — 真机部署中的安全与调试经验
- Ames et al., *Control Barrier Function Based Quadratic Programs for Safety Critical Systems*
- [PAC-MAN 论文策展](../../sources/papers/pac_man_perceptive_cbf_rl_arxiv_2607_28623.md) — 训练期 CBF vs 部署无滤波
- [Fail-Passive Gap 论文策展](../../sources/papers/fail_passive_gap_arxiv_2608_02809.md) — 功能安全认证边界，与在线过滤互补

## 关联页面

- [Control Barrier Function](./control-barrier-function.md)
- [Whole-Body Control](./whole-body-control.md)
- [Control Lyapunov Function](../formalizations/control-lyapunov-function.md)
- [Query：CLF 与 CBF 在 WBC/MPC 中的联合使用](../queries/clf-cbf-in-wbc.md)
- [Query：RL 策略真机调试 Playbook](../queries/robot-policy-debug-playbook.md)
- [真机安全 RL 微调](./safe-real-world-rl-fine-tuning.md) — 安全过滤作为真机微调三路径之一（CBF/CLF 安全壳）
- [Sim2Real 闭环误差分层工程](../queries/sim2real-closed-loop-engineering.md) — 部署段分层安全独立于策略
- [Importance Sampling + PCA（商业 AV 失败挖掘）](../entities/paper-importance-sampling-pca-av-failures.md) — 离线稀有失败发现与 eigenfailure 诊断；与在线过滤互补
- [PAC-MAN](../entities/paper-pac-man-perceptive-cbf-rl.md) — 感知感知 CBF-RL；部署刻意去掉运行时滤波
- [ActFovea](../entities/paper-actfovea.md) — 感知侧一致性防护：不给几何安全保证，但覆盖安全滤波管不到的「观测本身失真/失效」
- [Fail-Passive Gap](../entities/paper-fail-passive-gap.md) — 算法安全过滤 ≠ 可认证保护停；双足切电本身是危害
