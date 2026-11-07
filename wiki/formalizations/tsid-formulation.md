---
type: formalization
tags: [control, whole-body-control, math, optimization, dynamics]
status: complete
updated: 2026-11-07
related:
  - ../concepts/whole-body-control.md
  - ../concepts/floating-base-dynamics.md
  - ./friction-cone.md
sources:
  - ../../sources/papers/delprete_tsid_arxiv_1410_3863.md
  - ../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md
  - ../../sources/papers/whole_body_control.md
original:
  - ../../sources/papers/delprete_tsid_arxiv_1410_3863.md
  - ../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md
summary: "任务空间逆动力学（TSID）形式化：原论文（全驱动、伪逆+零空间投影、O(n)）与浮动基座+接触+不等式的 QP/HQP 扩展（Saab 2013）两代写法，给出决策变量、动力学/接触/摩擦约束与任务代价，是 WBC 最主流的数学实现方式。"
---

# Task Space Inverse Dynamics (TSID) 形式化

## 一句话定义

**TSID** 是一种在保持机器人物理一致性的前提下，实现多任务并行控制的数学框架：把任务空间的期望加速度，经"逆运动学（加速度层）+ 逆动力学"转换成底层的电机力矩（以及接触力）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TSID | Task-Space Inverse Dynamics | 任务空间逆动力学求解关节力矩的 WBC 实现 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| HQP | Hierarchical Quadratic Programming | 按严格优先级逐层求解的二次规划 |
| RNEA | Recursive Newton-Euler Algorithm | 递归牛顿-欧拉算法，$O(n)$ 计算逆动力学力矩 |
| UF | Unifying Framework | Peters 等的统一框架，原论文用作对比基线 |
| WBCF | Whole-Body Control Framework | Sentis–Khatib 的全身控制框架，原论文用作对比基线 |

## 为什么重要

- **自动协调**：你只需告诉手去哪里，TSID 会自动计算为了补偿手部动量，躯干应该如何反向摆动。
- **优先级处理**：任务冲突时让安全 / 接触任务优先于动作性能；原文 Test 2 证明在目标不可达（任务冲突）时，高优先级任务（力控、左手轨迹）误差仍保持在亚毫米级，冲突只落在低优先级任务上。
- **统一了力与运动**：刚性接触被当作最高优先级"任务"，运动与力控制用同一套框架。

## 核心原理

> **先分清两代写法**（已对照原文）：Del Prete 等 2014 的原论文只处理**全驱动**机器人，解是伪逆 + 零空间投影的**解析式**，不含不等式；浮动基座、接触力互补条件与不等式约束的 QP/HQP 写法来自 Saab 等 2013 及后续工程实现。

### A. 原论文形式（全驱动，无不等式）

动力学 $M(q)\ddot q + h(q,\dot q) = \tau$；任务 $f(q)=x_r(t)$ 对时间微分两次得 $J\ddot q+\dot J\dot q=\ddot x$，期望任务加速度

$$ \ddot x^* = \ddot x_r + K_d(\dot x_r-\dot x) + K_p(x_r-x). $$

从最高优先级 $N$ 到最低的姿态任务 $0$（$J_0=I$，$\ddot x_0^*=\ddot q_p^*=K_p(q_p-q)-K_d\dot q$）逐级求解，$W=I$ 时（原文式 10）：

$$ \tau^* = M\big(\ddot q_1 + N_{p(0)}\ddot q_p^*\big) + h,\qquad \ddot q_i=\ddot q_{i+1}+(J_iN_{p(i)})^{\dagger}\big(\ddot x_i^*-\dot J_i\dot q-J_i\ddot q_{i+1}\big). $$

因为有姿态任务，**$\tau^*$ 与权重矩阵 $W$ 无关**；运动学（求 $\ddot q$）与动力学（求 $\tau$）完全解耦，可用 RNEA 以 $O(n)$ 计算，无需显式 $M$ 或任务空间惯量 $\Lambda$。含刚性接触时 $M\ddot q+h-J_c^T f=\tau$，$J_c\ddot q=-\dot J_c\dot q$，解为 $\tau^*=M(J_c^\dagger b+N_c\ddot q_0)+h-J_c^Tf^*$。

### B. 浮动基座 + 接触 + 不等式：QP 形式（Saab 2013 起）

#### 1. 决策变量
$$ \mathcal{X} = [\ddot{q}^T, \mathbf{f}_{ext}^T, \tau^T]^T $$
其中 $\ddot{q}$ 是广义加速度，$\mathbf{f}_{ext}$ 是接触力，$\tau$ 是力矩（Saab 原文取 $u=(\tau,\ddot q,f)$；开源库 `tsid` 只取 $[\ddot q, f]$，再事后解出 $\tau$）。

#### 2. 硬约束 (Hard Constraints)
- **浮动基座动力学**（Saab 式 21）：
  $$ M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = S^T \tau + \sum J_i^T \mathbf{f}_{ext, i} $$
  其中 $M(q)$ 是惯性矩阵，$C(q,\dot q)$ 表示科氏/离心项，$g(q)$ 是重力项（合写为 $h$ / $b$），$J_i$ 是第 $i$ 个接触点的雅可比；$S$ 是**执行选择矩阵 (selection / actuation matrix)**，把可控的关节力矩 $\tau$ 投影到完整的 $n$ 维广义坐标上（前 6 维浮动基座对应位置全为 0，因此浮动基不可直接驱动）。
- **接触一致性**（支撑脚不滑移，Saab 式 22）：
  $$ J_i \ddot{q} + \dot{J}_i \dot{q} = 0 $$
- **单边接触条件**（Saab 式 27–29）：法向加速度 $\ddot x\ge0$、法向力 $f^\perp\ge0$、互补 $\ddot x\,f^\perp=0$。只加零运动约束而不显式要求 $f^\perp\ge0$ 会生成不可行的动态运动。
- **摩擦锥约束**：详见 [Friction Cone](./friction-cone.md)；`tsid` 库的 `Contact6d` 用线性化摩擦锥（金字塔）+ 法向力上下界。
- **平面接触降维**：Saab 用接触体 6D 广义力 $\phi$ 代替每个接触点一个力变量，平地站立时接触条件等价于 ZMP 在支撑多边形内。

#### 3. 目标函数 (Cost Function)
$$ \min_{\mathcal{X}} \sum w_k \| J_k \ddot{q} + \dot{J}_k \dot{q} - \ddot{x}_{des, k} \|^2 + \lambda \|\tau\|^2 $$
其中 $\ddot{x}_{des, k}$ 是各个任务（如 Base 平衡、手部跟踪）在任务空间期望的加速度；$\lambda\|\tau\|^2$ 是常见的力矩正则项（工程写法，原文两篇未专门论证）。

> **加权 vs 严格层级**：上式是单层加权和；严格优先级要写成 Saab 的层级 (21) ≺ (22) ≺ (28) ≺ (14)（动力学 ≺ 接触不动 ≺ 法向力非负 ≺ 任务），见 [HQP](../concepts/hqp.md)。

## 工程实践

- **接触可行性检查**：若"法向力非负"层的松弛变量非零（Saab 称强接触不稳定），解动力学不一致，应切换接触模式，而不是照样下发。
- **姿态任务必不可少**：原文证明加上最低优先级姿态任务后解唯一且与权重无关，同时稳定关节空间零空间。
- **开源实现**：[TSID](../concepts/tsid.md) 页给出 `stack-of-tasks/tsid` 的用法（硬约束层 + 加权任务层，默认求解器至多 2 层）。
- **数值**：原文用阻尼伪逆 $\lambda=0.02$ 防奇异；仿真增益 $K_p=10\,s^{-2}$、$K_d=5\,s^{-1}$（仅论文仿真取值）。

## 局限

- 原论文限定全驱动、无不等式、只保证瞬时最优（需要 MPC 把时域考虑进去）。
- Saab 2013 的 HQP 运动是离线生成（"接近但尚未实时"），实时闭环需要更快的求解器。
- 线性化摩擦锥与刚性接触假设在软地面、滑移场景下不成立（推测，原文未讨论）。

## 关联页面

- [仿真物理保真度链路选型指南](../queries/simulation-physics-fidelity.md) — 本页所述物理/仿真要素在保真度链路（建模 ① → 数值 ② → 接触 ③ → 随机化 ④）中的定位
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [TSID（方法页）](../concepts/tsid.md)
- [HQP](../concepts/hqp.md)
- [Floating Base Dynamics](../concepts/floating-base-dynamics.md)
- [Friction Cone 形式化](./friction-cone.md)

## 参考来源

- [Del Prete et al. 2014 — TSID 原论文通读归档](../../sources/papers/delprete_tsid_arxiv_1410_3863.md)（arXiv:1410.3863）
- [Saab et al. 2013 — 刚性接触下动态全身运动生成通读归档](../../sources/papers/saab_dynamic_whole_body_motion_tro_2013.md)
- [sources/papers/whole_body_control.md](../../sources/papers/whole_body_control.md)

## 推荐继续阅读

- Del Prete et al., *Prioritized motion-force control of constrained fully-actuated robots*：<https://arxiv.org/abs/1410.3863>
- Saab et al., *Dynamic whole-body motion generation under rigid contacts*：<https://hal-lirmm.ccsd.cnrs.fr/lirmm-00831097/document>
