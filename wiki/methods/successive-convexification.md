---
type: method
tags: [optimization, trajectory-optimization, optimal-control, sequential-convex-programming, rocket-landing]
status: complete
updated: 2026-11-06
related:
  - ../concepts/powered-descent-guidance.md
  - ./lossless-convexification.md
  - ./trajectory-optimization.md
  - ./lqr-ilqr.md
  - ../formalizations/unit-quaternion-so3.md
sources:
  - ../../sources/papers/szmuk_2018_scvx_6dof_free_final_time.md
  - ../../sources/papers/szmuk_2018_scvx_realtime_state_triggered.md
  - ../../sources/papers/malyuta_2021_convex_trajectory_generation_tutorial.md
  - ../../sources/repos/scptoolbox_jl.md
  - ../../sources/repos/scvx_embersarc.md
summary: "序列凸化（SCvx）：把非凸轨迹优化在上一轮轨迹附近线性化成凸子问题（通常是 SOCP），用虚拟控制防止人为不可行、用信赖域限制步长，迭代到收敛；是 6-DoF 火箭着陆制导的主流在线解法。"
---

# 序列凸化（Successive Convexification, SCvx）

**序列凸化**：一种序列凸规划（SCP）算法。每轮在上一条轨迹附近把非线性动力学和非凸约束线性化，解一个凸子问题，再用新轨迹做下一轮的线性化点。

## 一句话定义

> 6-DoF 火箭着陆没法一次凸化，那就"每次只凸化一小块"：线性化 → 解 SOCP → 更新轨迹，靠虚拟控制和信赖域保证每步都能解、不乱跳。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SCvx | Successive Convexification | 序列凸化，本页主题 |
| SCP | Sequential Convex Programming | 序列凸规划，SCvx/GuSTO/PTR 的总称 |
| GuSTO | Guaranteed Sequential Trajectory Optimization | 另一种 SCP，基于连续时间分析 |
| PTR | Penalized Trust Region | 罚信赖域 SCP，SCPToolbox.jl 中实现 |
| SOCP | Second-Order Cone Program | 每轮凸子问题的类型 |
| FOH | First-Order Hold | 控制在相邻节点间线性插值的离散化 |
| KKT | Karush-Kuhn-Tucker | SCvx 收敛保证所指的最优性条件 |

## 为什么重要

- **6-DoF 是真实飞控要的模型。** 姿态、万向节角、倾角、角速度约束只有在 6-DoF 里才能直接表达（[Szmuk & Açıkmeşe 2018](../../sources/papers/szmuk_2018_scvx_6dof_free_final_time.md)）。
- **有实时化证据。** Szmuk 等在 6-DoF + 气动 + 状态触发约束的问题上，MATLAB + ECOS 单次求解最长 0.682 s（[原文归档](../../sources/papers/szmuk_2018_scvx_realtime_state_triggered.md)）。
- **有飞行测试背景。** Malyuta 等称 SCvx 在 NASA SPLICE 项目中作为实验载荷在 Blue Origin New Shepard 上测试（[教程归档](../../sources/papers/malyuta_2021_convex_trajectory_generation_tutorial.md)；NASA 官方页面只说测试了 "advanced algorithms"）。

## 核心原理

### 1. 6-DoF 模型

状态 \(x=[m, r_I, v_I, q_{B/I}, \omega_B]\in\mathbb R^{14}\)，控制为机体系推力 \(T_B\)；约束含推力上下界（下界非凸）、万向节角、倾角、角速度、滑翔坡与干质量。

### 2. 自由终端时间

归一化时间 \(\tau\in[0,1]\)，伸缩因子 \(\sigma=(d\tau/dt)^{-1}\) 作为决策变量，\(x'(\tau)=\sigma f(x,u)\)。这样不需要对飞行时间做外层线搜索（LCvx 需要）。

### 3. 每轮的凸子问题

1. 在上一轮轨迹 \((\hat x,\hat u,\hat\sigma)\) 附近一阶线性化：\(A=\hat\sigma\,\partial f/\partial x\)，\(B=\hat\sigma\,\partial f/\partial u\)。
2. FOH 离散：用状态转移矩阵积分得到 \(x_{k+1}=\bar A_kx_k+\bar B_ku_k+\bar C_ku_{k+1}+\bar\Sigma_k\sigma+\bar z_k\)。
3. 推力下界线性化为 \(T_{min}\le\hat u^\top u/\|\hat u\|\)。
4. 加**虚拟控制** \(\nu_k\)（\(\ell_1\) 罚）防止线性化导致的人为不可行；加**信赖域**限制 \(\delta x,\delta u,\delta\sigma\)。
5. 收敛判据：信赖域步长与虚拟控制都足够小（Szmuk 2018 取 \(10^{-3}\)、\(10^{-10}\)）。

### 4. SCvx vs GuSTO（Malyuta 等）

| 维度 | SCvx | GuSTO |
|------|------|-------|
| 分析对象 | 离散化后的问题，保证落在 KKT 条件 | 连续时间问题，保证落在 Pontryagin 条件 |
| 信赖域 | 硬约束 | 软惩罚，权重逐步加大 |
| 处理不可行 | 虚拟控制 | 对路径约束违反的软惩罚 |
| 适用条件 | 较一般 | 代价对控制二次、动力学控制仿射 |
| 收敛 | 超线性；虚拟控制为零时得到驻点 | 可用对偶变量得到二次收敛 |

## 工程实践

- **初值**：直线插值（推力从 \(-m_{wet}g\) 插到 \(-m_{dry}g\)，四元数用 SLERP），"embed as much problem knowledge as possible in the initial guess"（Malyuta 等）。
- **缩放**："SCvx and GuSTO are not scale-invariant"，变量缩放到约 [0,1]，否则 "usually have a very hard time converging"。
- **罚权**：虚拟控制罚权从 \(10^2\)–\(10^4\) 试起；收敛后虚拟控制仍非零，先加大罚权。
- **离散化**：FOH 是常用折中；只在节点施加约束会导致节点间"clipping"。
- **量级参考**：Szmuk 2018 用 50 个节点、6–9 次迭代收敛（无量纲算例，未报耗时）；Szmuk 等 2018b 用 20 个节点、4–7 次迭代、最长 0.682 s。
- **开源实现**：

| 实现 | 语言 | 内容 |
|------|------|------|
| [SCPToolbox.jl](../../sources/repos/scptoolbox_jl.md) | Julia | PTR / SCvx / GuSTO / LCvx，含火星着陆与 Starship 翻身着陆示例 |
| [EmbersArc/SCvx](../../sources/repos/scvx_embersarc.md) | Python + CVXPY | 固定 / 自由终端时间 SCvx，6-DoF 与 2D 火箭模型 |
| PythonRobotics `rocket_powered_landing` | Python + CVXPY | 单文件 6-DoF SCvx 示例（见 [PythonRobotics 归档](../../sources/repos/python_robotics.md)） |

## 局限与风险

- **一般没有全局最优保证。** SCvx 保证的是收敛到驻点；Szmuk 等 2018b 明说 "no convergence guarantees are presented"（针对其改进形式）。
- **对缩放、初值、罚权敏感。** 见上文实践建议。
- **论文算例多为无量纲。** Szmuk 2018 的参数 "not intended to match a real-world system"，不能直接当工程指标。
- **实时数字依赖硬件。** 0.682 s 是 2014 年笔记本 + MATLAB 的结果，且不含约 10 ms 的传播步；上飞控需重测。

## 关联页面

- [动力下降制导（Powered Descent Guidance）](../concepts/powered-descent-guidance.md)
- [无损凸化（Lossless Convexification）](lossless-convexification.md)
- [轨迹优化（Trajectory Optimization）](trajectory-optimization.md)
- [LQR / iLQR](lqr-ilqr.md) — 另一类"线性化 + 迭代"的轨迹优化
- [单位四元数与 SO(3)](../formalizations/unit-quaternion-so3.md) — 6-DoF 姿态表示

## 参考来源

- [Szmuk & Açıkmeşe, 6-DoF SCvx Free-Final-Time（2018）](../../sources/papers/szmuk_2018_scvx_6dof_free_final_time.md)
- [Szmuk, Reynolds, Açıkmeşe, Real-Time 6-DoF PDG with State-Triggered Constraints（2018）](../../sources/papers/szmuk_2018_scvx_realtime_state_triggered.md)
- [Malyuta 等, Convex Optimization for Trajectory Generation](../../sources/papers/malyuta_2021_convex_trajectory_generation_tutorial.md)
- [SCPToolbox.jl](../../sources/repos/scptoolbox_jl.md)
- [EmbersArc/SCvx](../../sources/repos/scvx_embersarc.md)

## 推荐继续阅读

- Malyuta 等教程 Part II：<https://arxiv.org/abs/2106.09125>
- Szmuk & Açıkmeşe 2018：<https://arxiv.org/abs/1802.03827>
- ECOS（嵌入式 SOCP 求解器）：<https://github.com/embotech/ecos>
