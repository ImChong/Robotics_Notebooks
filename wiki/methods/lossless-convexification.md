---
type: method
tags: [optimization, optimal-control, trajectory-optimization, convex-relaxation, rocket-landing]
status: complete
updated: 2026-11-06
related:
  - ../concepts/powered-descent-guidance.md
  - ./successive-convexification.md
  - ./convex-relaxation-robotics.md
  - ../formalizations/convex-functions.md
  - ../concepts/optimal-control.md
sources:
  - ../../sources/papers/blackmore_2010_minimum_landing_error_pdg.md
  - ../../sources/papers/malyuta_2021_convex_trajectory_generation_tutorial.md
  - ../../sources/sites/jpl_gfold_xombie_flight_tests.md
original:
  - ../../sources/papers/blackmore_2010_minimum_landing_error_pdg.md
summary: "无损凸化（LCvx）：用松弛变量把火箭着陆中'推力不能低于下限'这类非凸控制约束放松成二阶锥约束，并用最大值原理证明松弛问题的最优解就是原问题的最优解，从而能用内点法实时、全局地求解。"
---

# 无损凸化（Lossless Convexification, LCvx）

**无损凸化**：把一个非凸最优控制问题放松成凸问题，再证明"松弛不会带来损失"——凸问题的最优解恰好满足原来的非凸约束，因此也是原问题的全局最优解。

## 一句话定义

> 推力下界让可行集变成"空心球壳"，LCvx 用一个松弛量把它填成实心锥，再证明最优解一定落在锥面上，所以放松了等于没放松。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LCvx | Lossless Convexification | 无损凸化，本页主题 |
| SOCP | Second-Order Cone Program | 二阶锥规划，松弛后的问题类型 |
| PMP | Pontryagin's Maximum Principle | 最大值原理，证明"无损"的工具 |
| IPM | Interior Point Method | 内点法，求解 SOCP 的标准算法 |
| FOH | First-Order Hold | 一阶保持，控制在相邻节点间线性插值的离散化 |
| G-FOLD | Fuel Optimal Large Divert Guidance | JPL 基于该方法的着陆制导算法 |

## 为什么重要

- **把"能不能实时求出全局最优"变成"能"。** Blackmore 等 2010 的论点：SOCP 能用内点法在多项式时间求到全局最优，内点法有确定的停止准则，"guaranteed to find a feasible solution if one exists"；而一般非线性规划 "cannot provide a priori guarantees on how many iterations"（[原文归档](../../sources/papers/blackmore_2010_minimum_landing_error_pdg.md)）。
- **有飞行验证。** JPL 的 G-FOLD 在 2013 年 Masten Xombie 飞行中实时规划约 0.8 km 的三维改道（[JPL 官方页面](../../sources/sites/jpl_gfold_xombie_flight_tests.md)）。

## 核心原理

### 1. 非凸来源

3-DoF 着陆问题里，推力约束 \(0<\rho_1\le\|T_c\|\le\rho_2\) 的下界是非凸的——"once started, the thrusters cannot be throttled below a certain level"。

### 2. 松弛

引入松弛量 \(\Gamma\)（Blackmore 2010 Problem 2）：

\[
\|T_c\|\le\Gamma,\qquad \rho_1\le\Gamma\le\rho_2,\qquad \dot m=-\alpha\Gamma,\qquad \min\int_0^{t_f}\Gamma\,dt
\]

\((T_c,\Gamma)\) 的可行集是凸的。

### 3. 变量替换让动力学变线性

令 \(\sigma=\Gamma/m,\ u=T_c/m,\ z=\ln m\)，得到 \(\ddot r=u+g\)、\(\dot z=-\alpha\sigma\)（线性），约束变成 \(\|u\|\le\sigma\) 与 \(\rho_1e^{-z}\le\sigma\le\rho_2e^{-z}\)。指数项用围绕参考质量曲线 \(z_0(t)=\ln(m_{wet}-\alpha\rho_2t)\) 的二阶锥近似处理。

### 4. 为什么无损

用最大值原理分析松弛问题的哈密顿量，证明最优解处 \(\|T_c\|=\Gamma\)（Malyuta 等称之为证明 "LCvx equality constraint" 取等号），所以松弛问题的最优推力自动满足 \(\|T_c\|\ge\rho_1\)。Malyuta 等给出的一般条件是完全能控 + 一个横截性相关的线性无关条件。

### 5. 离散化与飞行时间

等间隔节点、控制按一阶保持（Blackmore 2010 用分段线性基函数，即 FOH）得到有限维 SOCP；飞行时间 \(t_f\) 在外层用黄金分割搜索，单峰性只做了数值验证。

## 主要技术路线

| 路线 | 处理的非凸性 | 依据 |
|------|-------------|------|
| 推力上下界凸化 | \(\rho_1\le\|T\|\le\rho_2\) 的下界 | Açıkmeşe & Ploen 2007；Blackmore 2010 Problem 2 |
| 最小着陆误差两步法 | 目标不可达时最省燃料问题无解 | Blackmore 2010 Algorithm 1 |
| 带指向约束的 LCvx | 推力方向锥约束 | Malyuta 等 Part III 火星着陆算例（指向角 40°）；理论见 Açıkmeşe, Carson, Blackmore 2013（本次未读原文） |
| LCvx + SCP 组合 | LCvx 管控制约束，其余非凸交给序列凸化 | Malyuta 等："When LCvx cannot be used, convex optimization can be applied via sequential convex programming" |

## 工程实践

| 步骤 | 做法 | 来源 |
|------|------|------|
| 建模 | 3-DoF 质点 + 质量消耗 + 推力上下界 + 滑翔坡 + 干质量 | Blackmore 2010 §II |
| 凸化 | 松弛量 \(\Gamma\) + 变量替换 + 指数项二阶锥近似 | Blackmore 2010 式 11–49 |
| 求解 | 离散成 SOCP 交给内点法（论文用 JPL 自研 G-OPT；CVXPY + ECOS/Clarabel 可复现） | Blackmore 2010 §V |
| 外层 | 黄金分割搜索 \(t_f\)；目标不可达时先最小化着陆误差再最省燃料 | Blackmore 2010 §IV.E、Algorithm 1 |
| 数值量级 | 55 个节点，可达情形用燃料 399.4 kg、\(t_f\)=78.4 s、笔记本上总计 14.3 s（含 23 次黄金分割） | Blackmore 2010 §V |

练手建议（推测，非论文内容）：先用 CVXPY 复现 Blackmore 2010 §V 的参数，再故意把初速度调到目标不可达，观察两步法的行为。

## 局限与风险

- **适用面窄。** Malyuta 等："LCvx tends to apply only to very specific problems"；状态约束须是凸的，一般凸状态约束只在 "active pointwise in time" 时有保证。
- **定理有边界条件。** Blackmore 2010 的定理 1 要求轨迹至多在一点触及状态约束边界；附录更正了 2007 原文引理只在轨迹留在可行域内部时才成立。
- **只管平移。** 3-DoF 模型要求姿态内环足够快；6-DoF、气动、姿态约束需转向 [序列凸化](successive-convexification.md)。
- **工业应用多为推测。** "Falcon 9 末段算法与 LCvx 是近亲"是 Malyuta 等的推测；公开可确认的是 SpaceX 用 CVXGEN 生成机载凸优化代码（Blackmore 2016）。

## 关联页面

- [动力下降制导（Powered Descent Guidance）](../concepts/powered-descent-guidance.md)
- [序列凸化（Successive Convexification, SCvx）](successive-convexification.md)
- [机器人凸松弛](convex-relaxation-robotics.md) — 一般意义上的凸松弛（多数情况下不是无损的）
- [凸函数（Convex Functions）](../formalizations/convex-functions.md)
- [最优控制（Optimal Control）](../concepts/optimal-control.md)

## 参考来源

- [Blackmore, Açıkmeşe, Scharf, Minimum-Landing-Error PDG（JGCD 2010）](../../sources/papers/blackmore_2010_minimum_landing_error_pdg.md)
- [Malyuta 等, Convex Optimization for Trajectory Generation](../../sources/papers/malyuta_2021_convex_trajectory_generation_tutorial.md)
- [JPL G-FOLD × Masten Xombie 官方页面](../../sources/sites/jpl_gfold_xombie_flight_tests.md)

## 推荐继续阅读

- Açıkmeşe & Ploen, *Convex Programming Approach to Powered Descent Guidance for Mars Landing*, JGCD 30(5), 2007, doi:10.2514/1.27553（原始论文，付费）
- Malyuta 等教程 Part I：<https://arxiv.org/abs/2106.09125>
- CVXPY：<https://www.cvxpy.org/>
