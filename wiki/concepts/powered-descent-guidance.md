---
type: concept
tags: [optimal-control, trajectory-optimization, guidance, rocket-landing, aerospace]
status: complete
updated: 2026-11-06
related:
  - ../methods/lossless-convexification.md
  - ../methods/successive-convexification.md
  - ../concepts/optimal-control.md
  - ../methods/trajectory-optimization.md
  - ../methods/model-predictive-control.md
sources:
  - ../../sources/papers/klumpp_1974_apollo_lunar_descent_guidance.md
  - ../../sources/papers/blackmore_2010_minimum_landing_error_pdg.md
  - ../../sources/papers/blackmore_autonomous_precision_landing_space_rockets.md
  - ../../sources/papers/szmuk_2018_scvx_6dof_free_final_time.md
  - ../../sources/papers/malyuta_2021_convex_trajectory_generation_tutorial.md
  - ../../sources/papers/gaudet_2018_drl_6dof_powered_descent.md
  - ../../sources/sites/jpl_gfold_xombie_flight_tests.md
  - ../../sources/sites/spacex_reusable_booster_pages.md
  - ../../sources/sites/nasa_splice_new_shepard.md
original:
  - ../../sources/papers/blackmore_2010_minimum_landing_error_pdg.md
summary: "动力下降制导：火箭只靠发动机推力，从下降段某一状态出发，在推力上下界、滑翔坡、姿态等约束下飞到着陆点并以零速触地的在线轨迹规划问题；火箭回收、行星着陆的核心。"
---

# 动力下降制导（Powered Descent Guidance, PDG）

**动力下降制导**：火箭在着陆前的最后一段只靠发动机推力减速和横移，制导算法要实时算出一条推力剖面，使火箭在高度归零的同时速度也归零，且全程满足硬件和安全约束。

## 一句话定义

> 给定当前位置、速度、质量，求一条满足推力上下界、滑翔坡、姿态和燃料约束的推力剖面，让火箭"到地面那一刻刚好停住"，并能在几百毫秒到几秒内重算。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| PDG | Powered Descent Guidance | 动力下降制导，本页主题 |
| GNC | Guidance, Navigation and Control | 制导（算轨迹）、导航（估状态）、控制（跟轨迹）三件套 |
| SOCP | Second-Order Cone Program | 二阶锥规划，凸优化制导的标准形式 |
| LCvx | Lossless Convexification | 无损凸化，把非凸推力约束等价松弛成凸约束 |
| SCvx | Successive Convexification | 序列凸化，迭代求解 6-DoF 非凸着陆问题 |
| G-FOLD | Fuel Optimal Large Divert Guidance | JPL 的燃料最优大范围改道制导算法 |
| TRN | Terrain Relative Navigation | 地形相对导航，用相机与地图比对定位 |
| DoF | Degrees of Freedom | 3-DoF 只管质心平移，6-DoF 同时管姿态 |

## 为什么重要

- **可重复使用火箭的最后一公里。** Falcon 9 官方页面称它是"world's first orbital class reusable rocket"，可复用的前提是一级能自主精确着陆（[SpaceX 官方页面](../../sources/sites/spacex_reusable_booster_pages.md)）。
- **约束硬、时间短、算错就坠毁。** Blackmore 2016 指出，速度若在高度归零前降到零，火箭会重新上升；减速太晚则撞地；着陆点火还要留出横向改道能力，并需在极短时间内重规划（[Blackmore 2016](../../sources/papers/blackmore_autonomous_precision_landing_space_rockets.md)）。
- **和机器人运动控制同一套数学。** 它本质上是带约束的最优控制 / 轨迹优化，"离线保证 + 在线重规划"的工程套路与腿足 MPC 一脉相承（见 [最优控制](optimal-control.md)、[轨迹优化](../methods/trajectory-optimization.md)）。

## 核心原理

### 1. 3-DoF 标准问题

Blackmore 等 2010 的写法（[原文归档](../../sources/papers/blackmore_2010_minimum_landing_error_pdg.md)）：

\[
\ddot r = g + \frac{T_c}{m},\qquad \dot m = -\alpha\|T_c\|,\qquad 0<\rho_1\le\|T_c\|\le\rho_2
\]

外加滑翔坡（着陆点上方的锥）、\(m(t_f)\ge m_{dry}\)、终端 \(r(t_f)=0,\ \dot r(t_f)=0\)，目标是最省燃料或最小着陆误差，\(t_f\) 也是优化变量。难点在推力**下界**：发动机一旦点火就不能节流到某个水平以下，\(\|T_c\|\ge\rho_1\) 是非凸集合。

### 2. 三代解法

| 年代 | 代表 | 怎么解 | 局限 |
|------|------|--------|------|
| 1969 起 | Apollo 登月舱下降制导（[Klumpp](../../sources/papers/klumpp_1974_apollo_lunar_descent_guidance.md)） | 地面预算制导目标，机载程序按显式/隐式制导方程算推力与姿态；航天员可改选着陆点 | Blackmore 2010 把这类方法归为"忽略约束、求闭式解" |
| 2007–2013 | 无损凸化 + G-FOLD | 引入松弛变量把非凸推力约束变成 SOCP，可证明松弛后最优解仍满足原约束；2013 年在 Masten Xombie 上实时改道飞行 | 只适用于 3-DoF 和特定约束结构（见 [无损凸化](../methods/lossless-convexification.md)） |
| 2016 起 | 序列凸化 SCvx / PTR | 6-DoF 非线性动力学在上一轮轨迹附近线性化，迭代求解 SOCP，配虚拟控制与信赖域 | 一般无全局最优保证，需要缩放与调参（见 [序列凸化](../methods/successive-convexification.md)） |

学习型方法（PPO 直接输出推力、或监督学习拟合最优轨迹）是并行的一支，推理快但缺少约束与稳定性保证（[Gaudet 等](../../sources/papers/gaudet_2018_drl_6dof_powered_descent.md)、[Sánchez-Sánchez & Izzo](../../sources/papers/sanchez_izzo_2016_dnn_optimal_landing.md)）。

### 3. 助推器全剖面回收

公开资料里的 Falcon 9 / Super Heavy 回收顺序是：分离 → 返场点火（boostback）→ 大气内用栅格舵调姿（Falcon 9 栅格舵 "orient the rocket during reentry by moving the center of pressure"）→ 着陆点火 → 落到着陆腿上或被发射塔"筷子"臂接住（Starship 第五次试飞，2024-10-13）。动力下降制导负责最后的着陆点火；前几段压缩误差，留给着陆点火的改道量越小越好（[Blackmore 2016](../../sources/papers/blackmore_autonomous_precision_landing_space_rockets.md)、[SpaceX 官方页面](../../sources/sites/spacex_reusable_booster_pages.md)）。

## 工程实践

- **制导—控制分层。** 3-DoF 制导假设姿态由更快的内环控制器跟上；Malyuta 等明确写出该模型 "accurate as long as attitude can be controlled in an inner loop faster than the outer translation control loop"。6-DoF 制导则把姿态、万向节角和角速度约束直接放进优化。
- **求解器是飞控软件的一部分。** SpaceX 用 CVXGEN 生成定制机载凸优化代码（Blackmore 2016）；SCvx 论文用 ECOS 等内点法求子问题。内点法求 SOCP 有确定的停止准则，这是凸化路线能上飞控的关键论据（Blackmore 2010）。
- **导航决定能落多准。** NASA SPLICE 在 New Shepard NS-13（2020-10-13）上测试了地形相对导航（IMU + 相机比对地图）和导航多普勒激光雷达（测速度与高度）（[NASA 页面](../../sources/sites/nasa_splice_new_shepard.md)）。
- **目标不可达时要有退路。** Blackmore 2010 的两步法先最小化着陆误差、再在该误差下最省燃料，避免最省燃料问题直接报"不可行"。

## 局限与风险

- **公开的工业细节很少。** SpaceX 的制导算法未公开；"Falcon 9 末段着陆算法与 LCvx 是近亲"是 Malyuta 等的**推测**（原文 "We hypothesize"），不是公开事实。Blackmore 2016 也没有把 Falcon 9 算法称为 G-FOLD。
- **3-DoF 结果不能直接当 6-DoF 用。** 姿态约束、推力指向与气动在 3-DoF 里被忽略或近似。
- **凸化保证有前提。** 无损凸化的定理要求轨迹至多在一点触及状态约束边界等条件；SCvx 一般只保证收敛到驻点。
- **学习型策略缺少可证明性。** Gaudet 等的 RL 着陆在测试中最差横向误差超过了 5 m 目标，作者也承认没有最优性与稳定性保证。

## 关联页面

- [无损凸化（Lossless Convexification）](../methods/lossless-convexification.md)
- [序列凸化（Successive Convexification, SCvx）](../methods/successive-convexification.md)
- [最优控制（Optimal Control）](optimal-control.md)
- [轨迹优化（Trajectory Optimization）](../methods/trajectory-optimization.md)
- [模型预测控制（MPC）](../methods/model-predictive-control.md)
- [机器人凸松弛](../methods/convex-relaxation-robotics.md)
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md)

## 参考来源

- [Klumpp, Apollo Lunar Descent Guidance（1974）](../../sources/papers/klumpp_1974_apollo_lunar_descent_guidance.md)
- [Blackmore 等, Minimum-Landing-Error PDG（JGCD 2010）](../../sources/papers/blackmore_2010_minimum_landing_error_pdg.md)
- [Blackmore, Autonomous Precision Landing of Space Rockets（2016）](../../sources/papers/blackmore_autonomous_precision_landing_space_rockets.md)
- [Szmuk & Açıkmeşe, 6-DoF SCvx（2018）](../../sources/papers/szmuk_2018_scvx_6dof_free_final_time.md)
- [Malyuta 等, Convex Optimization for Trajectory Generation（2021/2022）](../../sources/papers/malyuta_2021_convex_trajectory_generation_tutorial.md)
- [Gaudet 等, DRL 6-DoF Powered Descent（2018）](../../sources/papers/gaudet_2018_drl_6dof_powered_descent.md)
- [JPL G-FOLD × Masten Xombie 官方页面](../../sources/sites/jpl_gfold_xombie_flight_tests.md)
- [SpaceX 官方页面：Falcon 9 / Starship 第五次试飞](../../sources/sites/spacex_reusable_booster_pages.md)
- [NASA SPLICE × New Shepard](../../sources/sites/nasa_splice_new_shepard.md)

## 推荐继续阅读

- Malyuta 等教程：<https://arxiv.org/abs/2106.09125>
- Blackmore, *Autonomous Precision Landing of Space Rockets*：<https://nap.nationalacademies.org/read/23659/chapter/10>
- CMU 16-745 Lecture 22（凸松弛与火箭着陆）：见 [CMU 最优控制课程](../entities/cmu-optimal-control-curriculum.md)
