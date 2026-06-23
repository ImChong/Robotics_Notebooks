---
type: method
tags: [motion-planning, navigation, optimization, path-smoothing, mobile-robot]
status: complete
updated: 2026-06-23
related:
  - ./trajectory-optimization.md
  - ../concepts/collision-distance-optimization.md
  - ../overview/navigation-slam-autonomy-stack.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
summary: "平滑导航路径生成：用无约束/软约束优化生成曲率连续、可跟踪的 2D/3D 导航路径，是移动机器人与无人机规划常见模块。"
---

# Smooth Navigation Path Generation（平滑导航路径生成）

**平滑导航路径生成**：在已知或采样的几何路径基础上，通过 **无约束或软约束优化** 最小化 jerk/曲率/能量，得到 **曲率连续、可跟踪** 的导航轨迹——课程 2.4–2.5 的无约束优化应用入口。

## 一句话定义

> A* / RRT 给出折线，平滑优化把它变成车/无人机真正能跟的曲线。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| NLP | Nonlinear Programming | 控制点/样条系数优化 |
| SDF | Signed Distance Field | 障碍距离软惩罚 |
| BFGS | Broyden–Fletcher–Goldfarb–Shanno | 平滑代价常用求解器 |
| MPC | Model Predictive Control | 跟踪平滑后的参考路径 |
| SLAM | Simultaneous Localization and Mapping | 导航栈上游定位建图 |

## 典型建模

**决策变量**：样条控制点、Elastic band 顶点或 $SE(2)$/ $SE(3)$ 序列。

**代价**：
$$\min \sum_k \big( w_j \|\Delta^3 p_k\|^2 + w_c \kappa(p_k)^2 + w_l \|p_k - p_k^{\text{ref}}\|^2 \big)$$

**约束/惩罚**：
- 障碍：$d(p_k) \ge d_{\min}$ 或罚项（见 [Collision Distance](../concepts/collision-distance-optimization.md)）
- 曲率：$|\kappa| \le \kappa_{\max}$（软/硬）

## 算法选型

| 规模 | 方法 |
|------|------|
| 低维、无硬约束 | [Line Search / BFGS](./quasi-newton-bfgs.md) |
| 含障碍软惩罚 | L-BFGS + 线搜索 |
| 硬约束 | SQP / 内点 / NMPC 跟踪 |

## 主要分类

| 建模 | 优化器 |
|------|--------|
| 样条控制点 + jerk 代价 | BFGS / L-BFGS |
| Elastic band 顶点 | 梯度下降 |
| 硬障碍约束 | SQP / NMPC 跟踪 |

## 与其他页面的关系

- [Navigation & SLAM Stack](../overview/navigation-slam-autonomy-stack.md)
- [Trajectory Optimization](./trajectory-optimization.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)

## 推荐继续阅读

- [Ego-Planner / swarm 规划](../entities/ego-planner-swarm.md)（若做集群）
- Boyd 凸优化 — 样条拟合作 QP

## 参考来源

- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 2 章 2.4–2.5 平滑导航路径
