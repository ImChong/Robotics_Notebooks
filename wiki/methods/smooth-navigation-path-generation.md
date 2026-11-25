---
type: method
tags: [motion-planning, navigation, optimization, path-smoothing, mobile-robot]
status: complete
updated: 2026-11-21
related:
  - ./trajectory-optimization.md
  - ../concepts/collision-distance-optimization.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ../comparisons/mobile-robot-navigation-planning-methods.md
  - ../entities/ego-planner-swarm.md
sources:
  - ../../sources/courses/numerical_optimization_foundations_robotics.md
  - ../../sources/papers/ego_planner_arxiv_2008_08835.md
original:
  - ../../sources/papers/ego_planner_arxiv_2008_08835.md
summary: "平滑导航路径生成：用无约束/软约束优化生成曲率连续、可跟踪的 2D/3D 导航路径，结合 ESDF-Free 样条 L-BFGS（EGO-Planner）实现亚毫秒在线规划。"
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

## 为什么重要

- **几何折线不可直跟：** A* / RRT 输出的分段折线在拐角处曲率跳变且不满足差速车/阿克曼底盘或四旋翼动力学（加速度/加加速度有界），直接执行会引发剧烈抖振或跟踪失稳。
- **在线重规划的计算瓶颈：** 在机载实时避障中，若每帧都维护全空间欧氏符号距离场（ESDF）往往吞噬 70% 以上 CPU 预算；而像 [EGO-Planner](../entities/ego-planner-swarm.md)（arXiv:2008.08835）通过 ESDF-Free 局部按需排斥场把单次规划压到 **0.81 ms**（Fast-Planner 需 7.30 ms），是高机动实机落地的关键跃迁。

## 核心机制与算法选型

典型平滑路径生成将几何初值路径（折线或拓扑路径）参数化为 3 阶均匀 **B 样条** 或离散弹性带顶点 $Q$，采用无约束 **L-BFGS** 求解光顺、防撞与动力学可行目标函数 $J = \lambda_s J_s + \lambda_c J_c + \lambda_d J_d$；其中 ESDF-Free 方案仅对进入障碍物的局部控制点计算最近表面锚点与逃逸法向 $\{p_{ij}, v_{ij}\}$。

```mermaid
flowchart LR
    front["前端折线 / 无碰撞拓扑引导路径"] --> param["均匀 B 样条控制点参数化"]
    param --> grad{"避障梯度构造"}
    grad -->|预建 ESDF| esdf["全局距离场插值查询"]
    grad -->|ESDF-Free 按需| pv["仅对穿透点算最近表面 {p, v}"]
    esdf --> opt["无约束 L-BFGS / SQP 优化<br/>光滑 + 防撞 + 动力学罚项"]
    pv --> opt
    opt -->|超速/超加速度| time["各向异性曲线拟合时间重分配"]
    time --> ctrl["输出 C2/C3 连续可执行参考轨迹"]
```

**决策变量**：样条控制点、Elastic band 顶点或 $SE(2)$/ $SE(3)$ 序列。

**代价**：
$$\min \sum_k \big( w_j \|\Delta^3 p_k\|^2 + w_c \kappa(p_k)^2 + w_l \|p_k - p_k^{\text{ref}}\|^2 \big)$$

**约束/惩罚**：
- 障碍：$d(p_k) \ge d_{\min}$ 或罚项（见 [Collision Distance](../concepts/collision-distance-optimization.md)）
- 曲率：$|\kappa| \le \kappa_{\max}$（软/硬）

## 主要技术路线

| 规模与需求 | 优化器 / 路线 |
|------|------|
| 低维、无硬约束平滑 | [Line Search / BFGS](./quasi-newton-bfgs.md) |
| 含障碍软惩罚（UAV/AMR 实时重规划） | 均匀 B 样条 + **L-BFGS**（ESDF-Free `{p, v}`，0.81 ms 级） |
| 严格硬边界走廊约束 | SQP / QP / NMPC 跟踪 |

## 工程实践

- **时间重分配防偏出走廊：** 若 L-BFGS 软约束未能完全压住速度/加速度峰值，执行时间拉伸时必须采用类似 EGO-Planner 的 **各向异性曲线拟合**（重权惩罚法向偏差、放松切向速度），防止拉伸时间后轨迹在空间拐角处拉直撞墙。
- **下游跟踪协同：** 地面机器人在 Nav2 中通常配合 [DWA](./dwa.md) / [MPPI](./mppi.md) 局部控制器；无人机将平滑后的平坦输出（位置/偏航及其高阶导）直供几何控制器或 MPC。

## 常见误区与局限

- **误区：必须先建稠密 ESDF 才能做梯度优化。** 局部滚动视窗内大多数体素不参与当前轨迹梯度，按需构造引导参考线表面向量 $\{p, v\}$ 可省去 ESDF 构建开销。
- **局限：** 非凸障碍环境下的梯度局部优化只能收敛到初值所属同伦类内的局部最优；若前端拓扑本身绕远或卡死，仅靠平滑优化无法跨越障碍换道。

## 关联页面

- [Navigation & SLAM Stack](../overview/navigation-slam-autonomy-stack.md)
- [Trajectory Optimization](./trajectory-optimization.md)
- [Numerical Optimization Curriculum](../entities/numerical-optimization-curriculum.md)
- [导航规划方法对比：全局·局部·平滑](../comparisons/mobile-robot-navigation-planning-methods.md) — 本页是其路径平滑后处理层的选型落地
- [Project Quiver](../entities/project-quiver.md) — 作业级 ArduPilot 机架；平滑路径仍在 GCS / 伴机侧，不在机架仓内

## 推荐继续阅读

- [Ego-Planner / swarm 规划](../entities/ego-planner-swarm.md)（若做集群）
- Boyd 凸优化 — 样条拟合作 QP

## 参考来源

- [sources/papers/ego_planner_arxiv_2008_08835.md](../../sources/papers/ego_planner_arxiv_2008_08835.md) — Zhou et al., *EGO-Planner: An ESDF-Free Gradient-Based Local Planner for Quadrotors*, IEEE RA-L / arXiv:2008.08835
- [sources/courses/numerical_optimization_foundations_robotics.md](../../sources/courses/numerical_optimization_foundations_robotics.md) — 第 2 章 2.4–2.5 平滑导航路径
