# MIGHTY: Hermite Spline-based Efficient Trajectory Planning

> 来源归档

- **标题：** MIGHTY: Hermite Spline-based Efficient Trajectory Planning
- **类型：** paper
- **出处：** 2026 · IEEE Robotics and Automation Letters (RA-L) · arXiv preprint
- **论文链接：** <https://arxiv.org/abs/2511.10822>
- **代码/项目：** <https://github.com/mit-acl/mighty>
- **演示视频：** <https://youtu.be/Pvb-VPUdLvg>
- **入库日期：** 2026-06-14
- **一句话说明：** MIT ACL 提出的 **五次 Hermite 样条** 软约束 UAV 轨迹规划器：在 **单次非线性优化** 中联合优化空间路径与段时间分配，相对 EGO-Planner / MINCO 等基线在仿真中 **计算时间 −9.3%、飞行时间 −13.1%**，真机 cluttered 环境 **最高 6.7 m/s** 与动态障碍长航时验证。

---

## 核心摘录（策展，非全文）

### 问题与动机

- **硬约束规划器**（显式安全约束 + 商业求解器）算力开销大，难以支撑高频重规划。
- **软约束规划器**（EGO-Planner、RAPTOR、MINCO/SUPER 等）更快，但常见两类局限：
  1. **空间与时间解耦**（先优化几何再调时间，或固定 knot span）→ 次优轨迹；
  2. **搜索空间受限**（B-spline 控制点间接约束高阶导数；MINCO 类 waypoint–duration 参数化全局耦合、缺乏 knot 级高阶动力学局部控制）。

### 方法要点

| 维度 | MIGHTY |
|------|--------|
| **表示** | 五次（quintic）**Hermite spline**：每 knot 显式优化 **位置 / 速度 / 加速度** + 各段 **时长** $T_s$ |
| **时空优化** | **联合** spatiotemporal，单次 unconstrained NLP（L-BFGS 类梯度法） |
| **连续性** | Hermite 结构保证 $\mathcal{C}^2$ 连续，无 MINCO 式全局系数耦合 |
| **代价评估** | 优化变量在 Hermite 空间；**碰撞/平滑等代价在 Bézier 基下高效采样**（闭式梯度链式回传） |
| **约束处理** | 软惩罚：ESDF 碰撞势、动力学超限惩罚；时长经 diffeomorphism 保证 $T_s>0$ |
| **微分平坦** | 面向四旋翼 flat output 轨迹；与 PX4 Offboard 设定点栈衔接 |

### 与 SOTA 对照（论文 Table I 摘要）

| 方法 | 表示 | 时间优化 | 搜索空间 | 局部控制 |
|------|------|----------|----------|----------|
| Polynomial [Richter 2016] | 9 阶多项式 | 解耦 | 受限 | 全局端点导数 |
| EGO-Planner [2021] | 均匀 B-spline | 固定 knot | 受限 | 几何局部、导数间接 |
| MINCO / SUPER [2022–2025] | MINCO | 联合 | MINCO 子空间 | 无几何局部控制 |
| **MIGHTY** | Hermite spline | **联合** | **完整多项式+时间空间** | **位置与高阶动力学直接局部** |

### 实验摘要

- **仿真（静态复杂场景）**：相对最强基线 **计算时间 −9.3%**、**飞行时间 −13.1%**，**成功率 100%**；动态障碍场景验证安全行为。
- **真机**：LiDAR 感知定位；静态 clutter **最高 6.7 m/s**；长航时飞行与 **在线新增动态障碍** 避障。

### 对 wiki 的映射

- [paper-mighty-hermite-spline-trajectory-planning](../../wiki/entities/paper-mighty-hermite-spline-trajectory-planning.md)
- [multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)
- [ego-planner-swarm](../../wiki/entities/ego-planner-swarm.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2511.10822>
- 代码：<https://github.com/mit-acl/mighty>
- 视频：<https://youtu.be/Pvb-VPUdLvg>
