---
type: formalization
tags: [differential-geometry, manifold, optimization, rl, motion-control, so3, se3, embodied-ai, shenlan]
status: complete
updated: 2026-06-04
related:
  - ../overview/shenlan-embodied-ai-fundamentals-series.md
  - ./lie-group-rigid-body-motions.md
  - ./3d-coordinate-transforms-vision-robotics.md
  - ./se3-representation.md
  - ../methods/model-predictive-control.md
  - ../methods/reinforcement-learning.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
sources:
  - ../../sources/blogs/wechat_shenlan_riemannian_manifold_tangent_space.md
  - ../../sources/raw/wechat_shenlan_riemannian_manifold_2026-06-04.md
  - ../../sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md
summary: "机器人合法状态生活在黎曼流形上（周期关节、SO(3)/SE(3) 姿态）；算法在切空间做线性增量，经 Exp/Log 映回流形。对比欧式补丁思维与流形原生计算，并列出局部线性、离散 Exp/Log、固定度量、欧式梯度等工程近似及误差边界。"
---

# 黎曼流形与切空间（具身运动的几何语言）

**一句话：** 具身体的状态空间 **整体弯曲、局部平直**；在流形上存合法姿态，在 **切空间** 做梯度与增量，用 **指数/对数映射** 往返——[李群页](./lie-group-rigid-body-motions.md) 中的 SO(3)/SE(3) 是这一框架最重要的机器人特例。

## 为什么欧式空间会「算错物理」

| 现象 | 欧式做法的问题 | 流形视角 |
|------|----------------|----------|
| 关节 $350° \to 10°$ | 差 $340°$ | 周期流形上最短 **20°** |
| 旋转 $360°$ 等价 $0°$ | 判为远距 | SO(3) 上同一点 |
| 9 维旋转矩阵自由回归 | 失去正交性 | 约束曲面，3 自由度 |

**最短路径是测地线（曲线）** ⇒ 空间本身弯曲，不能全局用直线插值。

文内对比：**人形** 以转动为主（「化圆为方」）vs **自动驾驶** 轨迹常折线再平滑（「化方为圆」）——弯曲建模需求不同。

## 黎曼流形（状态空间）

**定义（工程口径）：** 光滑流形 $\mathcal{M}$，每点 $x$ 的切空间 $T_x\mathcal{M}$ 上配备光滑、对称、正定的 **黎曼度量** $g_x$，用于测量切向量长度、夹角与测地距离。

| 机器人对象 | 典型流形 |
|------------|----------|
| 单关节角 | 圆 $S^1$ |
| 多关节 | 环面 $T^n$ |
| 三维旋转 | SO(3) |
| 刚体位姿 | SE(3) |

**弯曲来源（文内）：**

- **周期闭环**：$0°=360°$ 须黏合端点
- **正交约束**：SO(3) 矩阵元耦合，9→3 维
- **最优运动贴曲面**：测地线而非欧式弦

**地球球面类比：** 局部地面近似平面（切空间），全球最短路径为大圆。

## 切空间（计算空间）

对 $x \in \mathcal{M}$，$T_x\mathcal{M}$ 是与流形在该点 **相切** 的线性空间，$\dim T_x\mathcal{M} = \dim \mathcal{M}$。

| 角色 | 流形 $\mathcal{M}$ | 切空间 $T_x\mathcal{M}$ |
|------|---------------------|-------------------------|
| 存什么 | 当前绝对状态（$q$, $T$） | 增量、速度、梯度方向 |
| 运算 | 组合、约束天然满足 | 加减、内积、Adam 步 |

## Exp / Log：双向映射

```mermaid
flowchart LR
  TX["切空间增量 v ∈ T_x M"]
  MY["流形状态 y ∈ M"]
  TX -->|"Exp_x(v)"| MY
  MY -->|"Log_x(y)"| TX
```

- **指数映射** $\exp_x: T_x\mathcal{M} \to \mathcal{M}$：线性增量 → 合法新状态（沿测地线一步的抽象）
- **对数映射** $\log_x(y)$：状态差 → 切向量；测地距离 $\approx \|\log_x(y)\|_{g_x}$

与 SO(3) 的 Rodrigues、SE(3) 的 twist 指数映射一致；复数单位圆上 $e^{i\theta}$ 为最简单特例。

**RL 更新模板（文内）：**

$$
x_{k+1} = \exp_{x_k}\bigl(-\eta\, \mathrm{grad}\, f(x_k)\bigr)
$$

在切空间算梯度，再映回流形，避免「更新后四元数不归一化」类补丁。

## 工程近似（须知情）

| 近似 | 做法 | 小增量 | 大增量 |
|------|------|--------|--------|
| 切空间线性化 | $\exp_x(v) \approx x \oplus v$ | 几乎精确 | 偏离测地线 |
| Exp 一阶泰勒 | 离散控制常用 | 稳定 | 轨迹失真 |
| Log 差分 | $\log_x(y) \approx y-x$ | 姿态误差可用 | 大角失效 |
| 固定度量 | $g_x \approx \text{const}$ | 短程距离 | 长程累积 |
| 欧式梯度 | 忽略 $g_x$ 修正 | 快速训练 | 下降方向偏 |
| 线性插值 | 切空间弦 | 微调 | 应用 SLERP / 测地线 |

**严格精确** 的通常是定义与连续 Exp/Log；**深度学习离散步** 几乎总在近似 regime。

## 欧式补丁 vs 流形原生

| 思维 | 做法 | 后果 |
|------|------|------|
| **欧式补丁** | $\mathbb{R}^n$ 更新后再投影/归一化 | 大步长非物理、优化不稳定 |
| **流形原生** | 合法性内建于 $\mathcal{M}$ | 问题在「错误空间」里消失 |

## 在具身栈中的落点

| 模块 | 流形 | 切空间 |
|------|------|--------|
| **运动控制** | 关节/姿态轨迹 | 速度、QP 线性化 |
| **RL / 策略** | 动作含旋转 | 策略梯度、PPO 增量 |
| **世界模型** | 低维隐状态流形 | 预测沿测地线演化（文内主张） |
| **SLAM / 融合** | 位姿图节点 | 相对误差在 se(3) |

与 [MPC](../methods/model-predictive-control.md) 在 se(3) 切空间线性化、[世界模型 taxonomy](../overview/robot-world-models-training-loop-taxonomy.md) 互补。

## 常见误区

1. **「流形 = 高深无用」** — 大角度关节/姿态 RL 中，欧式插值会直接产生 **非法与冗余运动**。
2. **「有李群就不需要黎曼」** — 李群页是 **SO(3)/SE(3) 手册**；本页是 **一般框架 + 近似清单**。
3. **「Exp/Log 永远精确」** — 工程实现多为小增量近似；须监控步长。
4. **「流形只服务旋转」** — 周期关节、联合动作空间同样适用。

## 关联页面

- [《具身智能基础》专栏地图](../overview/shenlan-embodied-ai-fundamentals-series.md)
- [李群、李代数与刚体旋转](./lie-group-rigid-body-motions.md)
- [三维坐标变换](./3d-coordinate-transforms-vision-robotics.md)
- [SE(3) Representation](./se3-representation.md)
- [Model Predictive Control](../methods/model-predictive-control.md)

## 参考来源

- [深蓝具身智能：黎曼流形与切空间](../../sources/blogs/wechat_shenlan_riemannian_manifold_tangent_space.md)
- [深蓝具身智能：李群、李代数、四元数](../../sources/blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md)
- [抓取落盘摘要](../../sources/raw/wechat_shenlan_riemannian_manifold_2026-06-04.md)

## 推荐继续阅读

- Bullo & Lewis, *Geometric Control of Mechanical Systems*（外部教材）
- Lynch & Park, *Modern Robotics* Ch 3 — [modern_robotics_textbook.md](../../sources/papers/modern_robotics_textbook.md)
