---
type: entity
tags: [textbook, linear-algebra, education, foundational, kinematics, control]
status: complete
updated: 2026-05-31
related:
  - ../formalizations/se3-representation.md
  - ../formalizations/lie-group-rigid-body-motions.md
  - ../formalizations/lqr.md
  - ./modern-robotics-book.md
  - ./pinocchio.md
  - ../../roadmap/motion-control.md
sources:
  - ../../sources/courses/gatech_interactive_linear_algebra.md
  - ../../sources/courses/axler_linear_algebra_done_right_4e.md
  - ../../sources/courses/linear_algebra_teaching_materials_curated.md
summary: "运动控制 L0 线性代数策展：Georgia Tech 交互教材、Axler LADR4e 与 Strang/3Blue1Brown 等互补入口，按机器人矩阵语言（变换、子空间、最小二乘、谱）组织精读地图。"
---

# 线性代数学习策展（机器人 L0）

**一句话：** 机器人运动控制把位姿、速度、力都写成向量和矩阵；本页把 [Georgia Tech ILA](https://textbooks.math.gatech.edu/ila/)、[Axler *Linear Algebra Done Right* 4e](https://linear.axler.net/LADR4e.pdf) 及常见优秀配套材料，整理成可执行的 **L0 线性代数路线**，并接到 [运动控制成长路线](../../roadmap/motion-control.md) 的数学打底阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LQR | Linear Quadratic Regulator | 线性系统二次型代价下的最优反馈控制器 |
| API | Application Programming Interface | 应用程序编程接口 |
| IK | Inverse Kinematics | 满足末端/姿态约束求解关节角的运动学逆解 |
| iLQR | iterative Linear Quadratic Regulator | 对非线性系统迭代线性化求解的轨迹优化方法 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |

## 为什么重要？

1. **L0 不可跳过**：没有矩阵语言，读不懂 SE(3)、Jacobian、LQR 的 Riccati 递推。
2. **教材互补，不必单押一本**：几何直觉（3Blue1Brown / ILA）+ 线性映射公理（LADR）+ 工程四大子空间（Strang）覆盖机器人常见卡点。
3. **与 [Modern Robotics](./modern-robotics-book.md) 分工明确**：本页补「通用线性代数」；Modern Robotics 从 Ch 3 起用 SE(3) / twist 讲**刚体专用**语言。

## 推荐学习路径（L0，约 2–4 周并行）

```mermaid
flowchart LR
  W0["Week 0–1<br/>3Blue1Brown 几何直觉"]
  W1["Week 1–2<br/>GT ILA：矩阵 / 子空间 / 最小二乘"]
  W2["Week 2–3<br/>LADR 选读：线性映射 / 谱 / 内积"]
  W3["Week 3–4<br/>Modern Robotics Ch 2–3<br/>+ NumPy / Pinocchio 小实验"]
  L1["进入 L1<br/>FK / Jacobian"]

  W0 --> W1 --> W2 --> W3 --> L1
```

| 阶段 | 材料 | 机器人相关产出 |
|------|------|----------------|
| 直觉 | [Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) | 理解线性变换 = 空间变形；为旋转「不能线性插值」埋伏笔 |
| 计算 + 几何 | [Interactive Linear Algebra](https://textbooks.math.gatech.edu/ila/) | 手算 / 交互理解列空间、最小二乘、QR |
| 结构 | [LADR 4e PDF](https://linear.axler.net/LADR4e.pdf) | 把 Jacobian、线性化系统看成线性映射 + 谱 |
| 刚体语言 | [Modern Robotics](./modern-robotics-book.md) Ch 2–3 | SE(3)、矩阵指数、PoE 与 [SE(3) 表示](../formalizations/se3-representation.md) |
| 代码手感 | [Pinocchio](./pinocchio.md) 最小 FK Demo | 矩阵运算落到库 API |

## 章节地图：教材主题 → 本库页面

| 线性代数主题 | 在机器人里对应什么 | 本库延伸阅读 |
|-------------|-------------------|-------------|
| 矩阵乘法、线性变换复合 | 齐次变换链、传感器外参级联 | [SE(3) 表示](../formalizations/se3-representation.md) |
| 正交矩阵、保持长度 | 旋转 \(R\in SO(3)\) | [李群与刚体运动](../formalizations/lie-group-rigid-body-motions.md) |
| 列空间 / 秩 / 零空间 | 冗余臂 IK、约束是否独立 | [Whole-Body Control](../concepts/whole-body-control.md)（任务堆叠直觉） |
| 最小二乘、伪逆、QR | 数值 IK、状态估计、批最小二乘标定 | [Trajectory Optimization](../methods/trajectory-optimization.md) |
| 特征值、对称矩阵、SVD | 线性系统稳定性、LQR、病态雅可比 | [LQR / iLQR](../formalizations/lqr.md) |
| 内积、正交投影 | 任务空间投影、QP 解的几何意义 | [Optimal Control](../concepts/optimal-control.md) |

## 三套主教材怎么选（不必全读完）

| 你的背景 | 建议组合 |
|---------|---------|
| 只会高中矩阵，没几何直觉 | 3Blue1Brown → GT ILA 前半 → Modern Robotics Ch 3 |
| 工科已学过行列式版线代，但读机器人公式吃力 | 跳过 3b1b 或倍速；GT ILA 子空间 + 最小二乘；LADR 第 3–5 章选读 |
| 数学系 / 想补严格性 | LADR 为主；GT ILA 作计算练习；Trefethen & Bau 在要做数值 IK 时补 |

**扩展材料**（Strang 18.06、Immersive Math、数值线代等）见 source 策展页 [linear_algebra_teaching_materials_curated.md](../../sources/courses/linear_algebra_teaching_materials_curated.md)。

## 常见误区

- **误区 1：把旋转矩阵当普通矩阵做加法插值** → 应使用 SO(3) 上的指数映射 / 四元数 slerp（见 L0 自测题与 [SE(3) 表示](../formalizations/se3-representation.md)）。
- **误区 2：只刷题不做机器人代码** → L0 输出应是 NumPy + 一个 FK Demo，不是满分习题册。
- **误区 3：在 L0 深挖张量 / 泛函** → 人形 L4 之前，子空间 + 最小二乘 + 谱 + SE(3) 足够；张量等到读深度学习动作头再补。

## 关联页面

- [运动控制成长路线（L0）](../../roadmap/motion-control.md#l0-数学与编程基础) — 本策展的主挂载点
- [Modern Robotics Book](./modern-robotics-book.md) — L0 之后的刚体「语法书」
- [SE(3) Representation](../formalizations/se3-representation.md)
- [LQR / iLQR](../formalizations/lqr.md)
- [Pinocchio](./pinocchio.md)

## 参考来源

- [sources/courses/gatech_interactive_linear_algebra.md](../../sources/courses/gatech_interactive_linear_algebra.md)
- [sources/courses/axler_linear_algebra_done_right_4e.md](../../sources/courses/axler_linear_algebra_done_right_4e.md)
- [sources/courses/linear_algebra_teaching_materials_curated.md](../../sources/courses/linear_algebra_teaching_materials_curated.md)

## 推荐继续阅读（外部）

- [Interactive Linear Algebra](https://textbooks.math.gatech.edu/ila/)
- [Linear Algebra Done Right 4e（PDF）](https://linear.axler.net/LADR4e.pdf)
- [MIT 18.06 Linear Algebra（OCW）](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
