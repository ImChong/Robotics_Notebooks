# Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors（Diebel, 2006）

> 来源归档（ingest）

- **标题:** Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors
- **作者:** James Diebel
- **机构:** Stanford University
- **类型:** technical report / survey
- **日期:** 2006-10-20
- **PDF:** https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf
- **镜像:** http://robots.stanford.edu/papers/thrun/diebel_rep_attitude.pdf（Thrun 组页面常见链接）
- **入库日期:** 2026-09-09
- **一句话说明:** 机器人/视觉领域最常被引用的姿态参数化统一参考：旋转矩阵、欧拉角、**单位四元数**、旋转向量之间的转换表、求导与工程选型，并系统讨论 $q \equiv -q$ 双覆盖与单位范数约束。

## 开源状态（步骤 2.5）

- **无代码仓库**；PDF 为公开技术报告，可自由分发。
- **结论:** 理论一手资料；实现对照以各框架（Pinocchio、ROS、PyTorch3D 等）自声明的 **scalar 顺序** 为准。

## 摘录 1：为何需要统一参考

> Abstract: … (3) the unit quaternion. To these we add a fourth, the rotation vector … neither the singularities of the former, nor the quadratic constraint of the latter.

- 在线文献对欧拉角/四元数约定 **各说各话**；本文目标是 **单一、可检索的转换目录**（含 12 组欧拉角序列 catalog，Sec. 8）。
- 欧拉角缺点：奇异（gimbal lock）+ 积分增量时精度不如四元数。
- 四元数缺点：**单位范数二次约束**，直接优化时需 renormalize 或惩罚项（Sec. 6.16）。

**对 wiki 的映射:** [`wiki/formalizations/unit-quaternion-so3.md`](../../wiki/formalizations/unit-quaternion-so3.md) — 选型表与误区节。

## 摘录 2：Hamilton 四元数与乘法（Sec. 6.1–6.2）

Diebel 采用 **scalar-first** 记法 $q=[q_0, q_1, q_2, q_3]^\top$，$q_0$ 为实部、$q_{1:3}$ 为虚部向量。

Hamilton 积（非交换）：

$$
q \cdot p =
\begin{bmatrix}
q_0 p_0 - q_{1:3}^\top p_{1:3} \\
q_0 p_{1:3} + p_0 q_{1:3} - q_{1:3} \times p_{1:3}
\end{bmatrix}
$$

矩阵形式：$q \cdot p = Q(q)\,p$，其中 $Q(q)$ 为 $4\times4$ 四元数矩阵（式 108–109）。

**对 wiki 的映射:** 工程实践节强调 **读代码前必须先确认 $(w,x,y,z)$ 还是 $(x,y,z,w)$**；Diebel 与 DeepMimic README 的 $(w,x,y,z)$ 一致，Pinocchio / scipy `Rotation` 常用 `xyzw`。

## 摘录 3：单位四元数 ↔ 旋转矩阵（Sec. 6.4）

设 $\|q\|=1$，体坐标向量 $z_0$ 与全局 $z$ 满足：

$$
z = R_q(q)\, z_0, \quad z_0 = R_q(q)^\top z
$$

旋转复合：$R_q(q \cdot p) = R_q(q)\, R_q(p)$。

轴角参数化（Sec. 6.12）：绕单位轴 $n$ 转 $\alpha$ 弧度：

$$
q_a(\alpha, n) = \left[\cos\frac{\alpha}{2},\; \sin\frac{\alpha}{2}\, n^\top\right]^\top
$$

（scalar-first；与 Shoemake 的 $w=\cos(\theta/2)$ 一致。）

**对 wiki 的映射:** 与 [李群形式化页](../../wiki/formalizations/lie-group-rigid-body-motions.md) 的 Rodrigues / exp 映射对照。

## 摘录 4：四元数率 ↔ 角速度（Sec. 6.6）

$$
\omega = 2 W(q)\,\dot q, \quad \omega_0 = 2 W_0(q)\,\dot q
$$

$W, W_0$ 为 $3\times4$ 四元数率矩阵；分别对应 **世界系** 与 **体固定系** 角速度。

**对 wiki 的映射:** 浮动基 $n_q=7$（四元数姿态 + 平移）而 $n_v=6$ 的根因；见 [floating-base-dynamics](../../wiki/concepts/floating-base-dynamics.md)。

## 摘录 5：优化中的单位约束（Sec. 6.16）

- 迭代法：每步 **renormalize** $q \leftarrow q/\|q\|$。
- 直接法：目标中加入 $(1-\|q\|)^2$ 惩罚 + 仍常需 renormalize。
- 旋转向量 $q_v:\mathbb{R}^3\to S^3$ 可消二次约束，但仍有 $2\pi$ 周期歧义（Sec. 7）。

**对 wiki 的映射:** 与 Zhou 6D 等无单位约束观测表示的动机对齐；选型总表见 [旋转表示方法对比](../../wiki/comparisons/so3-rotation-representations.md)，位姿层见 [SE(3) 位姿表示](../../wiki/formalizations/se3-representation.md)。

## 建议 wiki 动作

- 新建 **`wiki/formalizations/unit-quaternion-so3.md`**
- 更新 **`wiki/formalizations/lie-group-rigid-body-motions.md`**、**`se3-representation.md`** 交叉引用与 `sources` frontmatter
