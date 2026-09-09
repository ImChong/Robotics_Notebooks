# Modern Robotics Ch 3：单位四元数与 SO(3)（Lynch & Park, 2017）

> 来源归档（ingest 摘录）

- **标题:** Modern Robotics: Mechanics, Planning, and Control — Chapter 3 *Rigid-Body Motions*
- **作者:** Kevin M. Lynch & Frank C. Park
- **出版:** Cambridge University Press, 2017
- **PDF:** https://hades.mech.northwestern.edu/images/7/7f/MR.pdf
- **配套:** http://modernrobotics.org/
- **入库日期:** 2026-09-09
- **关联 source:** [`modern_robotics_textbook.md`](modern_robotics_textbook.md)
- **一句话说明:** 教材 Ch 3 以 **旋转矩阵 + 指数坐标（旋转向量）** 为主轴建立 SO(3)/SE(3)；单位四元数作为与轴角等价的紧凑存储，并给出与 PoE / twist 一致的群论语言——机器人学课程级一手推导。

## 开源状态（步骤 2.5）

- **配套 Python/MAThematica 库:** [ModernRobotics packages](https://github.com/NxRobotics/ModernRobotics)（Northwestern 维护）
- **结论:** 教材 PDF 免费；代码 Apache-2.0。

## 摘录 1：SO(3) 与三种等价姿态存储（Ch 3.2 归纳）

| 表示 | 维度 | 约束 | 教材角色 |
|------|------|------|----------|
| 旋转矩阵 $R$ | 9 | $R^\top R=I$, $\det R=1$ | 变换复合、PoE |
| 指数坐标 / 旋转向量 $\omega$ | 3 | 无（表 **增量**） | 优化、twist |
| 单位四元数 | 4 | $\|q\|=1$, $q\sim -q$ | 紧凑存储、插值 |

**对 wiki 的映射:** [`wiki/formalizations/unit-quaternion-so3.md`](../../wiki/formalizations/unit-quaternion-so3.md) 与 [`lie-group-rigid-body-motions.md`](../../wiki/formalizations/lie-group-rigid-body-motions.md) 分工：本摘录提供 **教材 canonical** 定义，Diebel 提供 **转换表**，Shoemake 提供 **SLERP**。

## 摘录 2：指数映射与 Rodrigues（Ch 3.2.1）

$$
[\omega]_\times \in \mathfrak{so}(3), \quad R = \exp([\omega]_\times)
$$

Rodrigues 公式与 Diebel 旋转向量 / 轴角四元数一致。工程上常见链路：

**四元数 $\leftrightarrow$ 矩阵 $\leftrightarrow$ $\log(R)\to\omega$**

**对 wiki 的映射:** WBC / MPC 在 se(3) 切空间线性化时仍以 **矩阵 + 对数** 为主；四元数多用于 sim 状态与 MoCap。

## 摘录 3：SE(3) 与浮动基（Ch 3.3 + Ch 8 联系）

位姿 $T=(R,t)$；速度用 **twist** $\mathcal{V}\in\mathbb{R}^6$，而非对四元数分量直接求导。

- 配置 $q$ 含四元数时 $n_q \neq n_v$（与 Pinocchio / MuJoCo 浮动基一致）。
- 教材强调在 **se(3)** 上定义速度，避免把 $\dot q_{3:7}$ 误当角速度。

**对 wiki 的映射:** [`wiki/queries/pinocchio-quick-start.md`](../../wiki/queries/pinocchio-quick-start.md)、[`floating-base-dynamics.md`](../../wiki/concepts/floating-base-dynamics.md)。

## 摘录 4：与 DeepMimic / 运动模仿栈

DeepMimic 原始 motion JSON 用 **四元数 $(w,x,y,z)$** 存 spherical joint（[`humanoid_rl_stack_11` DeepMimic 摘录](../papers/humanoid_rl_stack_11_deepmimic_example_guided_deep_reinforcement_lear.md)）；MimicKit 动作 `.pkl` 改用 **exp map** 存盘、观测侧可用 tan_norm/四元数——三层表示勿混。

**对 wiki 的映射:** [`wiki/methods/deepmimic.md`](../../wiki/methods/deepmimic.md)。

## 建议 wiki 动作

- 新建 **`wiki/formalizations/unit-quaternion-so3.md`**
- 在 **`wiki/entities/modern-robotics-book.md`** 章节地图中增加指向本页链接（可选维护）
