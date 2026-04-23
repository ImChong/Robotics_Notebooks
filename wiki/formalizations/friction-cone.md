---
type: formalization
tags: [dynamics, contact, physics, control, math]
status: complete
updated: 2026-04-20
related:
  - ../concepts/contact-dynamics.md
  - ../concepts/centroidal-dynamics.md
  - ../concepts/whole-body-control.md
  - ./zmp-lip.md
  - ../queries/wbc-tuning-guide.md
sources:
  - ../../sources/papers/contact_dynamics.md
summary: "摩擦锥（Friction Cone）描述了接触力在接触面上不发生相对滑动的物理约束条件，是 WBC 和轨迹优化中最重要的不等式约束。"
---

# 摩擦锥 (Friction Cone)

**摩擦锥** 是机器人学中描述接触力物理约束的核心数学模型。它规定了接触力 $\mathbf{f}$ 必须满足的范围，以确保机器人脚部或手部与支撑环境之间不发生滑动。

## 库仑摩擦模型 (Coulomb Friction)

根据库仑摩擦定律，静态摩擦力 $f_t$（切向）与正向压力 $f_n$（法向）的关系如下：

$$
\| \mathbf{f}_t \| \leq \mu f_n, \quad f_n \geq 0
$$

其中：
- $\mu$ 是静摩擦系数。
- $\mathbf{f}_n = (\mathbf{f} \cdot \mathbf{n}) \mathbf{n}$ 是正向分量。
- $\mathbf{f}_t = \mathbf{f} - \mathbf{f}_n$ 是切向分量。

这个不等式在三维空间中定义了一个以 $\mathbf{n}$ 为中心轴、顶角为 $2\arctan(\mu)$ 的**圆锥体**。这就是“摩擦锥”名称的由来。

## 二次锥约束 (SOC)

在优化问题中，上述关系可以写成 **二阶锥约束（Second-Order Cone, SOC）** 的形式：

$$
\sqrt{f_x^2 + f_y^2} \leq \mu f_z, \quad f_z \geq 0
$$

虽然 SOC 描述最精确，但通用的 QP（二次规划）求解器（如 OSQP, qpOASES）通常不支持直接处理 SOC 约束。

## 线性化摩擦锥 (Linearized Friction Cone)

为了在实时控制器（如 WBC 或 MPC）中利用高效的 QP 求解器，通常将圆锥体通过其内切或外接的 **多面体（Polyhedral Cone）** 进行线性化逼近。

### 四棱锥近似

最常见的做法是使用四棱锥（4-sided pyramid）来近似圆锥。这意味着在切线 $x$ 和 $y$ 方向上分别满足：

$$
|f_x| \leq \frac{\mu}{\sqrt{2}} f_z, \quad |f_y| \leq \frac{\mu}{\sqrt{2}} f_z, \quad f_z \geq 0
$$

这可以展开为 4-5 个线性不等式：
1. $f_z \geq f_{min}$
2. $f_x - \frac{\mu}{\sqrt{2}} f_z \leq 0$
3. $-f_x - \frac{\mu}{\sqrt{2}} f_z \leq 0$
4. $f_y - \frac{\mu}{\sqrt{2}} f_z \leq 0$
5. $-f_y - \frac{\mu}{\sqrt{2}} f_z \leq 0$

## 在控制中的应用

1. **WBC 中的 QP 约束**：在求解关节力矩时，必须确保计算出的接触力 $\mathbf{f}$ 位于摩擦锥内。
2. **质心动力学优化**：在规划 CoM 轨迹时，外部合力必须由摩擦锥内的力合成。
3. **抓取规划**：在操纵物体时，指尖力必须位于接触面的摩擦锥内以防止物体滑脱。

## 关联页面
- [Contact Dynamics](../concepts/contact-dynamics.md)
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [ZMP + LIP 形式化](./zmp-lip.md)
- [Contact Wrench Cone（接触力旋量锥）](./contact-wrench-cone.md)
- [WBC 调参指南](../queries/wbc-tuning-guide.md)

## 参考来源
- [contact_dynamics.md](../../sources/papers/contact_dynamics.md)
- Siciliano, B., et al. (2009). *Robotics: Modelling, Planning and Control*.
