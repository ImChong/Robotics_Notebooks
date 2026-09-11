# MIT 2.003 — Modeling Dynamics and Control I, Ch.1 Natural Response

> 来源归档（一手教材节选）

- **标题：** Modeling Dynamics and Control I — Chapter 1: Natural Response (First- and Second-Order Systems)
- **作者：** MIT Department of Mechanical Engineering（课程讲义 Notes Installment 1 & 2）
- **类型：** course notes
- **链接：**
  - [Notes Installment 1 — First-order systems](https://ocw.mit.edu/courses/2-003-modeling-dynamics-and-control-i-spring-2005/c2a4a0e8708c2c0d57d1d0c79d97f7aa_NotesInstallment1.pdf)
  - [Notes Installment 2 — Second-order systems](https://ocw.mit.edu/courses/2-003-modeling-dynamics-and-control-i-spring-2005/57d44d83366ec969c16208c8fac3982d_notesinstalment2.pdf)
- **入库日期：** 2026-09-11
- **一句话说明：** LTI 系统自然响应的「积木」：一阶指数衰减（时间常数 τ）与二阶质量–弹簧–阻尼（ωₙ、ζ）四类极点情形。

## 核心定义（原文要点）

### 一阶系统

标准齐次方程：

$$\tau \frac{dy}{dt} + y = 0$$

- $\tau$：**时间常数**（秒）；特征值（极点）$\lambda_1 = -1/\tau$。
- 自然响应：$y(t) = c\, e^{-t/\tau}$；$\tau>0$ 时稳定，一个时间常数内衰减至初值的 $\approx 37\%$（$e^{-1}\approx 0.37$）。
- 物理域：机械（弹簧–阻尼无质量）、电气（RC/RL）、热、流体等均可化为该形。

### 二阶系统（质量–弹簧–阻尼）

$$m\frac{d^2x}{dt^2} + b\frac{dx}{dt} + kx = 0$$

特征方程 $ms^2 + bs + k = 0$；标准参数化：

$$\omega_n = \sqrt{\frac{k}{m}}, \qquad \zeta = \frac{b}{2\sqrt{km}} = \frac{b}{b_c}, \quad b_c = 2\sqrt{km}$$

规范形：

$$\frac{1}{\omega_n^2}\frac{d^2x}{dt^2} + \frac{2\zeta}{\omega_n}\frac{dx}{dt} + x = 0$$

| 情形 | 条件 | 极点 | 时域特征 |
|------|------|------|----------|
| 无阻尼 | $\zeta=0$ | $s=\pm j\omega_n$ | 等幅正弦 $x(t)=2M\cos(\omega_n t+\phi)$ |
| 欠阻尼 | $0<\zeta<1$ | 左半平面共轭复极点 | 衰减振荡；$\omega_d=\omega_n\sqrt{1-\zeta^2}$ |
| 临界阻尼 | $\zeta=1$ | 实轴重根 $s=-\omega_n$ | 最快无振荡回归；$x(t)=c_1 e^{-\omega_n t}+c_2 t e^{-\omega_n t}$ |
| 过阻尼 | $\zeta>1$ | 左半平面两不等实根 | 无振荡，慢于临界阻尼 |

## 对 wiki 的映射

- 沉淀 **[`wiki/formalizations/damped-systems.md`](../../wiki/formalizations/damped-systems.md)**
- 交叉 [关节动力学辨识实验设计](../../wiki/methods/sim2real-joint-sysid-experiment-design.md)、[极点配置](../../wiki/methods/pole-placement-control.md)、[Armature 建模](../../wiki/concepts/armature-modeling.md)

## 推荐继续阅读（外部）

- [MIT 2.003 OCW 课程主页](https://ocw.mit.edu/courses/2-003-modeling-dynamics-and-control-i-spring-2005/)
- [MIT 6.007 Signals and Systems Lec 21](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/resources/lecture-21-continuous-time-second-order-systems/)
