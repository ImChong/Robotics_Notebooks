---
type: formalization
tags: [control, dynamics, classical-control, damping, second-order, foundational]
status: complete
updated: 2026-09-11
related:
  - ./eigenvalues-eigenvectors.md
  - ./lqr.md
  - ../methods/pole-placement-control.md
  - ../methods/pid-control.md
  - ../methods/sim2real-joint-sysid-experiment-design.md
  - ../concepts/armature-modeling.md
  - ../concepts/impedance-control.md
  - ../concepts/joint-friction-models.md
sources:
  - ../../sources/courses/mit_2003_modeling_first_second_order_systems.md
  - ../../sources/courses/mit_6007_signals_second_order_systems.md
  - ../../sources/courses/mit_2161_first_second_order_transfer_functions.md
  - ../../sources/courses/openstax_physics_damped_oscillations.md
summary: "一阶时间常数 τ 与二阶阻尼比 ζ、自然频率 ωₙ 是读阶跃响应、闭环极点与关节 PD/阻抗调参的共同语言。"
---

# 阻尼系统（Damped Systems）

**阻尼系统**描述能量耗散如何塑造动态响应：一阶系统用**时间常数** $\tau$ 刻画指数衰减；二阶质量–弹簧–阻尼系统用**自然频率** $\omega_n$ 与**阻尼比** $\zeta$ 区分无阻尼振荡、欠阻尼衰减振荡、临界阻尼与过阻尼。机器人里，关节 PD 闭环、阻抗控制虚拟弹簧–阻尼、执行器黏性摩擦与 Sim2Real 阶跃辨识，都回到这套标准形。

## 一句话定义

> 线性时不变（LTI）系统的自然响应由极点决定：一阶为单实极点 $s=-1/\tau$；二阶为 $s^2+2\zeta\omega_n s+\omega_n^2=0$ 的根，$\zeta$ 划分欠阻尼（振荡衰减）、临界阻尼（最快无超调）与过阻尼（无振荡更慢）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LTI | Linear Time-Invariant | 线性时不变；自然响应为指数/指数调制正弦叠加 |
| τ | Time Constant | 一阶系统特征时间；一个 $\tau$ 内衰减至初值 $\approx 37\%$ |
| ωₙ | Natural Frequency | 二阶无阻尼振荡角频率 $\sqrt{k/m}$ |
| ζ | Damping Ratio | 实际阻尼与临界阻尼之比；$\zeta=b/(2\sqrt{km})$ |
| PD | Proportional–Derivative | 位置环增益提供等效 $k$ 与 $c$ |
| ROC | Region of Convergence | 拉普拉斯变换收敛域；因果系统极点左半平面 |

## 为什么重要

- **读曲线：** 阶跃响应的上升时间、超调、调节时间由 $\omega_n,\zeta$ 唯一标定（线性区）；Sim2Real 单关节辨识先归一化再读形状。
- **调增益：** PD 闭环 $\omega_n=\sqrt{K_p/J_{\mathrm{eff}}}$、$\zeta=(K_d+b)/(2\sqrt{K_p J_{\mathrm{eff}}})$；改 $K_d$ 主要改 $\zeta$，不改 $\omega_n$。
- **设柔顺：** 阻抗控制选 $M_d,B_d,K_d$；临界阻尼 $B_d\approx 2\sqrt{M_d K_d}$ 抑制接触后振荡。
- **连谱理论：** 闭环极点 = 状态矩阵特征值；与 [特征值与特征向量](./eigenvalues-eigenvectors.md) 同一套 $s$ 平面语言。

## 核心原理

### 一阶系统

标准齐次方程：

$$\tau \frac{dy}{dt} + y = 0$$

自然响应：

$$y(t) = c\, e^{-t/\tau}$$

| 量 | 含义 |
|----|------|
| $\tau>0$ | 稳定；极点 $s=-1/\tau$ 在左半平面 |
| $e^{-1}\approx 0.37$ | 经过一个 $\tau$，幅值降至初值 37% |
| 传递函数 | $H(s)=K_0/(\tau s+1)$ |

典型物理：RC 电路、热惯性、**弹簧–阻尼无质量**机械段、流体容性–阻性组合。

### 二阶系统（质量–弹簧–阻尼）

$$m\ddot x + b\dot x + kx = 0$$

或规范形：

$$\frac{1}{\omega_n^2}\ddot x + \frac{2\zeta}{\omega_n}\dot x + x = 0$$

其中

$$\omega_n = \sqrt{\frac{k}{m}}, \qquad \zeta = \frac{b}{2\sqrt{km}} = \frac{b}{b_c}, \quad b_c = 2\sqrt{km}$$

传递函数：

$$H(s) = \frac{K_0}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$

### 四类阻尼（$s$ 平面）

```mermaid
flowchart LR
  z0["ζ = 0 无阻尼\n虚轴共轭极点"]
  zu["0 < ζ < 1 欠阻尼\n左半平面共轭"]
  zc["ζ = 1 临界阻尼\n实轴重根 −ωₙ"]
  zo["ζ > 1 过阻尼\n两不等实根"]
  z0 --> zu --> zc --> zo
```

| 情形 | $\zeta$ | 极点 | 时域（初值响应） |
|------|---------|------|------------------|
| 无阻尼 | $0$ | $\pm j\omega_n$ | $x(t)=2M\cos(\omega_n t+\phi)$ 等幅振荡 |
| 欠阻尼 | $(0,1)$ | $-\zeta\omega_n \pm j\omega_d$ | 衰减振荡；$\omega_d=\omega_n\sqrt{1-\zeta^2}$ |
| 临界阻尼 | $1$ | $-\omega_n$（重根） | $c_1 e^{-\omega_n t}+c_2 t e^{-\omega_n t}$，最快无振荡 |
| 过阻尼 | $>1$ | 两负实根 | 两指数之和，无振荡、比临界慢 |

欠阻尼典型形（OpenStax / MIT 2.003）：

$$x(t) = A_0 e^{-\zeta\omega_n t}\cos(\omega_d t + \phi)$$

### 频域读法（轻阻尼共振）

$\zeta<1$ 时幅频在 $\omega_n$ 附近可出现**共振峰**；$\zeta\to 0$ 峰愈尖。机械结构需抑制不良共振；通信滤波器有时刻意利用共振选频（MIT 2.161 §5.2）。

## 机器人读法

### 关节 PD 闭环（无机械弹簧）

等效二阶：

$$\omega_n=\sqrt{\frac{K_p}{J_{\mathrm{eff}}}}, \qquad \zeta=\frac{K_d+b}{2\sqrt{K_p J_{\mathrm{eff}}}}$$

- $\omega_n$：**时间尺度**（多快）；$J_{\mathrm{eff}}$ 含 [Armature](../concepts/armature-modeling.md) 反射惯量。
- $\zeta$：**形状**（有无超调、振荡几次）；只调 $K_d$ 时 $\omega_n$ 不变。
- 见 [关节动力学辨识实验设计](../methods/sim2real-joint-sysid-experiment-design.md)。

### 阻抗 / 导纳控制

任务空间期望动力学：

$$M_d(\ddot x-\ddot x_d)+B_d(\dot x-\dot x_d)+K_d(x-x_d)=f_{\mathrm{ext}}$$

准静态常取 $B_d\approx 2\sqrt{M_d K_d}$（临界阻尼）避免接触后长时间振荡。见 [阻抗控制](../concepts/impedance-control.md)。

### 极点配置

指定闭环极点 $\{p_i\}$ 等价于指定各模态的 $\omega_n,\zeta$；二阶近似下直接映射超调与调节时间。见 [极点配置](../methods/pole-placement-control.md)。

### 摩擦与黏性阻尼

关节被动项 $b\dot q$ 进入 $\zeta$ 分子；与 [关节摩擦模型](../concepts/joint-friction-models.md) 辨识耦合。

## 工程实践

| 任务 | 建议 |
|------|------|
| 读阶跃曲线 | 先按 $\omega_n$ 归一化时间轴，再估 $\zeta$（超调 $\approx e^{-\pi\zeta/\sqrt{1-\zeta^2}}$，$0<\zeta<1$） |
| 设 PD | 先定目标 $\omega_n$（带宽），再定 $\zeta$（通常 $0.7$–$1$ 或故意 $>1$ 抑制振荡） |
| Sim2Real | 延迟/惯量/摩擦纠缠时换实验，不要只拟合一条曲线 |
| 接触任务 | 环境越硬，末端等效刚度宜越低，避免闭环 $\zeta$ 过小致不稳定 |
| 减震器设计 | 目标常是**临界阻尼**：无超调且最快回平衡（OpenStax §15.5） |

## 局限与风险

- **线性区：** $\zeta,\omega_n$ 公式假设小信号、线性阻尼；大摩擦/饱和破坏二阶读法。
- **高阶未建模：** 延迟、柔性模态表现为额外极点；二阶只是局部近似。
- **参数纠缠：** $J_{\mathrm{eff}}$ 与增益、$k_t$ 可互换；需实验设计拆开（见 SysID 实验设计页）。
- **热/流体一阶：** 热系统无二阶「惯量」类比；勿机械照搬 $m\ddot x$ 形。

## 关联页面

- [特征值与特征向量](./eigenvalues-eigenvectors.md) — 极点 = 特征值；$s$ 平面稳定性
- [LQR](./lqr.md) — 最优闭环极点放置
- [极点配置](../methods/pole-placement-control.md)
- [PID 控制](../methods/pid-control.md)
- [关节动力学辨识实验设计](../methods/sim2real-joint-sysid-experiment-design.md)
- [Armature 建模](../concepts/armature-modeling.md)
- [阻抗控制](../concepts/impedance-control.md)
- [关节摩擦模型](../concepts/joint-friction-models.md)
- [仿真物理保真度链路选型指南](../queries/simulation-physics-fidelity.md) — 仿真器的关节 `damping` / `armature` 与积分步长决定了 ζ、ωₙ 能否被如实复现，属保真度链路第 ① 建模层与第 ② 数值层
- [Physics Fidelity ↔ Sim2Real Gap](../concepts/physics-fidelity-sim2real-gap.md) — 仿真与实机阻尼比失配是关节级 sim2real gap 的常见来源

## 参考来源

- [MIT 2.003 Ch.1 自然响应归档](../../sources/courses/mit_2003_modeling_first_second_order_systems.md) — $\tau$、$\omega_n$、$\zeta$ 四类情形
- [MIT 6.007 Lec 21 归档](../../sources/courses/mit_6007_signals_second_order_systems.md) — $H(s)$ 与过/欠阻尼极点
- [MIT 2.161 §5.1–5.2 归档](../../sources/courses/mit_2161_first_second_order_transfer_functions.md) — 一阶滞后与二阶共振
- [OpenStax Physics §15.5 归档](../../sources/courses/openstax_physics_damped_oscillations.md) — 黏性阻尼物理直觉

## 推荐继续阅读（外部）

- [MIT 2.003 Notes Installment 2 (PDF)](https://ocw.mit.edu/courses/2-003-modeling-dynamics-and-control-i-spring-2005/57d44d83366ec969c16208c8fac3982d_notesinstalment2.pdf)
- [MIT 6.007 Lecture 21 (PDF)](https://ocw.mit.edu/courses/res-6-007-signals-and-systems-spring-2011/10c777db2cf27ca7a55b08d3fd6e3e78_MITRES_6_007S11_lec21.pdf)
- [OpenStax 15.5 Damped Oscillations](https://openstax.org/books/university-physics-volume-1/pages/15-5-damped-oscillations)
- [MIT 18.03SC Damped Harmonic Oscillators (PDF)](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/911bc225e5913ba15517dbd70528c27a_MIT18_03SCF11_s13_1text.pdf)
