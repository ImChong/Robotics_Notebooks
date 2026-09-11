# MIT 2.161 — First- and Second-Order Transfer Functions (Bode Notes §5)

> 来源归档（一手课程讲义）

- **标题：** 2.161 Signal Processing: Continuous and Discrete — Sinusoidal Frequency Response of Linear Systems, §5.1–5.2
- **作者：** David Rowell（MIT Department of Mechanical Engineering）
- **类型：** course notes (PDF)
- **链接：** <https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/a9a6fa4c4c56513e2d5df392df6123ee_bode.pdf>
- **入库日期：** 2026-09-11
- **一句话说明：** 一阶滞后 $H(s)=K_0/(\tau s+1)$ 与二阶标准形 $H(s)=K_0/(s^2+2\zeta\omega_n s+\omega_n^2)$ 的频域响应、共振峰与 $\zeta$ 参数。

## 核心定义（原文要点）

### 一阶系统（§5.1）

微分方程 $\tau \dot y + y = K_0 u(t)$；传递函数：

$$H(s) = \frac{K_0}{\tau s + 1}$$

- $\tau$：时间常数；极点 $s=-1/\tau$。
- 阶跃响应指数趋近稳态；Bode 幅频在 $\omega=1/\tau$ 附近转折。

### 二阶系统（§5.2）

标准传递函数：

$$H(s) = \frac{K_0}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$

- $\omega_n$：无阻尼自然频率；$\zeta$：阻尼比。
- **过阻尼**（$\zeta>1$）：幅频单调下降。
- **欠阻尼**（$\zeta<1$）：$\omega_n$ 附近出现**共振峰**；$\zeta\to 0$ 峰愈窄愈高。
- 共振频率 $\omega_m = \omega_n\sqrt{1-2\zeta^2}$（$\zeta\le 1/\sqrt{2}$ 时存在峰值）。

### 稳态正弦响应

稳定系统对 $u(t)=A\sin(\Omega t+\psi)$ 的稳态输出由 $H(j\Omega)$ 决定；瞬态齐次项 $e^{\lambda_i t}$ 衰减后仅剩特解。

## 对 wiki 的映射

- 沉淀 **[`wiki/formalizations/damped-systems.md`](../../wiki/formalizations/damped-systems.md)** — 频域与阶跃响应的统一语言
- 交叉 [关节摩擦模型](../../wiki/concepts/joint-friction-models.md)（黏性阻尼 $b$）

## 推荐继续阅读（外部）

- [MIT 2.161 OCW 课程主页](https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/)
