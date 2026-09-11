# OpenStax — University Physics Vol.1 §15.5 Damped Oscillations

> 来源归档（一手教材节选）

- **标题：** University Physics Volume 1 — 15.5 Damped Oscillations
- **作者：** OpenStax（Paul Peter Urone, Roger Hinrichs 等）
- **类型：** textbook section
- **链接：** <https://openstax.org/books/university-physics-volume-1/pages/15-5-damped-oscillations>
- **入库日期：** 2026-09-11
- **一句话说明：** 黏性阻尼 $F_D=-bv$ 下的质量–弹簧–阻尼方程、指数包络衰减振荡与欠/临界/过阻尼三分法（物理直觉入口）。

## 核心方程（原文要点）

受力平衡（黏性阻尼与速度成正比，$|v|$ 小时）：

$$ma = -bv - kx$$

即

$$m\frac{d^2x}{dt^2} + b\frac{dx}{dt} + kx = 0$$

欠阻尼解（小阻尼）：

$$x(t) = A_0 e^{-\frac{b}{2m}t}\cos(\omega t + \phi), \qquad \omega = \sqrt{\frac{k}{m} - \left(\frac{b}{2m}\right)^2}$$

自然角频率 $\omega_0 = \sqrt{k/m}$；有阻尼时 $\omega = \sqrt{\omega_0^2 - (b/2m)^2}$。

### 阻尼分类（与 $b$ 的关系）

| 类型 | 条件 | 行为 |
|------|------|------|
| 欠阻尼 | $b < 4mk$（即 $\zeta<1$） | 振荡 + 振幅指数衰减 |
| 临界阻尼 | $b = 4mk$（$\zeta=1$） | 无振荡，最快回到平衡（如汽车减震器设计目标） |
| 过阻尼 | $b > 4mk$（$\zeta>1$） | 无振荡，回归更慢 |

临界阻尼在工程上常受青睐：无超调且尽快到达新平衡。

## 对 wiki 的映射

- 沉淀 **[`wiki/formalizations/damped-systems.md`](../../wiki/formalizations/damped-systems.md)** — 物理弹簧–质量–dashpot 与 $\zeta$ 分类
- 交叉 [阻抗控制](../../wiki/concepts/impedance-control.md)（虚拟质量–弹簧–阻尼）

## 推荐继续阅读（外部）

- [OpenStax Univ. Physics Vol.1 Ch.15 Oscillations](https://openstax.org/books/university-physics-volume-1/pages/15-introduction)
- [MIT 18.03SC Damped Harmonic Oscillators](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/911bc225e5913ba15517dbd70528c27a_MIT18_03SCF11_s13_1text.pdf)
