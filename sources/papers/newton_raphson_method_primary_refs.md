# Newton–Raphson 法一手资料汇编

> 来源归档（ingest）

- **标题：** Newton–Raphson Method — Primary References
- **类型：** paper + textbook（历史原典 + 机器人教材算例）
- **入库日期：** 2026-09-21
- **一句话说明：** 牛顿–拉夫森（Newton–Raphson）迭代求根/求逆运动学的历史原典与机器人学标准教材算例，支撑 `wiki/methods/newtons-method.md` 与 IK 数值解交叉引用。

## 为什么值得保留

- **名称溯源：** 现代数值分析所称 **Newton–Raphson** 直接来自 Joseph Raphson 1690 年小册；Newton 的流数法（fluxions）更早但公开更晚，二者常被合并叙述。
- **机器人主入口：** Lynch & Park *Modern Robotics* **§6.2.1–6.2.2** 把标量 NR 推广到 **Jacobian 伪逆 IK**，并给出 **Example 6.1（平面 2R）** 完整迭代表——是本库 IK 数值解的一手教材证据。
- **与优化牛顿法关系：** 标量/向量 NR 求 $g(\theta)=0$ 与无约束优化中牛顿步 $p=-H^{-1}g$ 同族；后者见 [`second_order_optimizers.md`](second_order_optimizers.md)。

## 一手资料摘录

### 1) Raphson (1690) — *Analysis æquationum universalis*

- **作者：** Joseph Raphson（约 1668–1715/16）
- **出版：** London: Abel Swalle, **1690**（再版 1697 常见扫描）
- **链接：**
  - 书目记录（1690 首版）：<https://findit.library.nd.edu/Record/001459768>
  - 1697 版扫描（Internet Archive）：<https://archive.org/details/bub_gb_4nlbAAAAQAAJ>
  - e-rara PDF（1697）：<https://doi.org/10.3931/e-rara-13516>
- **核心贡献：** 对代数方程 $f(x)=0$ 给出 **通用、快捷** 的迭代逼近根方法；用无穷级数思想线性化并重复修正——即今日 NR 的雏形。
- **对 wiki 的映射：**
  - [Newton's Method / Newton–Raphson](../../wiki/methods/newtons-method.md) — 历史命名与标量求根

### 2) Newton — 流数法（Method of Fluxions，1669 手稿；1711 公开）

- **说明：** Newton 在 Raphson 之前已用类似迭代解方程，但长期以手稿/通信形式流传；**「Newton–Raphson」** 合称强调 Raphson 1690 独立出版物与 Newton 流数传统。
- **参考综述：** MacTutor History of Mathematics — Newton–Raphson method：<https://mathshistory.st-andrews.ac.uk/HistTopics/Newton-Raphson/>
- **对 wiki 的映射：** 同上，作为历史脚注，避免与 [Newton–Euler 动力学](../papers/modern_robotics_textbook.md)（Ch 8 刚体递推）混淆。

### 3) Lynch & Park — *Modern Robotics* §6.2.1–6.2.2 & Example 6.1

- **链接：** 官方 PDF <https://hades.mech.northwestern.edu/images/7/7f/MR.pdf> — **Ch 6, pp. 226–232**
- **归档：** [`modern_robotics_textbook.md`](modern_robotics_textbook.md)
- **标量 NR（§6.2.1）：** 对 $g(\theta)=0$，
  $$\theta_{k+1}=\theta_k-\left(\frac{\partial g}{\partial\theta}(\theta_k)\right)^{-1} g(\theta_k).$$
- **IK 向量形式（§6.2.2）：** $g(\theta)=x_d-f(\theta)$，$\Delta\theta=J^\dagger(\theta_0)(x_d-f(\theta_0))$；SE(3) 目标改用 body twist $V_b=\log(T_{bs}^{-1}T_{sd})$ 与 $J_b$。
- **Example 6.1（平面 2R，链长各 1 m）：**
  - 目标：$T_{sd}$ 对应 $(\theta_1,\theta_2)=(30^\circ,90^\circ)$，末端 $(x,y)=(0.366,1.366)$ m
  - 初值：$\theta_0=(0^\circ,30^\circ)$
  - 容差：$\omega=0.001$ rad，$v=10^{-4}$ m
  - **一步后：** $(34.23^\circ,79.18^\circ)$；**三步收敛**至 $(30.00^\circ,90.00^\circ)$
- **对 wiki 的映射：**
  - [newtons-method.md](../../wiki/methods/newtons-method.md) — 算例节
  - [inverse-kinematics.md](../../wiki/formalizations/inverse-kinematics.md) — 数值 IK
  - [modern-robotics-book.md](../../wiki/entities/modern-robotics-book.md) — Ch 6

### 4) Nocedal & Wright — *Numerical Optimization*（求根 ↔ 优化牛顿步）

- **链接：** <https://doi.org/10.1007/978-0-387-40065-5> — Ch 1（线搜索）、Ch 2（牛顿型方法）
- **要点：** 解 $g(x)=0$ 的 NR 与最小化 $f$ 的牛顿步在 $g=\nabla f$ 时一致；机器人 TrajOpt 语境见 [`second_order_optimizers.md`](second_order_optimizers.md)。
- **对 wiki 的映射：** [newtons-method.md](../../wiki/methods/newtons-method.md)、[second-order-optimizers](../../wiki/comparisons/second-order-optimizers.md)

## 当前提炼状态

- [x] 历史原典（Raphson 1690）与 MR Example 6.1 算例入库
- [x] 映射到 newtons-method / inverse-kinematics
- [ ] 后续可补：阻尼 NR / 信赖域与 DLS 在奇异 Jacobian 下的对照专节
