# MIT Underactuated Robotics 第 8 章 LQR 通读

> 来源归档（原文通读 · 支撑 `wiki/formalizations/lqr.md`）

- **标题：** *Underactuated Robotics* Ch. 8 — Linear Quadratic Regulators（Russ Tedrake, 2024 版）
- **类型：** textbook chapter / site
- **原文：** <https://underactuated.csail.mit.edu/lqr.html>（2026-11-07 抓取 HTML 转文本，约 6900 词）
- **入库日期：** 2026-11-07
- **一句话说明：** 从 HJB 推出连续时间 LQR，讲有限时域 / 时变 / 跟踪 / 离散时间 Riccati、约束 LQR、流形上的 LQR、LQR 的凸优化与最小二乘表述。
- **核对范围：** 全章正文（基本推导、双积分器例、非线性系统的局部稳定与轨迹稳定、有限时域与时变 LQR、线性最优跟踪、线性终端边界值问题、离散 Riccati 与折扣形式、拟合值迭代、带不等式约束与 MPC 的引出、流形上的 LQR 与 Segway 例、隐式形式线性系统、LQR 作为凸优化、有限时域最小二乘与 SLS）。章末"Notes"中的一般形式有限时域推导只浏览了结论；交互式代码单元与 Drake 实现、最小时间 LQR（原文标注"Coming soon"）、参考文献未读。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **基本推导：** LTI 系统 ẋ=Ax+Bu，无限时域代价 J=∫(xᵀQx+uᵀRu)dt，Q⪰0，R≻0。HJB：0=min_u[xᵀQx+uᵀRu+∂J*/∂x(Ax+Bu)]；猜 J*=xᵀSx，得 u*=−R⁻¹BᵀSx=−Kx，S 满足代数 Riccati 方程 0=SA+AᵀS−SBR⁻¹BᵀS+Q。原文称该方程有唯一正定解当且仅当系统可镇定（stabilizable）；其他（非正定）解对应不稳定控制器。双积分器 Q=I、R=1 的结果 K=[1, √3]，S=[[√3,1],[1,√3]]。−BᵀSx 是最速下降方向在输入空间的投影。
- **非线性系统的局部使用：** 在平衡点 (x0,u0) 线性化，u*=u0−K(x−x0)；沿标称轨迹线性化得到时变 LQR，u*=u0(t)−K(t)(x−x0(t))，且线性化在整条可行轨迹上有效；要谈"稳定轨迹"，轨迹须延伸到无穷时间，可令其终止于可镇定的不动点并用 S_∞ 作终端条件。
- **有限时域：** HJB 多一项 ∂J*/∂t，微分 Riccati 方程 −Ṡ=SA+AᵀS−SBR⁻¹BᵀS+Q，S(t_f)=Q_f；无限时域解是其稳态解；数值积分在弱可镇定系统上可能丢失对称 / 正定性，可积分平方根形式。
- **离散时间：** 价值递推给出 K[n]=(R+BᵀS[n]B)⁻¹BᵀS[n]A，Riccati 差分方程 S[n−1]=Q+AᵀS[n]A−(AᵀS[n]B)(R+BᵀS[n]B)⁻¹(BᵀS[n]A)，S[N]=0；无限时域为其正定不动点（DARE）；折扣形式 u*=−γ(R+γBᵀSB)⁻¹BᵀSAx。
- **拟合值迭代：** 每个 Riccati 解都是拟合值迭代的不动点，但只有正定解是稳定不动点（误差按 Δ_{i+1}=(A−BK_i)ᵀΔ_i(A−BK_i) 演化）。
- **约束：** 加线性不等式约束后一般不再有闭式策略，转而在有限时域上解凸优化，每步重解即模型预测控制（MPC）。
- **流形上的 LQR：** 对线性等式约束 Fx=0，取 F 零空间的正交基 P，在约化坐标 y 上做 LQR（A_y=PA P^T，B_y=PB，Q_y=PQP^T），K_x=K_yP；Segway / ballbot 例中直接在全坐标线性化会因不可镇定而失败。
- **其他表述：** LQR 可写成 LMI；有限时域 LQR 通过重新参数化 u[n]=K̃_n x0 化为最小二乘；System-Level Synthesis 直接优化闭环响应，适合加稀疏等额外结构；Riccati 解随标称点光滑变化。原文态度：若只要简单解仍首选解 Riccati 方程，其他表述在要联合其他目标 / 约束时才更强。

## 对 wiki 的映射

- [`wiki/formalizations/lqr.md`](../../wiki/formalizations/lqr.md)

## 参考来源

- Tedrake, R. *Underactuated Robotics*, Ch. 8. <https://underactuated.csail.mit.edu/lqr.html>；既有 ingest 归档 [`courses/mit_underactuated_kalman_lqr.md`](../courses/mit_underactuated_kalman_lqr.md)
