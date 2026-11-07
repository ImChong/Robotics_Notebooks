# Momentum Control with Hierarchical Inverse Dynamics on a Torque-Controlled Humanoid（Herzog et al., 2016）

> 来源归档（ingest · 原文通读）

- **标题：** Momentum Control with Hierarchical Inverse Dynamics on a Torque-Controlled Humanoid
- **作者：** Alexander Herzog、Nicholas Rotella、Sean Mason、Felix Grimminger、Stefan Schaal、Ludovic Righetti
- **Venue：** Autonomous Robots（Springer, 2016）
- **arXiv：** <https://arxiv.org/abs/1410.7284>
- **PDF：** <https://arxiv.org/pdf/1410.7284>
- **类型：** paper / whole-body-control / centroidal-momentum
- **入库日期：** 2026-11-07
- **核对版本：** arXiv PDF 全文（21 页）
- **核对范围：** 读完 §1 引言、§2 层级逆动力学（建模假设、运动方程、接触约束 / CoP / 摩擦锥、任务与约束求解器、运动方程分解 §2.3、第一优先级解 §2.4）、§3 线 / 角动量调节（动量模型 §3.1、LQR §3.2）与 §7 结论、参考文献中 Orin & Goswami 2008 条目；实验章节（§4–§6，平衡 / 跟踪 / 接触切换，Sarcos 液压人形实验数据）仅浏览结构，未逐项核对数值。
- **通读：** 已完成
- **开源状态：** 论文无官方仓库（未核对）。

## 关键内容摘录

- **运动方程（式 1）：** $M(q)\ddot q+N(q,\dot q)=S^\top\tau+J_c^\top\lambda$；$S=[I_{n\times n}\ 0]$ 表示欠驱动，浮动基为前 / 后 6 个未驱动自由度；$N$ 含科氏、离心、重力、摩擦；$\lambda$ 为广义接触力。接触固定约束 $J_c\ddot q+\dot J_c\dot q=0$（式 2），另有 CoP 在支撑多边形内、摩擦锥（用棱锥线性化）、力矩与关节限位。
- **运动方程分解（§2.3，论文贡献之一）：** 拆成驱动行 $M_u\ddot q+N_u=\tau+J_{c,u}^\top\lambda$（式 12）与最后 6 行浮动基行 $M_l\ddot q+N_l=J_{c,l}^\top\lambda$（式 13）；后者是整机 Newton–Euler 方程（动量变化 = 外力）；$\tau=M_u\ddot q+N_u-J_{c,u}^\top\lambda$（式 14）可消去，优化变量从 $2n+6+6c$ 降到 $n+6+6c$，使 QP 级联跑进 1 kHz。作者称相关工作通常忽略该分解。
- **质心动量矩阵（§3.1）：** $h=H_G(q)\dot q$（式 19，$h=[h_{lin};h_{ang}]$ 取在质心），$\dot h=H_G\ddot q+\dot H_G\dot q$（式 20）；Newton–Euler 形式 $\dot h=\begin{bmatrix}I&0&\cdots\\ [x_i-x_{cog}]_\times&I&\cdots\end{bmatrix}\lambda+\begin{bmatrix}mg\\0\end{bmatrix}$（式 21），质心速度 $\dot x_{cog}=h_{lin}/m$。动量变化率因此既可当运动学任务（$\ddot q$ 的函数）也可当力任务（$\lambda$ 的函数）；作者主张用式 21，免去数值求 $\dot H_G$ 带来的噪声。$H_G$ 引自 Orin & Goswami, *Centroidal Momentum Matrix of a humanoid robot: Structure and Properties*, IROS 2008。
- **控制：** 对式 21 线性化设计 LQR 反馈，得到线 / 角动量耦合的最优增益，代替各接触构型手调 PD；层级 QP 保证物理一致性为最高优先级。
- **结论与局限：** 实验在 Sarcos 液压力矩控制人形上展示单脚 / 双脚平衡与推扰恢复；作者指出未来需接更高层规划器以支持行走；模型误差与计算复杂度是实际部署的限制。

**对 wiki 的映射：** [`centroidal-dynamics.md`](../../wiki/concepts/centroidal-dynamics.md)、[`floating-base-dynamics.md`](../../wiki/concepts/floating-base-dynamics.md)、[`contact-dynamics.md`](../../wiki/concepts/contact-dynamics.md)
