# Del Prete 等：Task Space Inverse Dynamics（TSID 原论文）通读

> 来源归档（原文通读 · 支撑 `wiki/concepts/tsid.md`、`wiki/formalizations/tsid-formulation.md`）

- **标题：** Prioritized Motion-Force Control of Constrained Fully-Actuated Robots: "Task Space Inverse Dynamics"
- **作者：** Andrea Del Prete, Francesco Nori, Giorgio Metta, Lorenzo Natale（LAAS-CNRS / IIT）
- **类型：** paper
- **链接：** <https://arxiv.org/abs/1410.3863>（PDF：<https://arxiv.org/pdf/1410.3863>）；期刊版 Robotics and Autonomous Systems 63 (2015)，DOI 10.1016/j.robot.2014.08.016
- **版本 / 日期：** arXiv v1，2014-10-14
- **入库日期：** 2026-11-07
- **核对范围：** 通读全文 §1–§9（含 §2 关键思想、§3 related works 与 Table 1、§4 UF、§5 WBCF、§6 动机、§7 TSID 推导与力控制扩展、§8 两组 CoMan 仿真实验与 Table 3–4、§9 结论与局限）；参考文献列表仅浏览。未复现仿真，未读其后续的浮动基座扩展论文。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **适用对象：** 论文只处理 **全驱动（fully-actuated）** 机器人，运动方程 $M\ddot q + h = \tau$（浮动基座 / 欠驱动留给"后续论文"，§3、§9.1）；不含不等式约束（关节限位、力矩界），也不规划，只保证瞬时（局部）最优（§9.1 的三条局限）。
- **评价四准则（§3）：** soundness（低优先级任务不影响高优先级）、optimality（在不破坏高优先级的前提下最小化每个任务误差）、capabilities（力控制 / 欠驱动 / 不等式）、efficiency（（伪）逆与矩阵乘法次数）。
- **任务加速度（§2.1）：** $\ddot x^* = \ddot x_r + K_d(\dot x_r-\dot x)+K_p(x_r-x)$；任务经二次微分写成 $J\ddot q+\dot J\dot q=\ddot x$。
- **TSID 的优先级形式（式 8–10）：** 从最高优先级 $N$ 到最低优先级 0（姿态任务 $\ddot q_p^*=K_p(q_p-q)-K_d\dot q$）依次解"约束于不破坏更高优先级任务"的最小化；解为 $\tau^*=M\ddot q_0+h$，其中 $\ddot q_i=\ddot q_{i+1}+N_{p(i)}^W(J_iN_{p(i)}^W)^{\dagger W}(\ddot x_i^*-\dot J_i\dot q-J_i\ddot q_{i+1})$。因为有姿态任务，**最终 $\tau^*$ 与权重矩阵 $W$ 无关**，取 $W=I$ 时零空间投影器为正交投影，运动学（先解 $\ddot q$）与动力学（再算 $\tau$）完全解耦。
- **求解方式：** **伪逆 + 递归零空间投影（解析解），不是 QP / 数值优化。** 力矩可用 RNEA 以 $O(n)$ 计算，无需显式 $M$ 或任务空间惯量 $\Lambda$。
- **力控制（式 11–14）：** 刚性接触 $M\ddot q+h-J_c^Tf=\tau$、$J_c\ddot q=b$；刚性接触约束是 **最高优先级** 的"任务"；$\tau^*=M(\ldots)+h-J_c^Tf^*$。
- **对比实验（§8，CoMan 23 自由度，固定基座，线性弹簧阻尼接触）：** Test 1：TSID 与 WBCF（Sentis–Khatib）误差几乎一致（F 0.1 N、T2 0.4 mm、T1 0.1°、T0 7.1°），平均计算时间 TSID 0.24 ms vs WBCF 0.64 ms（约 2.6×）；UF 非主任务误差大（T2 36.8 mm、T1 30.1°）。Test 2（不可行任务层级）结论相同（TSID 0.25 ms、WBCF 0.67 ms、UF 0.26 ms）。阻尼伪逆 $\lambda=0.02$、$\sigma_{min}=2.5\times10^{-8}$，$K_p=10\,s^{-2}$、$K_d=5\,s^{-1}$。
- **局限（§9.1）：** ①无不等式约束；②只瞬时最优，需 MPC 类方法扩展；③仅全驱动。作者指出带不等式的框架通常靠"线性化 + 有效集把不等式变等式"迭代求解本文的简化问题。

## 对 wiki 的映射

- [`wiki/concepts/tsid.md`](../../wiki/concepts/tsid.md)、[`wiki/formalizations/tsid-formulation.md`](../../wiki/formalizations/tsid-formulation.md)

## 参考来源

- Del Prete, A., Nori, F., Metta, G., Natale, L. (2015). Prioritized motion–force control of constrained fully-actuated robots: "Task Space Inverse Dynamics". *Robotics and Autonomous Systems* 63, 150–157. <https://arxiv.org/abs/1410.3863>
