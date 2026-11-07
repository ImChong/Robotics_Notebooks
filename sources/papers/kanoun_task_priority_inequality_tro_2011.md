# Kanoun 等：把任务优先级框架推广到不等式任务（通读）

> 来源归档（原文通读 · 支撑 `wiki/concepts/hqp.md`）

- **标题：** Kinematic control of redundant manipulators: generalizing the task priority framework to inequality tasks
- **作者：** Oussama Kanoun, Florent Lamiraux, Pierre-Brice Wieber
- **类型：** paper
- **链接：** <https://hal.science/hal-00486755/document>；IEEE Transactions on Robotics 27(4), 2011, 785–792，DOI 10.1109/TRO.2011.2142450
- **版本 / 日期：** HAL 作者稿 v2（2011-06-01）
- **入库日期：** 2026-11-07
- **核对范围：** 通读摘要、引言、§IV 最小二乘解的性质（命题 4.1/4.2）、§V 等式与不等式系统的优先级化（公式 39–48、Algorithm 1）、§VI 两组 HRP-2 仿真；§II–§III 等式层级回顾与不等式写法仅浏览；参考文献未读。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题：** 速度级逆运动学中，等式任务可用零空间投影按优先级求解，但不等式（关节限位、避障）传统上被势场转成更严苛的等式；本文提出 **QP 序列** 来保留不等式。
- **算法（§V）：** 在第 k 级求解 $\min \tfrac12\|A_kx-b_k\|^2+\tfrac12\|w\|^2$ s.t. $C_kx-w\le d_k$、$w\ge0$，且 $x\in S_k$；下一级的可行集 $S_{k+1}$ 由本级最优点确定——等式任务加 $A_kx=A_kx_k^*$，不等式任务把已满足的约束保持为不等式、未满足的取等。因而 $S_{k+1}\subseteq S_k$：上层结果在下层被严格保持。
- **纯等式时退化：** 若只有等式且最后取最小范数，该过程退化为经典的零空间投影算法。
- **实验（§VI）：** HRP-2 的 28 个关节速度；场景 1 在横杆后伸手取物，把"视线"任务与"伸手"放同一优先级会得到折中姿态而够不着，降低视线优先级后手可到达；调节参数 $\rho_2\approx0.01$ 以免任务冲突时不稳定。QP 用通用求解器，计算时间"应在未来改进"。
- **局限：** 级联中每级 QP 变大、第一层约束被重复求解 p 次（Escande 2014 指出的低效点）；文中原型不满足实时。

## 对 wiki 的映射

- [`wiki/concepts/hqp.md`](../../wiki/concepts/hqp.md)

## 参考来源

- Kanoun, O., Lamiraux, F., Wieber, P.-B. (2011). Kinematic control of redundant manipulators: generalizing the task-priority framework to inequality tasks. *IEEE T-RO* 27(4).
