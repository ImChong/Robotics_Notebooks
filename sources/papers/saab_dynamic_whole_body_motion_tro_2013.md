# Saab 等：刚性接触与单边约束下的动态全身运动生成（通读）

> 来源归档（原文通读 · 支撑 `wiki/concepts/tsid.md`、`wiki/formalizations/tsid-formulation.md`）

- **标题：** Dynamic Whole-Body Motion Generation under Rigid Contacts and other Unilateral Constraints
- **作者：** Layale Saab, Oscar E. Ramos, François Keith, Nicolas Mansard, Philippe Souères, Jean-Yves Fourquet
- **类型：** paper
- **链接：** <https://hal-lirmm.ccsd.cnrs.fr/lirmm-00831097/document>；IEEE Transactions on Robotics 29(2), 2013，DOI 10.1109/TRO.2012.2234351
- **版本 / 日期：** HAL 作者稿（2012 年 12 月接收）
- **入库日期：** 2026-11-07
- **核对范围：** 通读摘要、§I 引言、§II 逆运动学回顾、§III 逆动力学、§IV 接触约束下的逆动力学（接触力并入、刚性点接触互补条件、HQP 层级）、§V 平面接触的降维表述、§VII 实验设置与结论 §VIII；§VI 控制律鲁棒性与附录推导仅浏览要点，未逐式核对；参考文献未读。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **目标：** 把"任务函数 + 严格优先级"从逆运动学扩展到 **完整多体动力学**，并容纳任意以状态 / 控制变量写成的等式或不等式约束（关节限位、力矩限、接触稳定）。
- **浮动基座 + 接触的动力学（式 21–22）：** $A\ddot q+b+J_c^\top f=S^T\tau$，刚性接触 $J_c\ddot q=-\dot J_c\dot q$；$S^T$ 使前 6 个（浮动基座）自由度无直接力矩。
- **刚性点接触条件（式 27–29）：** 法向加速度 $\ddot x\ge0$、法向力 $f^\perp\ge0$、互补 $\ddot x f^\perp=0$。只保证零运动约束（22）而不显式要求 $f^\perp\ge0$ 会生成不可行的动态运动。
- **HQP 写法：** 决策变量 $u=(\tau,\ddot q,f)$，层级 (21) ≺ (22) ≺ (28) ≺ (14)，即动力学方程 → 接触不动 → 法向力非负 → 任务。前两层恒可行，第三层可能不可行（"强接触不稳定"），此时松弛变量非零，解动力学不一致，需改变接触模式。
- **平面接触降维（§V）：** 用接触体上的 6D 广义力 $\phi$ 代替每个接触点一个力变量，使 HQP 代价不随接触点数线性增长；在平地站立情形，接触条件等价于 ZMP 在支撑多边形内。
- **实验（§VII）：** HRP-2；仿真器 AMELIF 以计算出的力矩前向积分，控制每 1 ms 更新；序列化坐扶手椅与动态接触切换，轨迹在真机上以位置控制回放。
- **局限（§VIII）：** 运动为 **离线生成**，算法"接近但尚未实时"，未做闭环；下一步需加速求解并引入基于传感器的控制。

## 对 wiki 的映射

- [`wiki/concepts/tsid.md`](../../wiki/concepts/tsid.md)、[`wiki/formalizations/tsid-formulation.md`](../../wiki/formalizations/tsid-formulation.md)

## 参考来源

- Saab, L. et al. (2013). Dynamic whole-body motion generation under rigid contacts and other unilateral constraints. *IEEE T-RO* 29(2).
