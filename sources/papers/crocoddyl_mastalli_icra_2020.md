# Crocoddyl：多接触最优控制框架（ICRA 2020）通读

> 来源归档（原文通读 · 支撑 `wiki/concepts/hqp.md`、`wiki/queries/mpc-solver-selection.md`）

- **标题：** Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control
- **作者：** Carlos Mastalli, Rohan Budhiraja, Wolfgang Merkt, Guilhem Saurel, Bilal Hammoud, Maximilien Naveau, Justin Carpentier, Ludovic Righetti, Sethu Vijayakumar, Nicolas Mansard
- **类型：** paper
- **链接：** <https://arxiv.org/abs/1909.04947>（PDF：<https://arxiv.org/pdf/1909.04947>）
- **版本 / 日期：** arXiv v2，2020-03-11
- **入库日期：** 2026-11-07
- **核对范围：** 通读全文 §I–§V（引言、多接触最优控制与接触作为完整约束、FDDP 算法的前向 / 后向 / 步长接受、结果与运行时间基准、结论）；参考文献仅浏览。未运行代码，未读仓库文档。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位：** 开源多接触最优控制框架，**给定预定义的接触序列**，求状态轨迹与控制策略；用稀疏解析导数、问题结构与数据共享提速，浮动基座建模为 SE(3)。
- **接触模型：** 把接触作为完整（holonomic）约束放进动力学（式 2），接触力由状态与力矩确定；该式**忽略摩擦锥与关节限位**，因此可用无约束 DDP 求解；不等式可经惩罚、有效集或增广拉格朗日加入。冲击（接触增益阶段）用冲击动力学（式 6）。
- **FDDP（Feasibility-driven DDP）：** 两处改动——后向传播接受不可行的状态-控制轨迹；前向 rollout 在早期"探索"迭代里保持间隙（gaps）不闭合；步长接受用 Goldstein 条件（$b_1=0.1,b_2=2$）。间隙收缩与"仅等式约束多重打靶的 KKT 牛顿法"一致，且不增加决策变量。
- **结果：** 双足 / 四足步态与跳跃、前空翻（ANYmal、iCub 等）在 12–36 次迭代内求出，用朴素不可行的 warm start；步态通常 10–34 次迭代、总计算 < 0.5 s；积分步长一般 $10^{-2}$ s（双足行走 $3\times10^{-2}$ s），节点数 60–115，对应视界 0.6–3 s；每次迭代的计算速率最高 859.6 Hz（jump-4f，i9-9900K，60 节点，多线程）。接触数不影响单次迭代耗时，随节点数线性增长。
- **限制：** 评测中**刻意忽略摩擦锥与力矩限**（作者说明可经二次惩罚加入，见补充视频）；未来工作为不等式约束（力矩限、摩擦锥）下的可行性。
- **与 WBC 的关系：** 引言称，"WBC 跟踪质心动力学规划"的框架难以正确处理机器人姿态（尤其腾空相）；本文属于**全身轨迹优化 / MPC 的 OC 求解器**，**不是 HQP 型**优先级求解器。

## 对 wiki 的映射

- [`wiki/concepts/hqp.md`](../../wiki/concepts/hqp.md)、[`wiki/queries/mpc-solver-selection.md`](../../wiki/queries/mpc-solver-selection.md)

## 参考来源

- Mastalli, C. et al. (2020). Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control. ICRA 2020. <https://arxiv.org/abs/1909.04947>
