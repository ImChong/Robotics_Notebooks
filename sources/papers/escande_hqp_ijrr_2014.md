# Escande 等：Hierarchical Quadratic Programming（HQP，IJRR 2014）通读

> 来源归档（原文通读 · 支撑 `wiki/concepts/hqp.md`）

- **标题：** Hierarchical quadratic programming: Fast online humanoid-robot motion generation
- **作者：** Adrien Escande, Nicolas Mansard, Pierre-Brice Wieber（JRL-CNRS/AIST、LAAS-CNRS、INRIA）
- **类型：** paper
- **链接：** <https://hal.science/hal-00751924/document>；The International Journal of Robotics Research 33(7), 2014, 1006–1028，DOI 10.1177/0278364914521306
- **版本 / 日期：** HAL 作者稿（提交 2012-12-11），含 Part 1（基础）与 Part 2（实现与实验）
- **入库日期：** 2026-11-07
- **核对范围：** 通读 Part 1 摘要、引言、§2 层级任务背景（等式层级、Kanoun 级联 QP、De Lasa 简化）、§3 等式 HQP / HCOD 分解、§4 带不等式的有效集算法大意、§5 示例与结论；Part 2 读摘要、引言、背景、§5 实验（计算时间、warm start、HRP-2 三组实验）与结论。HCOD 增量更新的细节算法、复杂度与连续性 / 稳定性证明、附录（有效集基础、Givens 变换）仅浏览，未逐式核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（式 27）：** 严格层级的词典序最小化 $\mathrm{lex\,min}_{x,w_1..w_p}\{\|w_1\|,\dots,\|w_p\|\}$，s.t. $A_kx\le b_k+w_k$（$w_k$ 为各级松弛）。等式、双边界、区域约束都可写进来。
- **背景谱系：** Nakamura/Siciliano–Slotine 等式层级用递归零空间投影；**Kanoun 2011** 首先给出"QP 级联"支持不等式（慢：第 k 层约束在其后各层 QP 里反复求解）；**De Lasa 2010** 只允许最高层有不等式、更快但表达力受限；本文的贡献是 **单一专用求解器**（HCOD 分层完全正交分解 + 统一有效集搜索），避免级联中有效集来回增删。
- **数值：** 40 变量（人形速度控制的典型规模）纯等式层级约 0.2 ms（最高比经典方法快 10 倍）；带不等式"certified 1 ms"；HRP-2 实验中 97–98% 的控制周期有效集搜索无需迭代，平均约 1.1 ms/周期；结论称在普通 PC 上可 **200 Hz** 实时运行（摘要称有望到 1 kHz 量级）。
- **适用层面：** 论文的实例是 **逆运动学（速度级）**，用于 HRP-2 的举伞行走、避障到达等；并非直接求力矩。
- **与加权 QP 的区别：** 论文 §2.2 说明用权重矩阵 $Q$ 是"人为平衡"，严格层级则保证低优先级任务永不破坏高优先级任务的最优值。
- **局限 / 展望：** 带不等式时仍需有效集迭代：对比用的 De Lasa 式级联在实验 B 中单周期迭代最多 30 次（均值 4.5），本文求解器的实验 C 最多 7 次（均值 0.035 次/周期，97.8% 周期无需额外迭代，约 1.1 ms/周期）；为得到确定耗时可加 warm start 与迭代次数上限，代价是少数周期解为次优（实验 B 中无更新周期占比降为 96.6%，机器人行为无可感知变化）。作者的后续工作是继续降低计算量。

## 对 wiki 的映射

- [`wiki/concepts/hqp.md`](../../wiki/concepts/hqp.md)

## 参考来源

- Escande, A., Mansard, N., Wieber, P.-B. (2014). Hierarchical quadratic programming: Fast online humanoid-robot motion generation. *IJRR* 33(7).
