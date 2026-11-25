# SRL-MPC: Shape-Aware Reinforcement Learned Model Predictive Control

- **类型：** paper
- **作者：** Ruihua Han、Rui Gao、Zhe Liu、Xinyi Wang、Chang Chen、Shuai Wang、Qi Hao、Jia Pan、Hengshuang Zhao
- **机构：** 香港大学、南方科技大学、密歇根大学、中国科学院深圳先进技术研究院
- **版本与会议：** arXiv:2608.21175v1（2026-08-21）；[HKU SAIL 论文列表](https://sail.ai.hku.hk/publication.html?topic=S) 标为 NeurIPS 2026。以下论文数字来自预印本 v1。
- **原文：** [摘要与版本记录](https://arxiv.org/abs/2608.21175) · [HTML 含原图](https://arxiv.org/html/2608.21175v1) · [PDF](https://arxiv.org/pdf/2608.21175v1)
- **项目页：** [HKU SAIL Project](https://hku-sail.github.io/srl_mpc/)；预印本所列旧入口本次返回 404。
- **官方仓库：** [hku-sail/srl_mpc](https://github.com/hku-sail/srl_mpc)（公开项目材料，未发布算法源码）
- **首次入库 / 本次核对：** 2026-08-25 / 2026-11-25
- **核对范围：** 英文 PDF 正文 §§1–6，图 1–6、表 1–3、技术附录 A.1–A.6（含图 7–8、表 4–7）；对照项目页多种子结果、OOD 补充和仓库文件树。
- **通读：** 已完成

## 核心摘录

1. **几何与可行性分开。** 用支撑函数表达占据集合最短距离。Q1 用几何最短线提取分离方向与距离；Q2 固定几何量，保留运动学和输入约束，对二阶 HOCBF 的负残差施加平方软罚。不要把原始耦合问题的硬屏障等同于实际求解器的约束。映射：[SRL-MPC](../../wiki/entities/paper-srl-mpc.md) · [MPC](../../wiki/methods/model-predictive-control.md)。
2. **强化学习只调参。** PPO 用当前速度、三项参数和预测时域内近邻 GSF 编码，生成有界参数增量；底盘输入是优化控制序列的第一项。映射：[SRL-MPC](../../wiki/entities/paper-srl-mpc.md) · [强化学习](../../wiki/methods/reinforcement-learning.md)。
3. **实验口径。** 表 1 的 25 机器人成功率为 92/100；项目页的 86.7% ±0.6 pp 是三条独立训练策略的均值和样本标准差。表 3 和附录 A.2 说明感知、执行延迟和求解计算的边界。映射：[SRL-MPC 评测](../../wiki/entities/paper-srl-mpc.md#评测) · [项目页归档](../sites/srl-mpc-hanruihua.md) · [官方仓库归档](../repos/srl-mpc.md)。
