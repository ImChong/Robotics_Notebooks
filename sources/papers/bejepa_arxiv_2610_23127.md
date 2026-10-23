# BeJEPA: A Goal-Conditioned Humanoid Behavior World Model via Unsupervised RL（arXiv:2610.23127）

> 来源归档（paper）；以下摘要按 arXiv v1 原文整理。

- **标题：** BeJEPA: A Goal-Conditioned Humanoid Behavior World Model via Unsupervised RL
- **类型：** paper / humanoid whole-body control + behavior world model
- **arXiv：** <https://arxiv.org/abs/2610.23127>（HTML：<https://arxiv.org/html/2610.23127v1>）
- **项目页：** <https://ylyem9x.github.io/BeJEPA-project/>
- **作者：** Liyun Yan, Haoyang Li, Yang Zhang, Jialing Zhu, Zhanxiang Cao, Zihao Zhou, Keqi Zhu, Yizhi Chen, Huang Chuxuan, Shengcheng Fu, Zhong Jiangwei, Qiao Sun, Chen Zhang, Yue Gao
- **机构：** 上海交通大学、上海创智学院、浙江大学、同济大学、复旦大学、联想研究院机器人实验室
- **提交日期：** 2026-10-16
- **一句话说明：** 用无监督 RL 从人类动作先验与机器人交互中联合学习 Dynamic 一步物理预测器和 Successor 长期行为预测器，在同一潜在空间里支持人形机器人的全身姿态、世界位置与朝向目标控制。

## 摘要级要点（与 arXiv 原文一致）

- **问题：** 以根坐标系为中心的动作跟踪器能复现身体姿态，却可能在受扰后偏离目标世界位置；如果只匹配当前姿态，机器人迈步回到目标位置时还可能被判作姿态更差。
- **共享表征：** JEPA 编码器把身体构型、根位置和朝向一起映射为状态嵌入；目标嵌入定义姿态与世界位置的联合目标。
- **双预测器：** Dynamic 根据当前嵌入和动作预测下一步状态嵌入，学习动作造成的局部物理变化；Successor 以目标为条件，汇总策略未来会访问的状态特征，提供长期行为价值。
- **训练信号：** 两个预测器从在线机器人转移数据与回放缓冲区联合学习；人类动作先验用于提供目标与运动先验，物理执行惩罚约束动作。观测历史编码器蒸馏特权状态表征供策略部署。
- **控制组合：** Actor 通过“一步 Dynamic 预测 + 一个 Successor 未来摘要”评估动作，不需要沿动态模型滚动很长的想象轨迹。跟踪时，固定 Dynamic 模型还能用于在线动作残差优化，对齐空间误差和时间参考。
- **评测：** 在 AMASS 与 LAFAN 的 20 秒跟踪片段上，BeJEPA-MPC (Balanced) 相比 SONIC 的全局位置误差更低；摘要报告相对 BFM-Zero（输入世界位姿）降低 61.7%，相对 SONIC 降低 58.7%。论文同时在 Unitree G1 真机展示世界位置/朝向目标、倒地起身、外力扰动恢复和连续切换目标。
- **复现状态：** 截至 2026-10-23，本次无法核验项目页源码区；arXiv 页面未列官方仓库链接，因此暂不把代码开放状态判定为已开源。见[项目页归档](../sites/bejepa-project-page.md)。

## 对 wiki 的映射

- 沉淀实体页：[paper-bejepa](../../wiki/entities/paper-bejepa.md) — 归纳目标条件行为世界模型、双预测器与长时程跟踪评测。
- 关联阅读：[BFM-Zero](../../wiki/entities/paper-bfm-zero.md) — 对照 world-pose 输入的 Forward–Backward BFM 基线。

## 主要原始资料

- Yan et al., *BeJEPA: A Goal-Conditioned Humanoid Behavior World Model via Unsupervised RL*, arXiv:2610.23127, 2026-10-16.
