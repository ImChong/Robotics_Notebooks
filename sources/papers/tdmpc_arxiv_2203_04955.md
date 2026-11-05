# TD-MPC: Temporal Difference Learning for Model Predictive Control（arXiv:2203.04955）

> 来源归档（ingest）

- **标题：** Temporal Difference Learning for Model Predictive Control
- **短名：** TD-MPC（学到的模型称 TOLD，Task-Oriented Latent Dynamics）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2203.04955>（提交 2022-03-09）
- **PDF：** <https://arxiv.org/pdf/2203.04955>
- **项目页 / 代码：** <https://nicklashansen.github.io/td-mpc>（论文给出的实现入口；本次未读官方源码）
- **作者：** Nicklas Hansen, Xiaolong Wang, Hao Su
- **发表：** ICML 2022（LeRobot 文档的 BibTeX 标注）
- **LeRobot 实现：** `src/lerobot/policies/tdmpc/`（固定提交 9e5ee9859480e425a97a6836bb3fee479539beb5）。**注意：** 源码文件头与训练损失实际实现的是 FOWM 的变体（见下），不是原版 TD-MPC 损失
- **相关论文（仅读摘要）：** *Finetuning Offline World Models in the Real World*（FOWM），[arXiv:2310.16029](https://arxiv.org/abs/2310.16029)，Yunhai Feng, Nicklas Hansen, Ziyan Xiong, Chandramouli Rajagopalan, Xiaolong Wang，2023-10-24
- **入库日期：** 2026-11-05
- **一句话说明：** 用 TD 学习联合训练潜空间动力学、奖励和终端价值函数，再用 MPPI 在潜空间做短视距规划。

## 核心摘录（面向 wiki 编译，读自 arXiv PDF）

- **摘要：** 用学到的任务导向潜空间动力学模型做短视距局部轨迹优化，并用学到的终端价值函数估计长期回报，两者通过时序差分学习联合训练；在 DMControl 与 Meta-World 的状态与图像连续控制任务上，样本效率与渐近性能优于此前工作。
- **§4 TOLD 目标：** 联合优化奖励预测（Eq.8）、价值预测（Eq.9，类 fitted Q-iteration）和潜状态一致性（Eq.10）三项；所有预测在潜空间递归完成，只有第一个观测经编码器，梯度沿时间反传；目标网络用在线网络的 EMA。
- **TD 目标与策略：** 直接对 max_a Q 规划代价太高，故额外学一个策略 π，最小化时间加权的 −ΣQ(z, π(sg(z)))（Eq.11，DDPG/SAC 式 actor 目标）；论文说复杂任务上 π 单独不如规划，但足以支持价值学习。
- **§5 实验设定：** 92 个任务（DMControl 与 Meta-World v2）；规划视距 H=5，规划 6 轮迭代，每轮采样 N=512 条轨迹（另有约 5% 来自 π），探索参数 ε 在前 25k 决策步线性从 0.5 退火到 0.05；用价值损失加权的优先经验回放。
- **FOWM 摘要（仅摘要）：** 在真机离线数据上预训练世界模型，再用带模型不确定性正则的规划器在线微调；评估于仿真与真机视觉运动控制任务。

## 对 wiki 的映射

- [paper-tdmpc-temporal-difference-model-predictive-control](../../wiki/entities/paper-tdmpc-temporal-difference-model-predictive-control.md)
- 后继 TD-MPC2 另有节点：[paper-td-mpc2](../../wiki/entities/paper-td-mpc2.md)（arXiv:2310.16828，不同论文）

## 当前提炼状态

- [x] TD-MPC 论文 §1、§4、§5 实现细节已读；FOWM 仅读 arXiv 摘要
- [x] LeRobot `policies/tdmpc/` 源码已读
- [ ] 官方仓库 `nicklashansen/tdmpc` 与 `fyhMer/fowm` 源码未读
