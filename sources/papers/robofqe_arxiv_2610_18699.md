# RoboFQE: Rollout-Free Evaluation of Visual Robot Policies from Offline Trajectories（arXiv:2610.18699）

> 来源归档（ingest）

- **标题：** RoboFQE: Rollout-Free Evaluation of Visual Robot Policies from Offline Trajectories
- **arXiv：** <https://arxiv.org/abs/2610.18699>（HTML：<https://arxiv.org/html/2610.18699>）
- **项目页：** <https://yfdeng.com/robofqe/>（归档：[robofqe-project.md](../sites/robofqe-project.md)）
- **代码：** 项目页 "Code coming soon"，无仓库链接（2026-10-17 核查）
- **作者：** Yunfu Deng、Sangwoo Shin、Brahma S. Pavse、Josiah P. Hanna
- **机构：** University of Wisconsin–Madison 计算机系（arXiv HTML 与项目页）；项目页标注 CoRL 2026
- **入库日期：** 2026-10-17
- **文章来源：** 具身智能小站 2026-10-17 公众号文章
- **一句话说明：** 用拟合 Q 评估（FQE）+ 预训练视觉编码器 + 学得的终端成功标注器，仅凭离线轨迹估计目标策略回报并保持策略排序，无需新 rollout，也避免视频世界模型的自回归误差累积。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 页面为 "Project page in progress"，"Code coming soon"，无 GitHub / Hugging Face。
- **结论：** **宣称将开源 / 未列链接**。

## 摘录 1：方法（arXiv HTML）

- 目标：不建模行为策略密度，直接对离线轨迹做 FQE；平方 Bellman 损失 + 周期同步的目标网络。
- 奖励：每条轨迹一个二值终端成功标签，对所有目标策略共享。
- 视觉编码器：DINOv3 并在离线数据上微调；文本用预训练文本编码器。
- 终端成功标注器：Large Reward Model（VLM）；LIBERO 上微调后准确率 97.4%，真机零样本 83.3%。
- 价值估计：各任务初始输入上的平均 Q 值。

**对 wiki 的映射：** [`paper-robofqe.md`](../../wiki/entities/paper-robofqe.md)；与 [sim-vs-real 评测差距](../../wiki/concepts/sim-vs-real-eval-gap.md)、[仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md) 互链。

## 摘录 2：实验（arXiv HTML）

- **LIBERO**：四个 suite（Spatial/Object/Goal/Long，horizon 220/280/300/520），每 suite 10 任务、750 条离线轨迹（每任务 50 示范 + 25 rollout）；目标策略 ResNet-T，5 seeds。绝对相对误差 ARE：RoboFQE 3.40 / 4.56 / 2.34 / 5.63%；Long 上世界模型 54.37–69.18%。与 Vπ 的 Pearson r = +0.96/+0.91/+0.90/+0.88，MMRV = 0.03/0.02/0.05/0.07。
- 从零训练编码器的 ARE：9.35 / 13.89 / 5.82 / 32.09%。
- **CALVIN ABC→D**（X-VLA）：真值平均链长 4.32，RoboFQE 3.61，WorldGym 1.04，Cosmos 2.5-AC 1.48。
- **真机**（SO-101 叠方块，3 任务，150 条遥操作 + 100 条 BC rollout；SmolVLA / π0.5 / 多任务 BC 真值成功率 58.5% / 81.0% / 33.0%）：RoboFQE ARE 7.5%、r +0.90、MMRV 0.11；Ctrl-World 10.3% / +0.49 / 0.19；WorldGym 17.8% / +0.44 / 0.40；Cosmos 2.5-AC 24.5% / +0.68 / 0.28。
- 作者局限：依赖离线数据覆盖、成功检测器误差敏感、条件于当前输入而非完整历史；零成功率的随机策略仍得到正价值估计（Spatial 上 r 从 +0.93 降到 +0.54）。

**对 wiki 的映射：** 评测、结论与局限小节。
