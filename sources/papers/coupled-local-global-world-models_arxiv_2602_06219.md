# Coupled Local and Global World Models for Efficient First Order RL（arXiv:2602.06219）

> 来源归档（paper）

- **标题：** Coupled Local and Global World Models for Efficient First Order RL
- **作者：** Joseph Amigo*、Rooholla Khorrambakht*、Nicolas Mansard、Ludovic Righetti（NYU Machines in Motion Lab；LAAS-CNRS / ANITI Toulouse）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2602.06219>（v1，2026-02-05，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2602.06219v1>
- **代码：** 论文正文未给出代码仓库链接
- **入库日期：** 2026-11-12
- **一句话说明：** 在 DMO（解耦前向 / 反向模型的一阶梯度 MBRL）框架上，用大型扩散世界模型（全局，负责前向展开）加 DreamerV3 RSSM（局部，负责潜空间梯度），在真机采集的 play 数据上学世界模型并在其中训练策略；Push-T 与 Go2 四足推方块两个任务。
- **核对范围：** 通读 arXiv v1 PDF 全部 9 页正文（摘要、引言、相关工作、方法 III-A 至 III-F、实验 IV-A 至 IV-E、结论、致谢、参考文献）、算法 1 与表 I、II，并查看算法页与图 2 的图像；论文无附录与补充材料（正文内未提及）；未核对被引工作 DMO（CoRL 2025）、DIAMOND、DreamerV4 的原文；图 3 的曲线数值仅依据图注与正文，未逐点读取。
- **通读：** 已完成

## 核心摘录

1. **方法（III-B、III-E）：** 前向用全局模型 f→ 生成轨迹，反向用局部模型 f←（RSSM）在前向下一状态处求雅可比，策略梯度不穿过扩散网络也不穿过编码器。
2. **全局模型（III-D）：** Push-T 用 DIAMOND（CNN 扩散，3 个扩散步，4 帧上下文 = 0.8 s）；自我中心推方块用 DreamerV4 式 Transformer 潜空间扩散（shortcut learning、diffusion forcing、轴向注意力、KV cache），上下文 96 帧 = 19.2 s，RL 时用 32 帧历史预填 KV cache（III-F）。
3. **奖励（III-D2）：** Push-T 用基于 Bradley-Terry 对比损失、仅用被动视频训练的目标图像奖励；推方块用示范 / play 二元标签（遥操作者按钮）作为奖励，在世界模型上加奖励 token。
4. **设置（IV-A/B）：** Flexiv Rizon-10S 桌面机械臂（Push-T，4 小时 play 数据）；Unitree Go2（自我中心推方块，12 小时随机交互数据）。
5. **结果（IV-D/E、表 I、II、图 3）：** Push-T 真机 9/10 成功（PPO 1/10，放宽判据 4/10；No Diffusion 0/10）；图 3 图注 DMO 8M 样本 vs PPO 40M（Push-T），4M vs 25M（推方块）；推方块 3 个评估回合 DMO 平均 91.3 步，PPO 134.3 步，ACT 无法到达终止态。

## 对 wiki 的映射

- [paper-coupled-local-global-world-models-fog-rl](../../wiki/entities/paper-coupled-local-global-world-models-fog-rl.md)
