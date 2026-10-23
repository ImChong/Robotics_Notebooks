# Player-1 论文来源归档

## 元数据

- **题目：** Player-1: Learning to Play from Synthetic Experience
- **作者：** Mingyu Ouyang、Yiran Wang、Siyuan Hu、Xingyilang Yin、Huiqiang Sun、Guangzhi Wang、Junfu Pu、Cheng Cheng、Honghao Chen、Ying Shan、Mike Zheng Shou、Hwee Tou Ng
- **机构：** ARC Lab, Tencent IEG；National University of Singapore（作者单位以论文署名为准）
- **预印本：** https://arxiv.org/abs/2610.20584
- **项目页：** https://player-1-project.github.io/
- **代码仓库：** https://github.com/TencentARC/Player-1
- **日期：** 2026-10-14

## 方法摘要

Player-1 将游戏控制视为计算机使用模型的后训练任务：先用 6,044 条从 172 款游戏整理的轨迹做监督微调，再在 Player-1-Gym 的 16 个合成 2D/3D 环境中用 PPO 强化学习。动作空间加入相对鼠标移动、持续按键与键鼠复合动作；环境状态直接生成可验证奖励，并随机化外观、相机、场景和控制动态。

论文报告 Player-1-9B 在 MineDojo-300 上的成功率由 SFT 的 10.54% 提升至 RL 后的 38.33%，提高 27.79 个百分点。作者还报告在 ViZDoom-10 与 GameWorld 上相对 SFT 的提升。Watch Dogs 2 是一个训练中未见商业游戏的定性案例，展示步行导航、车辆接近、上车与驾驶；论文未提供该游戏上的量化成功率。

## 结果与边界

- MineDojo-300 包含 300 个 Minecraft 任务，论文报告每任务 8 次试验。
- 38.33% 是 Player-1-9B 的 RL 结果；最强外部基线 GameWAM 为 12.63%，评测协议以论文为准。
- 对 Watch Dogs 2 的演示仅作为定性迁移例子，不能表述为有统计置信度的成功率或普遍泛化证明。
- 截至 2026-10-23，官方仓库仍将 Player-1-Data、Player-1-Gym、模型权重和训练/评测代码列为待发布；论文与项目页已公开。

## 对 wiki 的映射

- 详情节点：[Player-1](../../wiki/entities/paper-player-1.md)
- 对照阅读：[GameHorizon Suite](../../wiki/entities/paper-gamehorizon-suite.md)：侧重多时间尺度数据与评测；Player-1 侧重合成经验强化学习。
