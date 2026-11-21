# GAE: General Action Expert for Real-Time Humanoid Teleoperation

- **类型：** 论文
- **作者：** Yuefan Wang、Huaicheng Zhou、Xiao He、Zhijie He、Mingchuan Yang、Huayi Zhang、Li Chai、Jinxin Liu、Donglin Wang
- **机构：** 西湖机器人（Westlake Robotics）、西湖大学（Westlake University）
- **arXiv：** <https://arxiv.org/abs/2609.34233>
- **PDF：** <https://arxiv.org/pdf/2609.34233>
- **项目页：** <https://wangyf0928.github.io/gae-wlrobotics/>
- **日期：** 2026-09-28（arXiv v1）；2026-10-02（入库）
- **核对范围：** 通读 `arXiv:2609.34233v1`（2026-09-28）正文 Sections I–V 及 Appendix A–C（Table I–IV、Fig. 1–14），并核对官方项目页与 `westlakerobotics/GAEApp-API@cdcfd67`；未运行闭源二进制或复现训练。
- **通读：** 已完成
- **关联 wiki：** [GAE 全身遥操作](../../wiki/entities/paper-gae-general-action-expert.md)
- **区别：** 本文 General Action Expert，不是仓库既有的 Geometric Autoencoder（同缩写 GAE）。

## 核心摘录与 wiki 映射

1. **数据规模：** 汇集视频、动画与动捕，转换到统一 SMPL 表示，经左右镜像、时序拼接（stitching）、上下身重组（recomposition）形成超过 1 万小时动作，并拟合人形比例 SMPL 代理骨架。映射到 [论文实体页的数据与方法](../../wiki/entities/paper-gae-general-action-expert.md)。
2. **生成器—执行器：** 特权生成器（含 100 帧历史 + 32 帧未来目标，MoE Transformer 约 852M 参数）先在干净仿真（Isaac Lab）中把形态拟合的人类参考转为物理可行的机器人轨迹；执行器（约 852M 参数，观测 100 帧本体与人体目标）在逐步增强的课程域随机化（Table III 13 项）中学习跟踪该轨迹。训练结合 Multi-Critic（5 个价值头分别估计惩罚与位置/线速度/角速度/旋转跟踪回报）与未来目标/状态预测辅助损失（$L = L_{\text{PPO}} + \lambda_t L_{\text{target}} + \lambda_s L_{\text{state}}$）。执行器输入仍来自原始人类动作，生成轨迹只用于奖励目标，部署不需在线生成器。映射到 [训练流程](../../wiki/entities/paper-gae-general-action-expert.md)。
3. **延迟预判：** 执行器以延迟的人体动作和延迟参数为条件，使用 RoPE 时间索引偏移调整预判时域。50 Hz 下每帧 20 ms；论文测试 0–100 ms（$\tau=0\dots 5$），真机展示 80 ms（$\tau=4$）与 1,300 km 远程遥操（北京→杭州）及 VR 稀疏追踪接入。映射到 [同步机制](../../wiki/entities/paper-gae-general-action-expert.md)。
4. **评测边界与跨本体/自指发现：** 相比 SONIC，在 Easy / Medium / Hard 集上的序列成功率分别为 97.3 / 96.9 / 93.9%（SONIC 95.4 / 85.2 / 77.7%）；Hard 集预判 100 ms 时成功率为 92.1%（MKPE 由 0.045 升至 0.056 m）。真机主实验为 Unitree G1；Westlake O1 经形态专用微调后迁移；Appendix C 进一步展示 G1、Westlake O1 与 ENGINEAI PM01 三台异构人形由同一操作员同步控制，以及无外接目标时用 `FK(robot state)` 自指闭环维持行走/自由舞动极限环。映射到 [评测与局限](../../wiki/entities/paper-gae-general-action-expert.md)。

## 开放状态

截至 2026-10-02，[项目页](../sites/gae-general-action-expert.md) 提供论文与视频，但未列官方 GitHub、权重或数据集下载地址；不能把页面模板来源仓库当作算法代码。
