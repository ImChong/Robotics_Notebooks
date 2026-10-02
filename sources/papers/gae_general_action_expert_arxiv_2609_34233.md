# GAE: General Action Expert for Real-Time Humanoid Teleoperation

- **类型：** 论文
- **作者：** Yuefan Wang、Huaicheng Zhou、Xiao He、Zhijie He、Mingchuan Yang、Huayi Zhang、Li Chai、Jinxin Liu、Donglin Wang
- **机构：** 西湖机器人（Westlake Robotics）、西湖大学（Westlake University）
- **arXiv：** <https://arxiv.org/abs/2609.34233>
- **PDF：** <https://arxiv.org/pdf/2609.34233>
- **项目页：** <https://wangyf0928.github.io/gae-wlrobotics/>
- **日期：** 2026-09-28（arXiv v1）；2026-10-02（入库）
- **关联 wiki：** [GAE 全身遥操作](../../wiki/entities/paper-gae-general-action-expert.md)
- **区别：** 本文 General Action Expert，不是仓库既有的 Geometric Autoencoder（同缩写 GAE）。

## 核心摘录与 wiki 映射

1. **数据规模：** 汇集视频、动画与动捕，转换到统一 SMPL 表示，经左右镜像、时序拼接、上下身重组形成超过 1 万小时动作。映射到 [论文实体页的数据与方法](../../wiki/entities/paper-gae-general-action-expert.md)。
2. **生成器—执行器：** 特权生成器先在干净仿真中把形态拟合的人类参考转为物理可行的机器人轨迹；执行器在逐步增强的域随机化中学习跟踪该轨迹。执行器输入仍来自原始人类动作，生成轨迹只用于奖励目标，所以部署不需在线生成器。映射到 [训练流程](../../wiki/entities/paper-gae-general-action-expert.md)。
3. **延迟预判：** 执行器以延迟的人体动作和延迟参数为条件，使用 RoPE 时间索引偏移调整预判时域。50 Hz 下每帧 20 ms；论文测试 0–100 ms，真机展示 80 ms 的对照。映射到 [同步机制](../../wiki/entities/paper-gae-general-action-expert.md)。
4. **评测边界：** 相比 SONIC，在 Easy / Medium / Hard 集上的序列成功率分别为 97.3 / 96.9 / 93.9%（SONIC 95.4 / 85.2 / 77.7%）；Hard 集预判 100 ms 时成功率为 92.1%。序列成功定义为所有跟踪关键点与参考轨迹相距不超过 50 cm。真机为 G1，O1 需针对形态微调。映射到 [评测与局限](../../wiki/entities/paper-gae-general-action-expert.md)。

## 开放状态

截至 2026-10-02，[项目页](../sites/gae-general-action-expert.md) 提供论文与视频，但未列官方 GitHub、权重或数据集下载地址；不能把页面模板来源仓库当作算法代码。
