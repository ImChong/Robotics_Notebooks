# cyk579.github.io/Roller-Skating（Tsinghua 人形轮滑 AMP 项目页）

- **标题：** Learning Roller-Skating Motions of Humanoid Robots Based on Adversarial Motion Priors — 项目页
- **类型：** site / project-page
- **URL：** <https://cyk579.github.io/Roller-Skating/>
- **入库日期：** 2026-07-20
- **配套论文：** [arXiv:2607.10815](../papers/roller_skating_amp_arxiv_2607_10815.md)
- **机构：** 清华大学
- **平台：** Booster T1 + 被动轮滑改装

## 一句话摘要

清华团队 **被动轮人形轮滑** 官方演示站：展示 **Pump Glide** 与 **Push Glide** 在中/高摩擦面上的速度响应、多倍速与换速稳定性视频，以及 AMP-PPO 训练/部署框图。

## 公开信息要点（截至入库日）

- **核心主张：** 人体轮滑 MoCap → GMR 重定向 → 独立 AMP 训练两套 gait；真机 50 Hz 闭环控制。
- **演示分区：** Pump / Push 各自 Medium & High Friction、Multi-Speed Switching 视频。
- **Methods 页：** Actor 用历史观测编码；PPO + 判别器运动先验 + 多 critic；部署侧 IMU/关节/命令 → 关节动作。
- **致谢：** Booster Robotics 提供平台与实验支持。
- **代码：** 项目页 **未列出 GitHub / Hugging Face / 数据集链接** → 截至入库日按 **未开源** 标注。

## 为何值得保留

- 被动轮碰撞建模与双 gait 真机视频是 arXiv 摘要难以替代的 **sim2real 证据**。
- 与 [SKATER](../papers/humanoid_pnb_skater-synthesized-kinematics-for-advanced-trave.md) 构成人形轮滑 **AMP vs 任务奖励** 对照入口。

## 关联资料

- 论文归档：[`sources/papers/roller_skating_amp_arxiv_2607_10815.md`](../papers/roller_skating_amp_arxiv_2607_10815.md)
- Wiki 实体：[`wiki/entities/paper-roller-skating-amp-humanoid-passive-wheels.md`](../../wiki/entities/paper-roller-skating-amp-humanoid-passive-wheels.md)
