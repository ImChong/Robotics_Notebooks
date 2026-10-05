# 具身智能从入门到精通 Day 3：动作跟踪与全身控制

- **作者：** Yuanxq（具身智能研究室）
- **发表：** 2026-10-04
- **文章：** <https://mp.weixin.qq.com/s?__biz=Mzg5Mjc3MjA5Nw==&mid=2247503024&idx=1&sn=ddcbec9f9b7f8880a88f38127fd6a44c>
- **原项目：** <https://github.com/RealXiaoze/humanoid-motion-intelligence>
- **归档依据：** 用户提供的 27 页 PDF；仅整理阅读索引，不转录原文。
- **独立文章节点：** [Day 3 导读](../../wiki/overview/humanoid-motion-intelligence-day3-motion-tracking-wbc.md)

文章主线：动作参考闭环跟踪 → 稀疏目标补全全身动作 → 接入任务规划与环境反馈 → 跌倒恢复和命令重获。以下 29 项分别链接到唯一详情节点；已有页面直接复用。

| 工作 | 独立详情 | 作用 |
|---|---|---|
| DeepMimic | [DeepMimic](../../wiki/methods/deepmimic.md) | 物理动作模仿与恢复 |
| H2O | [H2O](../../wiki/entities/paper-hrl-stack-07-learning_human_to_humanoid_real_time.md) | 视频遥操作和教师—学生全身控制 |
| TWIST | [TWIST](../../wiki/entities/paper-twist.md) | 遥操作与动作示范采集 |
| TWIST 2 | [TWIST 2](../../wiki/entities/paper-twist2.md) | 便携人形动作数据采集 |
| ExBody | [ExBody](../../wiki/entities/paper-exbody-expressive-humanoid.md) | 上身跟踪与下肢步态补全 |
| OmniH2O | [OmniH2O](../../wiki/entities/paper-hrl-stack-08-omnih2o.md) | 头手目标补全全身动作 |
| MaskedMimic | [MaskedMimic](../../wiki/entities/paper-bfm-17-maskedmimic.md) | 遮罩条件动作补全 |
| HOVER | [HOVER](../../wiki/entities/paper-bfm-14-hover.md) | 多模态命令统一控制 |
| BeyondMimic | [BeyondMimic](../../wiki/methods/beyondmimic.md) | 动作跟踪与引导扩散规划 |
| SONIC | [SONIC](../../wiki/methods/sonic-motion-tracking.md) | 规模化全身动作跟踪 |
| GAE | [GAE](../../wiki/entities/paper-gae-general-action-expert.md) | 跨本体实时遥操作 |
| PLAT | [PLAT](../../wiki/entities/paper-plat-sparse-keyframe-tracking.md) | 稀疏关键帧跟踪 |
| Runway Humanoid | [Runway Humanoid](../../wiki/entities/paper-runway-expressive-locomotion.md) | 从单目视频迁移步态风格 |
| X-WBC | [X-WBC](../../wiki/entities/paper-x-wbc.md) | 跨本体全身控制模型 |
| ViBe | [ViBe](../../wiki/entities/paper-vibe.md) | 视觉适配全身控制 |
| PGMT | [PGMT](../../wiki/entities/paper-pgmt.md) | 地形感知动作跟踪 |
| AdaPT | [AdaPT](../../wiki/entities/paper-adapt.md) | 网球风格动作自适应规划 |
| GigaBrain-WBC-0.5 | [GigaBrain-WBC-0.5](../../wiki/entities/paper-gigabrain-wbc-0-5.md) | 交互感知行为世界模型 |
| HumanTracker | [HumanTracker](../../wiki/entities/paper-humantracker.md) | 人类对齐跟踪评测 |
| PFM-HR | [PFM-HR](../../wiki/entities/paper-pfm-hr.md) | 姿态先验与动作跟踪 |
| StableMimic | [StableMimic](../../wiki/entities/paper-stablemimic.md) | 跟踪策略的跌倒恢复 |
| Teleopit | [Teleopit](../../wiki/entities/paper-teleopit.md) | 全身遥操作和数据采集 |
| Extreme-RGMT | [Extreme-RGMT](../../wiki/entities/paper-extreme-rgmt.md) | 高动态技能持续学习 |
| YAHMP | [YAHMP](../../wiki/entities/paper-yahmp.md) | 通用跟踪策略消融 |
| ScaleBFM | [ScaleBFM](../../wiki/entities/paper-scaling-bfm-humanoid.md) | 行为基础模型规模化 |
| MimicLite | [MimicLite](../../wiki/entities/mimiclite.md) | 高效动作跟踪 |
| HEFT | [HEFT](../../wiki/entities/paper-heft.md) | 重载遥操作 |
| ReactiveBFM | [ReactiveBFM](../../wiki/entities/paper-reactivebfm.md) | 本体反馈下分块规划 |
| AnyBody | [AnyBody](../../wiki/entities/paper-anybody-keypoint-humanoid-control.md) | 任意关键点全身控制 |
| FADA | [FADA](../../wiki/entities/paper-fada-humanoid.md) | 少样本动力学适配 |

文章中的定量结果是定位线索；指标、代码状态和真机范围需回到论文或项目页核对。
