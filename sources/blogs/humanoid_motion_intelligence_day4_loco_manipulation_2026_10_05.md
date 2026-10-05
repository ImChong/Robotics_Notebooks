# 具身智能从入门到精通 Day 4：移动操作

- **作者：** Yuanxq（具身智能研究室）
- **发表：** 2026-10-05
- **文章：** <https://mp.weixin.qq.com/s/UeBpHKuQDRnXG9lAnoH6AA>
- **原项目：** <https://github.com/RealXiaoze/humanoid-motion-intelligence>
- **归档依据：** 用户提供的 25 页 PDF；仅整理阅读索引，不转录原文。
- **独立文章节点：** [Day 4 导读](../../wiki/overview/humanoid-motion-intelligence-day4-loco-manipulation.md)

文章主线：全身命令协调 → 接触稳定和柔顺 → 物体状态驱动任务行为。以下 30 项分别链接到独立详情节点；已有页面直接复用。

| 工作 | 独立详情 | 作用 |
|---|---|---|
| Deep Whole-Body Control | [Deep Whole-Body Control](../../wiki/entities/paper-deep-whole-body-control-loco-manip.md) | 统一腿臂策略 |
| ULC | [ULC](../../wiki/entities/paper-loco-manip-161-048-ulc.md) | 统一移动和双臂命令 |
| Visual Whole-Body Control | [Visual Whole-Body Control](../../wiki/entities/paper-visual-whole-body-control-vbc.md) | 视觉目标分层控制 |
| FALCON | [FALCON](../../wiki/entities/paper-loco-manip-161-109-falcon.md) | 外力下全身协调 |
| SoFTA | [SoFTA](../../wiki/entities/paper-gentlehumanoid.md) | 末端稳定与步态分频 |
| CHIP | [CHIP](../../wiki/entities/paper-hrl-stack-36-chip.md) | 可调接触柔顺 |
| SkillBlender | [SkillBlender](../../wiki/entities/paper-loco-manip-161-077-skillblender.md) | 技能连续混合 |
| HDMI | [HDMI](../../wiki/entities/paper-hrl-stack-06-hdmi.md) | 跟踪人体、物体与接触 |
| VIRAL | [VIRAL](../../wiki/entities/paper-viral-humanoid-visual-sim2real.md) | 视觉仿真到现实移动操作 |
| DoorMan | [DoorMan](../../wiki/entities/paper-doorman-opening-sim2real-door.md) | 视觉开门策略 |
| CEER2 | [CEER2](../../wiki/entities/paper-ceer2-directional-compliance.md) | 方向可调末端和根部柔顺 |
| EgoHumanoid-V2 | [EgoHumanoid-V2](../../wiki/entities/paper-loco-manip-161-060-egohumanoid.md) | 人体全身技能迁移 |
| Uni-VLaT | [Uni-VLaT](../../wiki/entities/paper-uni-vlat.md) | 全身触觉适配 VLA |
| DexRoam | [DexRoam](../../wiki/entities/paper-dexroam-mobile-bimanual-manipulation.md) | 移动双手灵巧操作 |
| DexWeave | [DexWeave](../../wiki/entities/paper-dexweave-humanoid-loco-manipulation.md) | 人体交互重定向和全身策略 |
| CompliantWBC | [CompliantWBC](../../wiki/entities/paper-compliantwbc-heavy-humanoid.md) | 重型人形全身柔顺 |
| Praxis | [Praxis](../../wiki/entities/paper-praxis-egocentric-interaction-priors.md) | 第一视角交互先验（标题待核） |
| VisForce | [VisForce](../../wiki/entities/paper-visforce-force-grounding.md) | 视觉对齐当前力和目标力 |
| HOTICE | [HOTICE](../../wiki/entities/paper-hotice.md) | 拥挤环境物体运输 |
| STRIDER | [STRIDER](../../wiki/entities/paper-strider-multi-gait-loco-manip.md) | 多步态移动操作 |
| Whole-Body UMI | [Whole-Body UMI](../../wiki/entities/paper-whole-body-umi-realtime-motion.md) | 迁移 UMI 操作技能 |
| KINO | [KINO](../../wiki/entities/paper-kino.md) | 关键帧规划和全身执行 |
| ViLoMan | [ViLoMan](../../wiki/entities/paper-viloman.md) | 视觉—本体全身移动操作 |
| Weave | [Weave](../../wiki/entities/paper-weave.md) | 人体示范灵巧移动操作 |
| ForeTime-VLA | [ForeTime-VLA](../../wiki/entities/paper-foretime-vla.md) | 未来 token 蒸馏 |
| DECOWAM | [DECOWAM](../../wiki/entities/paper-decowam.md) | 解耦世界动作模型 |
| MobileWAM | [MobileWAM](../../wiki/entities/paper-mobilewam-mobile-manipulation-wam.md) | 移动操作世界动作模型 |
| TF-ART | [TF-ART](../../wiki/entities/paper-tf-art-tactile-force-survey.md) | 力觉和触觉学习综述 |
| FARO | [FARO](../../wiki/entities/paper-faro-feasibility-aware-robot-motion-optimization.md) | 可行性约束的运动优化 |
| SteadyTray | [SteadyTray](../../wiki/entities/paper-notebook-steadytray.md) | 托盘物体平衡 |

## 来源核查

PDF 将 Praxis 标为 *Distilling Physical Interaction Priors from Egocentric Videos for Generalizable Whole-Body Manipulation*，但所给项目 URL 当前显示 *Scaling One-Shot Human Demonstration to Generalist Policy for Whole-Body Manipulation*。尚未确认两者身份，本次分开保留并标为待核。
