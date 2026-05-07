# SONIC（规模化人体运动跟踪驱动的人形全身控制）

- **标题**: SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control
- **论文**: https://arxiv.org/abs/2511.07820
- **项目页**: https://nvlabs.github.io/SONIC/ （与历史别名页 https://nvlabs.github.io/GEAR-SONIC/ 指向同一项目材料）
- **类型**: paper / foundation-controller
- **机构**: NVIDIA、CMU 等（论文Author列表以官网为准）
- **收录日期**: 2026-05-07

## 一句话摘要

将 **运动跟踪（motion tracking）** 作为可规模化监督信号，用海量高质量动捕帧训练通用人形策略，把「跟踪参考运动」学到表征里，从而支持 VR、视频、文本、音乐等多接口输入，并作为下游任务的通用低层执行器。

## 为何值得保留

- **范式**: 用密集轨迹监督替代大量手工任务奖励工程，与 BeyondMimic / DeepMimic 族思路相承但在 **数据量、模型规模与接口多样性** 上强调 scaling。
- **系统集成**: 被视频驱动人形流水线用作「物理过滤器」——把估计的人体运动映射到真实机器人可行域（见 ExoActor 等）。
- **公开材料**: arXiv 论文与项目页提供方法概述与引用资源。

## 技术要点（来自论文公开描述）

1. **Scaling 维度**：同时增大网络容量、训练数据规模（论文量级描述为亿级帧、数百小时 MoCap）与算力投入，性能随规模稳步提升。
2. **统一 token 空间**：多种控制模态映射到同一表示，便于替换上游（如 VLA、视频估计）。
3. **实时性与规划**：强调实时全身跟踪与通用运动学规划桥接，支撑交互式部署。
4. **与 BeyondMimic 关系**：同属 Isaac / 人形模仿生态中的高性能跟踪路线；SONIC 侧重「规模化跟踪即基础能力」的产品化叙事。

## 对 Wiki 的映射

- **wiki/methods/sonic-motion-tracking.md**：人形通用动作跟踪基础模型方法页。
- **wiki/methods/beyondmimic.md**：历史与技术脉络对齐。
- **wiki/methods/exoactor.md**：作为「SMPL 估计轨迹 → 机器人执行」的执行层实例。
