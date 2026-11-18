# DriveWAM: Video Generative Priors Enable Scalable World-Action Modeling for Autonomous Driving

- **类型：** 原始论文摘录（不转存论文）
- **论文：** https://arxiv.org/html/2605.28544
- **arXiv ID：** 2605.28544
- **机构：** 香港中文大学（深圳）等
- **代码：** https://github.com/chenshi3/DriveWAM
- **核对日期：** 2026-11-18
- **核对范围：** 论文§3.1视频动作生成、§3.2场景引导、§3.3 KV记忆及实验附录；官方README训练与评测入口
- **通读：** 已完成

## 方法

把历史视频与动作组织为时序潜变量，在预训练视频DiT中联合生成未来视频与动作；冻结VLM提供块级场景引导，选择性KV记忆管理长时上下文。

## 原文实验及边界

官方代码提供NAVSIM及PhysicalAI-Autonomous-Vehicles训练与评测入口。

生成画面不是物理精确的交通仿真；KV记忆预算与自回归累积误差会限制长期性能。

## 对 wiki 的映射

- [唯一详情节点](../../wiki/entities/paper-rcl-2605-28544-drivewam-video-generative-priors-enable-scalable.md)
- [2026 自动驾驶十项目](../../wiki/overview/autonomous-driving-open-source-top10-2026.md)
