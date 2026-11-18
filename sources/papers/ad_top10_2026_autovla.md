# AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning

- **类型：** 原始论文摘录（不转存论文）
- **论文：** https://arxiv.org/html/2506.13757
- **arXiv ID：** 2506.13757
- **机构：** UCLA Mobility
- **代码：** https://github.com/ucla-mobility/AutoVLA
- **核对日期：** 2026-11-18
- **核对范围：** 论文 §3的动作token、CoT/SFT/GRPO，§4实验与附录；核对README action_token_cluster/SFT/RFT/evaluation入口
- **通读：** 已完成

## 方法

将轨迹增量映射为动作codebook；简单场景直接产生动作token，复杂场景先生成驾驶CoT再生成动作；SFT之后用GRPO强化微调。

## 原文实验及边界

官方开放NAVSIM与nuScenes等训练和评测脚本，不能跨基准混用PDMS与轨迹误差。

动作token的离散误差及额外CoT的延时需单独评估；离线规划指标不是实车安全证明。

## 对 wiki 的映射

- [唯一详情节点](../../wiki/entities/paper-rcl-2506-13757-autovla-a-vision-language-action-model-for-end-t.md)
- [2026 自动驾驶十项目](../../wiki/overview/autonomous-driving-open-source-top10-2026.md)
