# From Depth to Safe Depth: Uncertainty-Aware Monocular Perception for Mobile Robot Navigation｜不确定性校准单目深度导航

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.13769
- **PDF：** https://arxiv.org/pdf/2611.13769
- **核对范围：** arXiv:2611.13769v1 PDF 全部 5 页：核心示意图、方法/保形标定公式、实验配置、定量表、消融及局限；核对 GitHub README 的 ROS 2 Humble/Gazebo 评测入口。未运行模拟器。
- **通读：** 已完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://github.com/saeed5959/depth_safe_robot

## 核心摘录

1. 从 RGB 获得单目深度，并用小型 0.58M 参数头估计对数偏差 μ 和方差尺度 σ。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-safe-depth.md)
2. 留出世界做保形标定求 k，形成偏向障碍物一侧的安全深度界。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-safe-depth.md)
3. 由 Nav2 消费安全深度并据不确定性判断可通行地面。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-safe-depth.md)

## 作者报告的结果与限制

论文摘要称 120 次配对仿真中碰撞率 63%→3%、导航成功率 37%→70%，未在真机复测。

仿真中采用真值里程计，跨地图/相机/材质的校准失配与真机可靠性仍需验证。

## 原文核对摘录（2026-11-14）

方法 §III：冻结 Depth Anything V2 ViT-S，加入 36 通道 mini U-Net（约 0.58M 参数）预测 log-depth 修正 μ 和不确定性 σ，使用 β-NLL β=0.5 训练。单侧保守深度 d_safe=d_DA2×exp(μ−kσ)，论文验证域使用 k=3.0（约 97.7% 单侧覆盖）；对地面使用不确定性辅助清理错误障碍。§IV–V：ROS 2 Humble + Gazebo Fortress、四个未见测试世界、120 个配对回合，原单目深度碰撞率 63%，Safe Depth 3%；成功率从 37% 到 70%，GT 深度约 93%；correction-only 60%、uniform margin 30%。**评测使用真值里程计，没有真机迁移**，所以不能当成形式化安全保证。

## 复现边界

尚未独立运行训练、推理或机器人试验；论文数字为作者报告。对 wiki 的映射：[论文详情](../../wiki/entities/paper-safe-depth.md)。
