# LHNav — Long-Horizon Humanoid Navigation
- **原文：** https://arxiv.org/html/2611.12136v1
- **论文编号：** arXiv:2611.12136v1
- **项目主页：** https://robbyant-research.github.io/LHNav/
- **官方项目仓：** https://github.com/robbyant-research/LHNav
- **日期：** 2026-11-11（本次核对）
- **核对范围：** arXiv HTML 全文的摘要、架构与训练方法、仿真及 Unitree G1 真机实验、对照与局限；并核对项目页和官方 GitHub 仓根目录。
- **通读：** 已完成

## 核心摘录与推理边界
1. 分层为 Brain（慢速任务规划/到达核验）→ Cerebellum（GR00T N1.7 VLA 视觉/指令条件动作与航点）→ Brainstem（SONIC 类冻结运动解码器，关节控制 50 Hz）。Brain 是独立任务层，不能把三者误写成同一高频 policy。
2. Cerebellum 使用目标与当前图像、机器人本体感知，生成运动 latent 与 2D/位姿航点；采用 SDF/路径约束对复杂接触区域进行训练。训练包含视觉渲染的几何场景和仿真的人形运动跟踪示范。
3. 仿真中 522 个 held-out episodes：作者报告 precise SR@0.5m 52.3%、SR@1.5m 75.7%、collision 8.6%、0 falls。OmniNav 分别 36.0%/69.0%/19.5%/7 falls；DualVLN 18.6%/48.7%/25.8%/91 falls。分母为该评估协议，**不**是全部现实场景成功率。
4. 45 个 long-horizon tasks 的论文实验报告约 75.56% 任务成功；G1 三站点任务 4/5 次完成。独立单目标 G1 探索 25/25 等成绩不能替代多站点长时程成功率。
5. 文中评估的物体位置变化是**导航目标变化**，不构成已完成抓取或物体操控的证据。

## 官方材料开放检查
- [项目页](../sites/lhnav-official.md)提供演示与论文。
- [官方 GitHub](../repos/lhnav-official.md)只有网站托管文件，未找到可执行的训练、推理或 G1 部署入口；模型权重/数据发布状态须后续追踪。
- **独立复现：** 本次无硬件/场景，不等于复现作者指标。

## 对 wiki 的映射
- [LHNav 主实体](../../wiki/entities/paper-lhnav.md)
- [视觉语言导航](../../wiki/tasks/vision-language-navigation.md)
