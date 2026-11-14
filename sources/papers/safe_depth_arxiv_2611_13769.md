# From Depth to Safe Depth: Uncertainty-Aware Monocular Perception for Mobile Robot Navigation｜不确定性校准单目深度导航

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.13769
- **PDF：** https://arxiv.org/pdf/2611.13769
- **核对范围：** 核对论文 arXiv v1 摘要、用户 2026-11-12 资讯附件、官方源码 README 与公开入口；未通读全文方法、实验与附录，未执行实验。
- **通读：** 未完成
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

## 待补

需通读论文图表、正文和附录并逐项校验结果；当前内容为原始摘要及资讯整理，不替代原文。
