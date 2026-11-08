# Flow Matching in Feature Space for Stochastic World Modeling / 特征空间流匹配随机世界模型

- **类型：** 论文原文归档（非转载）
- **主论文：** <https://arxiv.org/abs/2606.29059>
- **HTML 原文：** <https://arxiv.org/html/2606.29059>
- **官方代码：** <https://github.com/facebookresearch/Flow-World-Models>
- **入库日期：** 2026-11-08
- **核对范围：** 全文 1–6 章（Flow Matching / 训练目标 / Waymo 与合成数据评测），以及附录 A–K 的推导、消融、失败变体与局限；核验官方 main.py/README 配置。
- **通读：** 已完成

## 核心方法

基于冻结的 DINOv3 高维 Patch 特征；在条件流匹配中预测未来特征分布，而非像素或确定性均值；加入单步投影的时间一致性与下游任务监督。

## 实验与边界

合成 Bouncing Shapes 与真实 Waymo FuturePerception：论文 Table 2 的 FlowWM AP_L(3)=20.9、AP_L(6)=21.7、深度 RMSE=0.078；这些不是机器人成功率。

随机采样和高维特征的算力代价；Waymo 检测/深度指标不直接证明闭环机器人操控。

## 对 wiki 的映射

- [paper-flowwm-feature-space-stochastic-wm.md](../../wiki/entities/paper-flowwm-feature-space-stochastic-wm.md)

## 2026-11-08 一手资料精读笔记

- **模型架构：** 冻结 DINOv3 高维 patch 特征，在未来视频特征空间执行条件流匹配并配合单步投影时间一致性及检测任务损失；**按历史视觉上下文条件化，不输入候选机器人动作**。
- **实验口径：** Waymo FuturePerception 附录消融 AP_L(3)：基线19.1，加时间一致性20.1，加 detector loss 20.9；多采样可提升随机未来覆盖度，但这些是目标感知指标不是机器人成功率。
- **局限风险：** 作者附录 J 明确 action-conditioned dynamics 与 planning 仍是 future work；冻结编码器、长视频采样成本与 CC BY-NC 许可约束。
