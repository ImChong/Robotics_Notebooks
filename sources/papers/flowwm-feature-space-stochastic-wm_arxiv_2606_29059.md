# Flow Matching in Feature Space for Stochastic World Modeling / 特征空间流匹配随机世界模型

- **类型：** 论文原文归档（非转载）
- **主论文：** <https://arxiv.org/abs/2606.29059>
- **HTML 原文：** <https://arxiv.org/html/2606.29059>
- **官方代码：** <https://github.com/facebookresearch/Flow-World-Models>
- **入库日期：** 2026-11-08
- **核对范围：** 阅读 HTML §3、§4、§5 部分实验与 README 运行说明；附录、所有消融和源码执行路径尚未全读
- **通读：** 未完成

## 核心方法

基于冻结的 DINOv3 高维 Patch 特征；在条件流匹配中预测未来特征分布，而非像素或确定性均值；加入单步投影的时间一致性与下游任务监督。

## 实验与边界

合成 Bouncing Shapes 与真实 Waymo FuturePerception：论文 Table 2 的 FlowWM AP_L(3)=20.9、AP_L(6)=21.7、深度 RMSE=0.078；这些不是机器人成功率。

随机采样和高维特征的算力代价；Waymo 检测/深度指标不直接证明闭环机器人操控。

## 对 wiki 的映射

- [paper-flowwm-feature-space-stochastic-wm.md](../../wiki/entities/paper-flowwm-feature-space-stochastic-wm.md)
