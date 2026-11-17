# Cosmos Policy

> 来源归档（ingest · 深蓝世界模型 15 项目 第 11/15）

- **标题：** Cosmos Policy
- **类型：** paper
- **路线分类：** 02 联合架构（[微信公众号编译](../blogs/wechat_shenlan_world_models_15_open_source_2026.md)）
- **出处：** —
- **论文链接：** <https://arxiv.org/abs/2601.16163>
- **文内引用（2026-06-02）：** 61
- **索引来源：** [wechat_shenlan_world_models_15_open_source_2026.md](../blogs/wechat_shenlan_world_models_15_open_source_2026.md)（<https://mp.weixin.qq.com/s/KZT8sI4n7GvHWyM20wN3gg>）
- **入库日期：** 2026-06-03
- **一句话说明：** 微调 Cosmos-Predict2 视频基础模型实现视觉运动控制与规划。

## 核心摘录（策展，非全文）

- **在 15 项目地图中的位置：** 02 联合架构，编号 **11/15**。
- **公众号导读要点：** 微调 Cosmos-Predict2 视频基础模型实现视觉运动控制与规划。

## 对 wiki 的映射

- Wiki：[`wiki/entities/paper-shenlan-wm-11-cosmos-policy.md`](../../wiki/entities/paper-shenlan-wm-11-cosmos-policy.md)
- [world-models-15-open-source-technology-map](../../wiki/overview/world-models-15-open-source-technology-map.md)
- [world-models-route-02-joint](../../wiki/overview/world-models-route-02-joint.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2601.16163>
- 微信公众号编译：[wechat_shenlan_world_models_15_open_source_2026.md](../blogs/wechat_shenlan_world_models_15_open_source_2026.md)

## 一手论文复核（2026-11-17）

- **原文：** [Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning, arXiv:2601.16163](https://arxiv.org/html/2601.16163)。
- **核读范围：** 正文问题设定、latent frame injection 与 action/world/value 的联合建模、示范数据后训练、test-time value-guided planning、LIBERO/RoboCasa/真机实验及限制；原论文公开项目入口。未实际执行官方代码。
- **通读：** 已完成

使用 Cosmos-Predict2 的视频生成先验，将动作、未来视觉及价值估计编码成同一扩散过程中的潜在帧，在目标平台示范数据上单阶段训练。论文报告 LIBERO 98.5% 和 RoboCasa 67.1% 的平均成功率（不同任务基准不可直接比较），基于世界与价值预测的候选动作筛选可改善困难任务，但耗时的测试期规划不是每步 50Hz 的低层闭环控制。原 survey 仅作索引，不作为方法和数字依据。
