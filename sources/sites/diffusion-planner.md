# Diffusion Planner 官方项目主页归档

- **类型：** site / autonomous-driving / planning / diffusion
- **标题：** Diffusion-Planner / Diffusion-Based Planning for Autonomous Driving with Flexible Guidance
- **项目主页：** <https://zhengyinan-air.github.io/Diffusion-Planner/>
- **论文：** <https://arxiv.org/abs/2501.15564>（v2）
- **代码：** <https://github.com/ZhengYinan-AIR/Diffusion-Planner>
- **模型：** <https://huggingface.co/ZhengYinan2001/Diffusion-Planner>
- **数据集：** 配送车 200 小时自采集数据，页面标为「coming soon」，未提供公开下载
- **入库日期：** 2026-11-21
- **核对范围：** 通读官方项目页 Abstract、Methods、Classifier Guidance、nuPlan 闭环、配送车数据集与表格、BibTeX、公开资源入口；对照仓库 README 的下载/运行说明。
- **通读：** 已完成
- **开源状态：** 已公开代码与模型权重；**配送车原始数据尚未公开**。

## 官方声明核对

- 使用 DiT 建模自车+关键邻车未来轨迹，与矢量化地图/车道及导航条件融合。
- 运行时通过目标车速、碰撞、舒适度、可行驶区域等**可微引导项**调整规划行为；无需为新的引导项重新训练模型。
- 官方报告 nuPlan **无 refine** Test14 (Reactive) 82.93、Test14-hard (Reactive) 69.22；计划未来 8 秒、采样约 20 Hz。
- 200 h 物流配送车数据作为跨驾驶风格评估的附加证据；页面「Dataset coming soon」意味着当前无法由页面获取原始数据进行等价复测。
- 指向 [唯一论文实体节点](../../wiki/entities/paper-diffusion-planner-autonomous-driving.md)，不为站点另外创建重复项目实体。

## 一手来源

- [项目主页](https://zhengyinan-air.github.io/Diffusion-Planner/)
- [论文原文归档](../papers/diffusion_planner_arxiv_2501_15564.md)
- [源码入口归档](../repos/diffusion-planner.md)
