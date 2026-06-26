# TrajBooster

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 080/161）

- **标题：** TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning
- **类型：** paper
- **Loco-Manip 161 分类：** 03 视觉感知驱动的人形移动操作
- **机构：** Limited Teleoperation Data
- **项目页：** https://jiachengliu3.github.io/TrajBooster
- **发表日期：** 2026年3月19日
- **入库日期：** 2026-06-26
- **一句话说明：** TrajBooster 的实现路径是先把相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、IK/动作重定向预测全身轨迹/动作序列、末端执行器/腕手目标。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 03 视觉感知驱动的人形移动操作，编号 **080/161**。
- **算法实现总结（公众号）：** TrajBooster 的实现路径是先把相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、IK/动作重定向预测全身轨迹/动作序列、末端执行器/腕手目标。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 对 wiki 的映射

- [paper-loco-manip-161-080-trajbooster](../../wiki/entities/paper-loco-manip-161-080-trajbooster.md)
- [loco-manip-161-category-03-visuomotor](../../wiki/overview/loco-manip-161-category-03-visuomotor.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
