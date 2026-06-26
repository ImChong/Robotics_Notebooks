# AgiBot World Colosseo

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 146/161）

- **标题：** AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems
- **类型：** paper
- **Loco-Manip 161 分类：** 09 人形 VLA、世界模型与通用操作
- **机构：** （见原文）
- **项目页：** https://github.com/OpenDriveLab/AgiBot-World
- **发表日期：** 2025年8月4日
- **入库日期：** 2026-06-26
- **一句话说明：** AgiBot World Colosseo 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用VLM 语义规划/路由、潜变量/动作 token、MM-DiT/Transformer 动作头预测全身轨迹/动作序列、末端执行器/腕手目标、动作 chunk/token。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 09 人形 VLA、世界模型与通用操作，编号 **146/161**。
- **算法实现总结（公众号）：** AgiBot World Colosseo 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用VLM 语义规划/路由、潜变量/动作 token、MM-DiT/Transformer 动作头预测全身轨迹/动作序列、末端执行器/腕手目标、动作 chunk/token。关键点是把任务拆成可路由的技能或专家策略，再用高层模块在执行中选择和组合。

## 对 wiki 的映射

- [paper-loco-manip-161-146-agibot-world-colosseo](../../wiki/entities/paper-loco-manip-161-146-agibot-world-colosseo.md)
- [loco-manip-161-category-09-vla-world-models](../../wiki/overview/loco-manip-161-category-09-vla-world-models.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
