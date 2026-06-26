# 学习人对人的实时全身远程操作

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 029/161）

- **标题：** Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation
- **类型：** paper
- **Loco-Manip 161 分类：** 01 运控基座与通用全身跟踪
- **机构：** Carnegie Mellon University
- **项目页：** https://human2humanoid.com
- **发表日期：** 2024年3月7日
- **入库日期：** 2026-06-26
- **一句话说明：** 这篇工作主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹采集人类操作和机器人状态，再通过IK/动作重定向转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是把IK/动作重定向放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 01 运控基座与通用全身跟踪，编号 **029/161**。
- **算法实现总结（公众号）：** 这篇工作主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹采集人类操作和机器人状态，再通过IK/动作重定向转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是把IK/动作重定向放在同一条训练/部署链路里，减少高层目标到低层动作之间的断点。

## 对 wiki 的映射

- [paper-loco-manip-161-029-n029](../../wiki/entities/paper-loco-manip-161-029-n029.md)
- [loco-manip-161-category-01-motion-base-wbt](../../wiki/overview/loco-manip-161-category-01-motion-base-wbt.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
