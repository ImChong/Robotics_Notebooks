# 打开仿真到现实世界的人形像素到动作策略传输之门

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 053/161）

- **标题：** Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer
- **类型：** paper
- **Loco-Manip 161 分类：** 02 上半身中心控制与移动操作接口
- **机构：** NVIDIA、UC Berkeley、CMU
- **项目页：** https://doorman-humanoid.github.io/
- **发表日期：** 2025年11月30日
- **入库日期：** 2026-06-26
- **一句话说明：** 这篇工作主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过教师-学生知识迁移、全身控制器/WBC/MPC、闭环纠错/人类干预转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是用特权信息训练教师策略，再把能力蒸馏到只能使用部署观测的学生策略。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 02 上半身中心控制与移动操作接口，编号 **053/161**。
- **算法实现总结（公众号）：** 这篇工作主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过教师-学生知识迁移、全身控制器/WBC/MPC、闭环纠错/人类干预转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是用特权信息训练教师策略，再把能力蒸馏到只能使用部署观测的学生策略。

## 对 wiki 的映射

- [paper-loco-manip-161-053-n053](../../wiki/entities/paper-hrl-stack-29-opening_the_sim_to_real_door_for_hum.md)
- [loco-manip-161-category-02-upper-body-interface](../../wiki/overview/loco-manip-161-category-02-upper-body-interface.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
