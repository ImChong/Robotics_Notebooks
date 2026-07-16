# Open-TeleVision

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 139/161）

- **标题：** Open-TeleVision: Teleoperation with Immersive Active Visual Feedback
- **类型：** paper
- **Loco-Manip 161 分类：** 08 硬件平台、感知配置与部署扩展
- **机构：** UC San Diego1、MIT2、San Diego、and GR- operator are at San Diego (approximately miles away). j: interactions with humans
- **项目页：** https://robot-tv.github.io/
- **发表日期：** 2024年7月8日
- **入库日期：** 2026-06-26
- **一句话说明：** Open-TeleVision 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、扩散策略/流匹配、MM-DiT/Transformer 动作头转成可训练、可复用的关节位置/力矩命令、末端执行器/腕手目标、动作 chunk/token。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 08 硬件平台、感知配置与部署扩展，编号 **139/161**。
- **算法实现总结（公众号）：** Open-TeleVision 主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过ACT/行为克隆模仿学习、扩散策略/流匹配、MM-DiT/Transformer 动作头转成可训练、可复用的关节位置/力矩命令、末端执行器/腕手目标、动作 chunk/token。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 对 wiki 的映射

- [paper-loco-manip-161-131-open-television](../../wiki/entities/paper-loco-manip-161-131-open-television.md)
- [loco-manip-161-category-08-hardware-deployment](../../wiki/overview/loco-manip-161-category-08-hardware-deployment.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
