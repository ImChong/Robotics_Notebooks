# 通过触摸梦学习多样化人形操作

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 092/161）

- **标题：** Learning Versatile Humanoid Manipulation with Touch Dreaming
- **类型：** paper
- **Loco-Manip 161 分类：** 03 视觉感知驱动的人形移动操作
- **机构：** Carnegie Mellon University、UT Arlington、Bosch Center for AI、deformable objects (towel folding). B: mixed prehensile and non-prehensile manipulation for thin-profile rigid objects with limited grasp
- **项目页：** https://humanoid-touch-dream.github.io/
- **发表日期：** 2026年4月27日
- **入库日期：** 2026-06-26
- **一句话说明：** 这篇工作主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过PPO/RL 策略训练、扩散策略/流匹配、MM-DiT/Transformer 动作头转成可训练、可复用的末端执行器/腕手目标、动作 chunk/token。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 03 视觉感知驱动的人形移动操作，编号 **092/161**。
- **算法实现总结（公众号）：** 这篇工作主要解决数据闭环：用相机图像/多视角观测、本体状态与关节序列、遥操作/外骨骼数据采集人类操作和机器人状态，再通过PPO/RL 策略训练、扩散策略/流匹配、MM-DiT/Transformer 动作头转成可训练、可复用的末端执行器/腕手目标、动作 chunk/token。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 对 wiki 的映射

- [paper-loco-manip-161-092-n092](../../wiki/entities/paper-loco-manip-161-092-n092.md)
- [loco-manip-161-category-03-visuomotor](../../wiki/overview/loco-manip-161-category-03-visuomotor.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
