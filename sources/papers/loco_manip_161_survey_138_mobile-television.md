# Mobile-TeleVision

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 138/161）

- **标题：** Mobile-TeleVision: Predictive Motion Priors for Humanoid Whole-Body Control
- **类型：** paper
- **Loco-Manip 161 分类：** 08 硬件平台、感知配置与部署扩展
- **机构：** UC San Diego、MIT
- **项目页：** https://mobile-tv.github.io/
- **发表日期：** 2025年3月9日
- **入库日期：** 2026-06-26
- **一句话说明：** Mobile-TeleVision 把本体状态与关节序列、人类视频/动捕轨迹、接触力/触觉信号转成可跟踪的身体目标，并通过PPO/RL 策略训练、AMP/运动先验、扩散策略/流匹配训练或组合全身策略，最终输出关节位置/力矩命令、全身轨迹/动作序列、地形/场景表征。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 08 硬件平台、感知配置与部署扩展，编号 **138/161**。
- **算法实现总结（公众号）：** Mobile-TeleVision 把本体状态与关节序列、人类视频/动捕轨迹、接触力/触觉信号转成可跟踪的身体目标，并通过PPO/RL 策略训练、AMP/运动先验、扩散策略/流匹配训练或组合全身策略，最终输出关节位置/力矩命令、全身轨迹/动作序列、地形/场景表征。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 对 wiki 的映射

- [paper-loco-manip-161-138-mobile-television](../../wiki/entities/paper-loco-manip-161-138-mobile-television.md)
- [loco-manip-161-category-08-hardware-deployment](../../wiki/overview/loco-manip-161-category-08-hardware-deployment.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
