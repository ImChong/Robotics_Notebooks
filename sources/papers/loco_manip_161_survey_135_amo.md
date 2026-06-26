# AMO

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 135/161）

- **标题：** AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control
- **类型：** paper
- **Loco-Manip 161 分类：** 08 硬件平台、感知配置与部署扩展
- **机构：** UC San Diego
- **项目页：** https://amo-humanoid.github.io/
- **发表日期：** 2025年5月6日
- **入库日期：** 2026-06-26
- **一句话说明：** AMO 把本体状态与关节序列、仿真交互数据、接触力/触觉信号转成可跟踪的身体目标，并通过PPO/RL 策略训练、ACT/行为克隆模仿学习、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 08 硬件平台、感知配置与部署扩展，编号 **135/161**。
- **算法实现总结（公众号）：** AMO 把本体状态与关节序列、仿真交互数据、接触力/触觉信号转成可跟踪的身体目标，并通过PPO/RL 策略训练、ACT/行为克隆模仿学习、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。

## 对 wiki 的映射

- [paper-loco-manip-161-135-amo](../../wiki/entities/paper-loco-manip-161-135-amo.md)
- [loco-manip-161-category-08-hardware-deployment](../../wiki/overview/loco-manip-161-category-08-hardware-deployment.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
