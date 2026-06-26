# 人形机器人富有表现力的全身控制

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 026/161）

- **标题：** Expressive Whole-Body Control for Humanoid Robots
- **类型：** paper
- **Loco-Manip 161 分类：** 01 运控基座与通用全身跟踪
- **机构：** UC San Diego
- **项目页：** https://expressive-humanoid.github.io
- **发表日期：** 2024年3月6日
- **入库日期：** 2026-06-26
- **一句话说明：** 这篇工作把本体状态与关节序列、人类视频/动捕轨迹、仿真交互数据转成可跟踪的身体目标，并通过异构动捕与合成平衡数据、PPO/RL 策略训练、ACT/行为克隆模仿学习训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 01 运控基座与通用全身跟踪，编号 **026/161**。
- **算法实现总结（公众号）：** 这篇工作把本体状态与关节序列、人类视频/动捕轨迹、仿真交互数据转成可跟踪的身体目标，并通过异构动捕与合成平衡数据、PPO/RL 策略训练、ACT/行为克隆模仿学习训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把示范轨迹压成可监督的动作预测问题，再通过动作 chunk 或闭环执行降低时序抖动。

## 对 wiki 的映射

- [paper-loco-manip-161-026-n026](../../wiki/entities/paper-loco-manip-161-026-n026.md)
- [loco-manip-161-category-01-motion-base-wbt](../../wiki/overview/loco-manip-161-category-01-motion-base-wbt.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
