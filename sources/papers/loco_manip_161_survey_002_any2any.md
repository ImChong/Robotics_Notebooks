# Any2Any

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 002/161）

- **标题：** Any2Any: Efficient Cross-Embodiment Transfer for Humanoid Whole-Body Tracking
- **类型：** paper
- **Loco-Manip 161 分类：** 01 运控基座与通用全身跟踪
- **机构：** （见原文）
- **项目页：** https://any2any.top/
- **发表日期：** 2026年6月18日
- **入库日期：** 2026-06-26
- **一句话说明：** Any2Any 把视觉、状态和动作数据转成可跟踪的身体目标，并通过源/目标机器人运动学对齐、PEFT 动力学适配、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是先对齐源/目标机器人的状态和动作空间，再只微调动力学敏感模块，尽量保留原策略的运动先验。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 01 运控基座与通用全身跟踪，编号 **002/161**。
- **算法实现总结（公众号）：** Any2Any 把视觉、状态和动作数据转成可跟踪的身体目标，并通过源/目标机器人运动学对齐、PEFT 动力学适配、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是先对齐源/目标机器人的状态和动作空间，再只微调动力学敏感模块，尽量保留原策略的运动先验。

## 对 wiki 的映射

- [paper-loco-manip-161-002-any2any](../../wiki/entities/paper-loco-manip-161-002-any2any.md)
- [loco-manip-161-category-01-motion-base-wbt](../../wiki/overview/loco-manip-161-category-01-motion-base-wbt.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
