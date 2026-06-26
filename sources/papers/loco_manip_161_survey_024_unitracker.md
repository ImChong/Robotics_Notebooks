# UniTracker

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 024/161）

- **标题：** UniTracker: Learning Universal Whole-Body Motion Tracker for Humanoid Robots
- **类型：** paper
- **Loco-Manip 161 分类：** 01 运控基座与通用全身跟踪
- **机构：** Shanghai Jiao Tong Univeristy、Shanghai Artificial Intelligence Laboratory、Shanghai Innovation Institute、Peking University
- **项目页：** https://yinkangning0124.github.io/Humanoid-UniTracker/
- **发表日期：** 2025年9月18日
- **入库日期：** 2026-06-26
- **一句话说明：** UniTracker 把相机图像/多视角观测、本体状态与关节序列、仿真交互数据转成可跟踪的身体目标，并通过教师-学生知识迁移、PPO/RL 策略训练、扩散策略/流匹配训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是用特权信息训练教师策略，再把能力蒸馏到只能使用部署观测的学生策略。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 01 运控基座与通用全身跟踪，编号 **024/161**。
- **算法实现总结（公众号）：** UniTracker 把相机图像/多视角观测、本体状态与关节序列、仿真交互数据转成可跟踪的身体目标，并通过教师-学生知识迁移、PPO/RL 策略训练、扩散策略/流匹配训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是用特权信息训练教师策略，再把能力蒸馏到只能使用部署观测的学生策略。

## 对 wiki 的映射

- [paper-loco-manip-161-024-unitracker](../../wiki/entities/paper-loco-manip-161-024-unitracker.md)
- [loco-manip-161-category-01-motion-base-wbt](../../wiki/overview/loco-manip-161-category-01-motion-base-wbt.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
