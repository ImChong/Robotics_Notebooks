# 人形机器人行为基础模型

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 027/161）

- **标题：** Behavior Foundation Model for Humanoid Robots
- **类型：** paper
- **Loco-Manip 161 分类：** 01 运控基座与通用全身跟踪
- **机构：** Peking University、The Chinese University of Hong Kong, Shenzhen、Shanghai Jiaotong University、Fudan University
- **项目页：** https://bfm4humanoid.github.io
- **发表日期：** 2025年9月17日
- **入库日期：** 2026-06-26
- **一句话说明：** 这篇工作主要解决数据闭环：用本体状态与关节序列、遥操作/外骨骼数据、仿真交互数据采集人类操作和机器人状态，再通过教师-学生知识迁移、扩散策略/流匹配、全身控制器/WBC/MPC转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是用特权信息训练教师策略，再把能力蒸馏到只能使用部署观测的学生策略。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 01 运控基座与通用全身跟踪，编号 **027/161**。
- **算法实现总结（公众号）：** 这篇工作主要解决数据闭环：用本体状态与关节序列、遥操作/外骨骼数据、仿真交互数据采集人类操作和机器人状态，再通过教师-学生知识迁移、扩散策略/流匹配、全身控制器/WBC/MPC转成可训练、可复用的全身轨迹/动作序列、低层控制器目标。关键点是用特权信息训练教师策略，再把能力蒸馏到只能使用部署观测的学生策略。

## 对 wiki 的映射

- [paper-loco-manip-161-027-n027](../../wiki/entities/paper-loco-manip-161-027-n027.md)
- [loco-manip-161-category-01-motion-base-wbt](../../wiki/overview/loco-manip-161-category-01-motion-base-wbt.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
