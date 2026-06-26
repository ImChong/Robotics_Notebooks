# SplitAdapter

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 045/161）

- **标题：** SplitAdapter: Load-Aware Humanoid Loco-Manipulation via Factorized Adaptation
- **类型：** paper
- **Loco-Manip 161 分类：** 02 上半身中心控制与移动操作接口
- **机构：** （见原文）
- **项目页：** https://splitadapter.github.io/
- **发表日期：** 2026年6月2日
- **入库日期：** 2026-06-26
- **一句话说明：** SplitAdapter 把仿真交互数据转成可跟踪的身体目标，并通过PPO/RL 策略训练、扩散策略/流匹配、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 02 上半身中心控制与移动操作接口，编号 **045/161**。
- **算法实现总结（公众号）：** SplitAdapter 把仿真交互数据转成可跟踪的身体目标，并通过PPO/RL 策略训练、扩散策略/流匹配、全身控制器/WBC/MPC训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 对 wiki 的映射

- [paper-loco-manip-161-045-splitadapter](../../wiki/entities/paper-loco-manip-161-045-splitadapter.md)
- [loco-manip-161-category-02-upper-body-interface](../../wiki/overview/loco-manip-161-category-02-upper-body-interface.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
