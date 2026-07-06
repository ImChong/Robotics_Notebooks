# CoorDex

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 035/161）

- **标题：** CoorDex: Coordinating Body and Hand Priors for Continuous Dexterous Humanoid Loco-Manipulation
- **类型：** paper
- **Loco-Manip 161 分类：** 02 上半身中心控制与移动操作接口
- **机构：** University of North Carolina at Chapel Hill、University of California, Berkeley
- **项目页：** https://skevinci.github.io/coordex/
- **发表日期：** 2026年6月22日
- **入库日期：** 2026-06-26
- **一句话说明：** CoorDex 把本体状态与关节序列、接触力/触觉信号转成可跟踪的身体目标，并通过PPO/RL 策略训练、扩散策略/流匹配、MM-DiT/Transformer 动作头训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标、地形/场景表征。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 02 上半身中心控制与移动操作接口，编号 **035/161**。
- **算法实现总结（公众号）：** CoorDex 把本体状态与关节序列、接触力/触觉信号转成可跟踪的身体目标，并通过PPO/RL 策略训练、扩散策略/流匹配、MM-DiT/Transformer 动作头训练或组合全身策略，最终输出全身轨迹/动作序列、低层控制器目标、地形/场景表征。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 对 wiki 的映射

- [paper-loco-manip-161-035-coordex](../../wiki/entities/paper-coordex-dexterous-humanoid-loco-manipulation.md)
- [loco-manip-161-category-02-upper-body-interface](../../wiki/overview/loco-manip-161-category-02-upper-body-interface.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
