# HEX

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 038/161）

- **标题：** HEX: Humanoid-Aligned Experts for Cross-Embodiment Whole-Body Manipulation
- **类型：** paper
- **Loco-Manip 161 分类：** 02 上半身中心控制与移动操作接口
- **机构：** Beijing Innovation Center of Humanoid Robotics、Xi’an Jiaotong University、Nankai University、Peking University
- **项目页：** https://hex-humanoid.github.io/
- **发表日期：** 2026年5月19日
- **入库日期：** 2026-06-26
- **一句话说明：** HEX 的实现路径是先把本体状态与关节序列编码成多模态表征，再用PPO/RL 策略训练、VLA 多模态动作模型、VLM 语义规划/路由预测全身轨迹/动作序列、低层控制器目标。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 02 上半身中心控制与移动操作接口，编号 **038/161**。
- **算法实现总结（公众号）：** HEX 的实现路径是先把本体状态与关节序列编码成多模态表征，再用PPO/RL 策略训练、VLA 多模态动作模型、VLM 语义规划/路由预测全身轨迹/动作序列、低层控制器目标。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 对 wiki 的映射

- [paper-loco-manip-161-038-hex](../../wiki/entities/paper-loco-manip-161-038-hex.md)
- [loco-manip-161-category-02-upper-body-interface](../../wiki/overview/loco-manip-161-category-02-upper-body-interface.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
