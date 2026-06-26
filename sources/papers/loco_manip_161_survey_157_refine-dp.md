# REFINE-DP

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 157/161）

- **标题：** REFINE-DP: Diffusion Policy Fine-tuning for Humanoid Loco-manipulation via Reinforcement Learning
- **类型：** paper
- **Loco-Manip 161 分类：** 09 人形 VLA、世界模型与通用操作
- **机构：** IEEE ROBOTICS AND AUTOMATION LETTERS
- **项目页：** https://refine-dp.github.io/REFINE-DP/
- **发表日期：** 2026年3月17日
- **入库日期：** 2026-06-26
- **一句话说明：** REFINE-DP 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用PPO/RL 策略训练、ACT/行为克隆模仿学习、扩散策略/流匹配预测关节位置/力矩命令、低层控制器目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 09 人形 VLA、世界模型与通用操作，编号 **157/161**。
- **算法实现总结（公众号）：** REFINE-DP 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用PPO/RL 策略训练、ACT/行为克隆模仿学习、扩散策略/流匹配预测关节位置/力矩命令、低层控制器目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 对 wiki 的映射

- [paper-loco-manip-161-157-refine-dp](../../wiki/entities/paper-loco-manip-161-157-refine-dp.md)
- [loco-manip-161-category-09-vla-world-models](../../wiki/overview/loco-manip-161-category-09-vla-world-models.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
