# DIAL

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 147/161）

- **标题：** DIAL: Decoupling Intent and Action via Latent World Modeling for End-to-End VLA
- **类型：** paper
- **Loco-Manip 161 分类：** 09 人形 VLA、世界模型与通用操作
- **机构：** The University of Hong Kong、XPENG Robotics、University of North Carolina at Chapel Hill、Limited Robot Data
- **项目页：** https://xpeng-robotics.github.io/dial
- **发表日期：** 2026年4月28日
- **入库日期：** 2026-06-26
- **一句话说明：** DIAL 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用扩散策略/流匹配、VLA 多模态动作模型、VLM 语义规划/路由预测全身轨迹/动作序列、动作 chunk/token、地形/场景表征。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 09 人形 VLA、世界模型与通用操作，编号 **147/161**。
- **算法实现总结（公众号）：** DIAL 的实现路径是先把语言指令、相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用扩散策略/流匹配、VLA 多模态动作模型、VLM 语义规划/路由预测全身轨迹/动作序列、动作 chunk/token、地形/场景表征。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。

## 对 wiki 的映射

- [paper-loco-manip-161-147-dial](../../wiki/methods/dial-instruction-augmentation.md)
- [loco-manip-161-category-09-vla-world-models](../../wiki/overview/loco-manip-161-category-09-vla-world-models.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
