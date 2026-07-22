# MotionWAM

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 100/161）

- **标题：** MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation
- **类型：** paper
- **Loco-Manip 161 分类：** 04 生成式运动、语言控制与轨迹规划
- **机构：** Mondo Robotics、HKUST (GZ)、HKUST
- **项目页：** https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset
- **arXiv：** https://arxiv.org/abs/2606.09215
- **代码：** 未发现官方代码或项目页（2026-07-22 核查）
- **发表日期：** 2026年6月8日
- **入库日期：** 2026-06-26
- **复核日期：** 2026-07-22
- **一句话说明：** MotionWAM 的实现路径是先把相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、扩散策略/流匹配、VLA 多模态动作模型预测全身轨迹/动作序列、低层控制器目标、地形/场景表征。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 04 生成式运动、语言控制与轨迹规划，编号 **100/161**。
- **算法实现总结（公众号）：** MotionWAM 的实现路径是先把相机图像/多视角观测、本体状态与关节序列编码成多模态表征，再用ACT/行为克隆模仿学习、扩散策略/流匹配、VLA 多模态动作模型预测全身轨迹/动作序列、低层控制器目标、地形/场景表征。关键点是让视频/世界模型提供可预测的物理先验，再由动作头把语义目标变成连续控制。
- **论文复核修正：** MotionWAM 的核心不是一般 VLA action head，而是 **Video DiT hidden state → Motion DiT → SONIC unified motion token**；统一动作空间覆盖 locomotion、torso、height、foot interaction 与 hand manipulation。
- **结果与开源：** G1 九项真机任务平均 **76.1%**，GR00T-N1.7 **43.9%**；A100 推理 **4.9 Hz**。arXiv 未列官方项目页或代码。

## 对 wiki 的映射

- [paper-loco-manip-161-100-motionwam](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md)
- [loco-manip-161-category-04-generative-language-trajectory](../../wiki/overview/loco-manip-161-category-04-generative-language-trajectory.md)
- [loco-manip-contact-category-05-vla-world-models](../../wiki/overview/loco-manip-contact-category-05-vla-world-models.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 接触横切面编译：[wechat_embodied_ai_lab_loco_manip_contact_survey.md](../blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
- arXiv：<https://arxiv.org/abs/2606.09215>
