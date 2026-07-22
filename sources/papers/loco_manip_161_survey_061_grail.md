# GRAIL

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 061/161）

- **标题：** GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors
- **类型：** paper
- **Loco-Manip 161 分类：** 03 视觉感知驱动的人形移动操作
- **机构：** NVIDIA、UCLA
- **项目页：** https://research.nvidia.com/labs/dair/grail/
- **arXiv：** https://arxiv.org/abs/2606.05160
- **代码：** https://github.com/NVlabs/GRAIL
- **发表日期：** 2026年6月3日
- **入库日期：** 2026-06-26
- **复核日期：** 2026-07-22
- **一句话说明：** GRAIL 先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用ACT/行为克隆模仿学习、扩散策略/流匹配、世界模型/视频预测生成末端执行器/腕手目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 03 视觉感知驱动的人形移动操作，编号 **061/161**。
- **算法实现总结（公众号）：** GRAIL 先从相机图像/多视角观测、本体状态与关节序列、人类视频/动捕轨迹恢复场景、目标或运动表征，再用ACT/行为克隆模仿学习、扩散策略/流匹配、世界模型/视频预测生成末端执行器/腕手目标。关键点是把动作生成看成条件生成问题，用扩散或流匹配在多模态动作分布里采样可执行轨迹。
- **论文复核修正：** GRAIL 更准确的主线是 **3D 资产 + VFM 先验 → metric 4D HOI 重建 → G1 retargeting / task-general tracking → egocentric RGB sim-to-real**；不是单纯 diffusion/action head 论文。
- **开源状态：** 官方 [NVlabs/GRAIL](https://github.com/NVlabs/GRAIL) 已开放 Docker quick start、pipeline modules、checkpoint 下载和 docs；Hugging Face 数据集已开放，manipulation dataset 仍在 README TODO 中。

## 对 wiki 的映射

- [paper-grail](../../wiki/entities/paper-grail.md)
- [loco-manip-161-category-03-visuomotor](../../wiki/overview/loco-manip-161-category-03-visuomotor.md)
- [loco-manip-contact-category-03-generative-data](../../wiki/overview/loco-manip-contact-category-03-generative-data.md)
- [grail-project](../sites/grail-project.md)
- [grail_nvlabs](../repos/grail_nvlabs.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 接触横切面编译：[wechat_embodied_ai_lab_loco_manip_contact_survey.md](../blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
