# LeVERB

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 122/161）

- **标题：** LeVERB: Humanoid Whole-Body Control with Latent Vision-Language Instruction
- **类型：** paper
- **Loco-Manip 161 分类：** 06 特殊任务、接触规划与视觉闭环
- **机构：** University of California Berkeley、Norwegian University of Science and Technology、Simon Fraser University、Carnegie Mellon University
- **项目页：** https://github.com/ember-lab-berkeley/LeVERB-Website
- **发表日期：** 2025年9月25日
- **入库日期：** 2026-06-26
- **一句话说明：** LeVERB 的实现路径是先把语言指令、相机图像/多视角观测、仿真交互数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、潜变量/动作 token预测全身轨迹/动作序列、动作 chunk/token、低层控制器目标。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 核心摘录（策展，非全文）

- **在 161 篇地图中的位置：** 06 特殊任务、接触规划与视觉闭环，编号 **122/161**。
- **算法实现总结（公众号）：** LeVERB 的实现路径是先把语言指令、相机图像/多视角观测、仿真交互数据编码成多模态表征，再用ACT/行为克隆模仿学习、VLA 多模态动作模型、潜变量/动作 token预测全身轨迹/动作序列、动作 chunk/token、低层控制器目标。关键点是保留 VLM 的语义理解，同时增加机器人状态和动作头，避免只停留在语言规划。

## 对 wiki 的映射

- [paper-loco-manip-161-122-leverb](../../wiki/entities/paper-loco-manip-161-122-leverb.md)
- [loco-manip-161-category-06-contact-tasks](../../wiki/overview/loco-manip-161-category-06-contact-tasks.md)

## 参考来源（原始）

- 微信公众号编译：[wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md)
- 原始抓取：[wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)
