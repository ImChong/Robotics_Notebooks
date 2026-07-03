---
type: entity
tags: [paper, loco-manip-contact-survey, humanoid, loco-manipulation, vla, 3dgs, synthetic-data, unitree-g1, uc-berkeley]
status: complete
updated: 2026-07-03
arxiv: "2606.30645"
venue: "2026 · arXiv"
summary: "VLK 在 3DGS 重建场景中合成 48k 视觉-语言-全身运动学三元组，π₀.₅ 初始化 VLK 策略 + SceneBot tracker 在 G1 零样本真机 loco-manip。"
related:
  - ../overview/loco-manip-contact-technology-map.md
  - ../overview/loco-manip-contact-category-01-contact-data.md
  - ../overview/loco-manip-contact-category-03-generative-data.md
  - ./paper-scenebot.md
  - ./paper-legs-embodied-gaussian-splatting-vla.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/vlk_arxiv_2606_30645.md
  - ../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md
---

# VLK（Vision-Language-Kinematics）

**VLK**（arXiv:2606.30645）收录于 [具身智能研究室 · Loco-Manip 接触专题](https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw) **01 接触数据** 组（亦属生成式基础设施）：在 **3DGS 重建场景** 中合成 **视觉-语言-运动学** 监督，训练感知 loco-manip 策略。

## 一句话定义

iPhone 重建 metric-scale **3DGS** 室内场景 → 特权几何采样路点 → 条件扩散生成 **G1** 全身运动 → 回放渲染 ego 图像与语言指令；**48k** 合成轨迹训练 VLK 策略（π₀.₅ 初始化），**SceneBot** tracker 将运动学预测转为关节命令，**零样本**真机五类任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLK | Vision-Language-Kinematics | 本文提出的三联监督与策略名 |
| VLA | Vision-Language-Action | 骨干来自 π₀.₅ VLA 微调 |
| 3DGS | 3D Gaussian Splatting | 场景重建与 egocentric 渲染 |
| WBT | Whole-Body Tracking | SceneBot 作低层 tracker |
| Loco-Manip | Loco-Manipulation | 导航 + 单物体搬运等全身任务 |

## 为什么重要

- **完整三元组规模化：** 同时提供 ego 图像、语言与机器人可执行全身轨迹，填补感知 loco-manip 数据瓶颈。
- **与 LEGS 对照：** 同属 3DGS 数据工厂；VLK 强调 **重建场景内合成交互**，LEGS 强调 VLA 微调流水线。
- **执行闭环：** 显式依赖 [SceneBot](paper-scenebot.md) 将运动学预测落地，体现接触专题「数据→策略」链路。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 分组 | Loco-Manip 接触专题 · 01 接触数据 / 03 生成式补数 |
| 原文题目 | VLK: Learning Humanoid Loco-Manipulation from Synthetic Interactions in Reconstructed Scenes |
| 机构 | UC Berkeley；CMU；Shanghai AI Lab 等 |
| 论文/项目 | <https://arxiv.org/abs/2606.30645> · <https://vision-language-kinematics.github.io> |

## 与其他页面的关系

- 分类 hub：[loco-manip-contact-category-01-contact-data.md](../overview/loco-manip-contact-category-01-contact-data.md)
- Tracker：[SceneBot](paper-scenebot.md)
- 姊妹：[LEGS](paper-legs-embodied-gaussian-splatting-vla.md)

## 参考来源

- [vlk_arxiv_2606_30645.md](../../sources/papers/vlk_arxiv_2606_30645.md)
- [wechat_embodied_ai_lab_loco_manip_contact_survey.md](../../sources/blogs/wechat_embodied_ai_lab_loco_manip_contact_survey.md)

## 推荐继续阅读

- [SceneBot](paper-scenebot.md)
- [LEGS VLA 实体](paper-legs-embodied-gaussian-splatting-vla.md)
