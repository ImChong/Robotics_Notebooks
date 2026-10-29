---
type: entity
tags: [paper, survey, world-models, video, embodied-ai]
status: complete
updated: 2026-10-27
project_id: video-world-model-survey
arxiv: "2610.31668"
project: https://vid-gen.github.io/video-world-model-survey
code: https://github.com/vid-gen/awesome-video-world-model
venue: "arXiv 2026"
summary: "以读取、生成、共享和交互四种操作统一梳理视频基础模型与视频世界模型，强调世界状态一致性和动作条件评测。"
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ./paper-sa-2606-20781-world-action-models-a-survey.md
sources:
  - ../../sources/papers/video_world_model_survey_2610_31668.md
  - ../../sources/sites/video-world-model-survey.md
  - ../../sources/repos/awesome-video-world-model.md
---

# 视频世界模型综述：Reading, Writing, Sharing, and Interacting

## 一句话定义

这篇综述以世界建模为视角，将视频理解、视频生成、理解与生成的统一以及交互预测组织为四种互补操作。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 对世界状态及其演化进行建模 |
| VWM | Video World Model | 基于视频的世界模型 |
| VLA | Vision-Language-Action | 视觉、语言与动作模型 |

## 核心信息

| 字段 | 内容 |
|------|------|
| 论文 | [arXiv:2610.31668v1](https://arxiv.org/abs/2610.31668v1)，2026-10-22 |
| 机构 | 牛津大学、新加坡国立大学、南洋理工大学、加州大学默塞德分校 |
| 作者 | Shengqiong Wu、Meng Luo（共同一作）、Bobo Li、Ziwei Liu、Shuicheng Yan、Ming-Hsuan Yang、Philip Torr、Hao Fei（项目负责人、通讯） |
| 规模 | 102 页、17 幅图、19 张表 |

## 核心框架

| 操作 | 问题 |
|------|------|
| Reading | 从观测视频推断世界属性 |
| Writing | 依据给定或学习到的条件生成视频 |
| Sharing | 通过共同建模机制整合理解与生成 |
| Interacting | 结合动作与持续观测预测、更新世界状态 |

视频是潜在动态世界的部分且有损的投影。画面在时间上平滑，并不自动保证遮挡物体、物理关系与世界状态在长时间内保持一致。

## 对机器人研究的启发

交互模型可按**观测空间或潜状态空间**，以及**被动预测、动作条件预测、交互模型**两轴分类。机器人应用包括合成数据、想象中训练策略、策略评测及预测后规划；需要分别验证动作可控性、长程状态一致性、物理和因果合理性与实时响应。

## 评测与指标

综述区分针对技术挑战的评测与面向应用的评测。对机器人而言，视频视觉质量不足以证明控制有效，应进一步检查动作干预下的状态变化和下游任务表现；不同基准的设定与指标应回到原文核对。

## 与其他综述对比

| 综述 | 组织轴 | 与本综述的差异 |
|------|--------|----------------|
| [World Action Models 综述](./paper-sa-2606-20781-world-action-models-a-survey.md) | 世界模型、视频生成、动作条件视频世界模型、VLA 与 WAM 的边界 | 聚焦「动作」一端，大致对应本综述的 Interacting；本综述还覆盖不涉及动作的 Reading / Writing / Sharing |
| [具身世界模型综述](./paper-embodied-world-model-survey.md) | 表示、训练目标、规划/策略用法与评测 | 以机器人决策为出发点；本综述以视频为出发点，机器人只是 Interacting 的一类应用 |
| [扩散式世界模型综述](./paper-diffusion-based-world-models-survey.md) | 按驾驶/具身/通用三域划分扩散类方法 | 按生成范式（扩散）切分；本综述按操作类型切分，不限定生成范式 |
| [3D/4D 世界建模综述](./paper-sa-2509-07996-3d-and-4d-world-modeling-a-survey.md) | VideoGen / OccGen / LiDARGen | 覆盖占据栅格与 LiDAR 等非视频模态；本综述只讨论视频，但把理解与生成放在同一框架 |

用法建议：需要理解视频世界模型全貌时先读本综述的四操作框架；为机器人选型时，再转到 WAM 或具身世界模型综述，看动作条件与决策相关的评测。

## 结论

**该综述为视频理解、生成与交互提供共同的世界建模坐标，而非提出一个新的机器人控制模型。**

- 选型时先确定需要理解视频、生成视频，还是预测动作后果。
- 长时预测需关注世界状态，而不仅是相邻帧平滑。
- 机器人应用应将动作条件响应和下游任务效果纳入评测。
- 通过资源库追踪具体工作时，逐项核实代码、权重及实验设定。

## 源码运行时序图

不适用：官方仓库是文献清单与参考文献，不是可运行的模型训练或推理实现。

## 局限与开放状态

这是 arXiv 预印本综述；官方 [资源库](https://github.com/vid-gen/awesome-video-world-model)公开文献列表。2026-10-27 项目页无法访问，无法直接核查其资源链接，也不能据此推断各项模型的开源状态。

## 关联页面

- [世界–动作模型](../concepts/world-action-models.md)
- [生成式世界模型](../methods/generative-world-models.md)
- [World Action Models 综述](./paper-sa-2606-20781-world-action-models-a-survey.md)

## 参考来源

- [论文归档](../../sources/papers/video_world_model_survey_2610_31668.md)
- [项目页归档](../../sources/sites/video-world-model-survey.md)
- [资源库归档](../../sources/repos/awesome-video-world-model.md)

## 推荐继续阅读

- [论文 HTML 全文](https://arxiv.org/html/2610.31668v1)
- [官方资源库](https://github.com/vid-gen/awesome-video-world-model)
