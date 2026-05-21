# lucidrains/mimic-video

> 来源归档

- **标题：** Mimic-Video（社区 PyTorch 实现草案）
- **类型：** repo
- **作者：** Phil Wang（lucidrains 组织）
- **代码：** <https://github.com/lucidrains/mimic-video>
- **论文：** <https://arxiv.org/abs/2512.15692>
- **项目页：** <https://mimic-video.github.io/>
- **入库日期：** 2026-05-17
- **一句话说明：** 面向 **mimic-video / VAM** 论文的 **非官方** 教学式复现仓库（README 级入口：声明 SOTA 泛化机器人控制方向、链回论文与官方项目页）；**权重、训练脚本成熟度与上游 mimic 官方发布** 以克隆时两个仓库的 README 为准，本知识库不缓存易过期命令行。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [mimic-video（Video-Action Model）](../../wiki/methods/mimic-video.md) | 方法级归纳页：架构直觉、训练分工、与 VLA / 像素视频策略对照 |
| [VLA](../../wiki/methods/vla.md) | 对照阅读：静态多模态骨干 vs 视频动力学先验 + 轻量动作头 |
| [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) | 相关概念谱系：此处视频模型主要提供 **潜空间计划表征**，推理默认不强调完整像素 rollout |

## 对 wiki 的映射

- 沉淀 **[`wiki/methods/mimic-video.md`](../../wiki/methods/mimic-video.md)**；论文式摘录见 [`sources/papers/mimic_video_arxiv_2512_15692.md`](../papers/mimic_video_arxiv_2512_15692.md)。
