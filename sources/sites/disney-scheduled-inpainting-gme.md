# Disney Scheduled Inpainting / Interactive Generative Motion Editing — 项目页

- **来源：** <https://studios.disneyresearch.com/2026/07/30/interactive-generative-motion-editing-via-scheduled-inpainting/>
- **类型：** site（机构研究页）
- **机构：** DisneyResearch\|Studios · ETH Zürich
- **归档日期：** 2026-08-21
- **论文 PDF：** <https://studios.disneyresearch.com/app/uploads/2026/07/Interactive-Generative-Motion-Editing-via-Scheduled-Inpainting-Paper.pdf>
- **arXiv：** <https://arxiv.org/abs/2607.29133>

## 一句话说明

Disney Research Studios 发布 **scheduled inpainting**：在预训练 **direct-manipulation 运动扩散模型** 上做 **training-free 推理编辑**，统一 MoCap/片段 **保留** 与 **生成式结构性改写**（延长、拼接、合成、直接拖拽约束），面向 VFX/游戏 motion editing 工作流。

## 为什么值得保留

- 官方入口含摘要、作者、PDF 与 arXiv；与 [Generative Motion Rig](../papers/generative_motion_rig_siggraph_talks_2026.md) 同属 Disney **Neural Motion Rig** 谱系，但本文聚焦 **exemplar clip 编辑** 而非 Blender 插件集成。
- 对机器人知识库：说明 **生成式运动 prior** 如何用于 **非破坏性 clip 编辑**——与 [机器人关键帧编辑工具](../../wiki/entities/robot-motion-keyframe-editors.md)（确定性残差/贝塞尔）形成方法对照。

## 开源状态（2026-08-21 核查）

| 项 | 结论 |
|----|------|
| 项目页资源 | PDF + arXiv；**无代码仓库** |
| 复现 | 以论文算法描述与视频为准 |

## 对 wiki 的映射

1. **[Scheduled Inpainting / GME（实体页）](../../wiki/entities/paper-scheduled-inpainting-gme.md)**
2. **[sources/papers/scheduled_inpainting_arxiv_2607_29133.md](../papers/scheduled_inpainting_arxiv_2607_29133.md)**
3. **[Generative Motion Rig（Disney）](../../wiki/entities/generative-motion-rig.md)** — 同组 DCC 集成对照
