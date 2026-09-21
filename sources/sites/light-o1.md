# Light-O1 项目页（lightorigins.com）

- **标题：** Light-O1: Scaling Whole-Body Intelligence with Human Action Pretraining
- **类型：** site（官方 Tech Blog）
- **机构：** 亮源新创（Light Origins）
- **项目页：** <https://www.lightorigins.com/en/blog/light-o1>
- **代码：** <https://github.com/lightorigins/Light-O1>
- **预览模型：** <https://huggingface.co/LightOriginsHQ/Light-O1-Preview>
- **Playground：** <https://huggingface.co/spaces/LightOriginsHQ/Light-O1-Preview-playground>
- **入库日期：** 2026-09-21

## 开源核查（入库日）

| 资产 | 状态 |
|------|------|
| Tech Blog | **已上线** |
| 推理代码 | **已开源** `lightorigins/Light-O1`（Apache-2.0） |
| Light-O1-Preview 权重 | **已发布** Hugging Face `LightOriginsHQ/Light-O1-Preview` |
| 完整 Light-O1 预训练 / loco-manip 权重 | **未公开** |
| 论文 / arXiv | **未列** |

## 能力摘要

- **问题：** 机器人专项采集难扩展；人类视频可提供互补的动作知识，但本体/视角/控制差异阻碍迁移。
- **方法：** 从视频恢复结构化人类动作 → 高保真 action tokenization → 语言/视觉/动作交错序列上的自回归预训练 → 目标本体 post-training。
- **Scaling Law：** 预训练 multimodal token 预算 D 至 120B（≈10 万动作小时）后，适配 Nymeria / HIW-500 / LightBot held-out 数据的 next-action loss 与开环 MPJPE 呈幂律改善。
- **范式位置：** 亮源新创「规模化预训练 — 规模化对齐 — 规模化部署」中 **预训练段** 首个公开模型（对齐段见 [LightNav-0](./lightnav-0.md)，部署段见 [Light REACT](./light-react.md)）。

## 交叉链接

- 微信归档：[wechat_lightorigins_light_o1_2026-09-21](../blogs/wechat_lightorigins_light_o1_2026-09-21.md)
- 主实体：[Light-O1](../../wiki/entities/light-o1.md)
- 代码归档：[lightorigins-light-o1.md](../repos/lightorigins-light-o1.md)
- 同机构：[LightNav-0](./lightnav-0.md)、[Light REACT 项目发布](./light-react.md)
