---
type: entity
tags: [paper, dense-prediction, depth-estimation, vision-transformer, monocular-depth]
status: complete
updated: 2026-09-09
arxiv: "2103.13413"
code: https://github.com/isl-org/DPT
related:
  - ./paper-dinov2.md
  - ./paper-nbs-no-bias-stereo.md
  - ../methods/stereo-matching-foundation-models.md
  - ../methods/unet.md
  - ../methods/fcn-semantic-segmentation.md
  - ../concepts/state-estimation.md
  - ../concepts/vision-transformer.md
  - ./eth3d-stereo-benchmark.md
sources:
  - ../../sources/papers/dpt_arxiv_2103_13413.md
  - ../../sources/repos/dpt.md
summary: "DPT（arXiv:2103.13413）：ViT 多尺度特征融合密集预测头，用于深度/分割；NBS 立体匹配用作视差解码；isl-org/DPT 已开源。"
---

# DPT：Vision Transformers for Dense Prediction

**DPT**（*Vision Transformers for Dense Prediction*，[arXiv:2103.13413](https://arxiv.org/abs/2103.13413)，[代码](https://github.com/isl-org/DPT)）将 **Vision Transformer** 各层特征通过 **多尺度融合模块** 解码为 **像素级密集预测**（深度、表面法线、语义分割等）。[NBS](./paper-nbs-no-bias-stereo.md) 在 **无相关体 ViT 立体匹配** 中采用 **DPT 风格视差头** 输出 sub-pixel disparity。

## 一句话定义

**把 ViT 的层次特征重新组合成密集图，是「Transformer 骨干 + 像素任务」之间的标准适配器。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DPT | Dense Prediction Transformer | 本文密集预测头框架 |
| ViT | Vision Transformer | 上游骨干 |
| SSL | Self-Supervised Learning | 常与 DINO 等骨干联用 |
| ReLU | Rectified Linear Unit | 融合块常用激活 |

## 核心信息

| 字段 | 内容 |
|------|------|
| **arXiv** | [2103.13413](https://arxiv.org/abs/2103.13413) |
| **代码** | [isl-org/DPT](https://github.com/isl-org/DPT)（**已开源**） |
| **典型用途** | 单目深度、立体视差解码（如 NBS）、分割 |

## 为什么重要

- **密集预测标准件：** 在 monocular depth 时代即为主流解码范式；进入 ViT 时代后仍是最常见的 **head 设计** 之一。
- **NBS 关键组件：** NBS 去掉 correlation volume，把 **匹配 + 解码** 全部交给 ViT attention + **DPT 视差头** — 理解 NBS 需同时读 [DINOv2](./paper-dinov2.md) 与本文。

## 实验与评测

> **归档口径提醒：** 本库的 [DPT 论文摘录](../../sources/papers/dpt_arxiv_2103_13413.md) 是**书目型归档**，未收录原文指标表。下面只写「在哪些轴上被评」与**库内可核的间接证据**，具体数值以 [原文](https://arxiv.org/abs/2103.13413) 为准，不要引用本页当数据源。

| 评测轴 | 说明 | 库内可核证据 |
|--------|------|-------------|
| **单目深度** | 原文主任务，与卷积解码器在同一 ViT/骨干设定下对比 | 归档未收录数值 |
| **语义分割** | 原文的第二类密集任务，验证 head 不是只对深度有效 | 归档未收录数值 |
| **立体视差解码** | 本库更关心的用法：作为视差头而非独立模型 | [NBS](./paper-nbs-no-bias-stereo.md) 用 DPT 风格视差头，在 ETH3D 报 EPE 0.09 / bad@1 0.16 / bad@4 0.02（见 [ETH3D 页](./eth3d-stereo-benchmark.md)） |

**读法：** NBS 的榜位是 **DINOv2 骨干 + 交替 attention + DPT 头** 的联合结果，DPT 单独贡献多少原文未拆分——不要把 NBS 的 ETH3D 数字读成「DPT 头的成绩」。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| **卷积式密集解码头**（FPN / U-Net 式上采样，DPT 要替代的默认做法） | 差别在**全局感受野从哪来**：卷积解码靠逐级上采样 + skip 连接把局部特征拼回分辨率，DPT 直接重组 ViT 各层 token 成多尺度图，每一层本身已是全局注意力的产物。代价是依赖 ViT 骨干，纯 CNN 栈用不上 |
| [U-Net](../methods/unet.md) / [FCN 语义分割](../methods/fcn-semantic-segmentation.md) | 库内的经典密集预测对照：同为「骨干 + 解码回像素」，但为卷积特征金字塔设计；把它们直接接到 ViT 上会丢掉 token 的层次差异，这正是 DPT 的动机 |
| [DINOv2](./paper-dinov2.md) | **互补而非竞争**：DINOv2 解决骨干怎么预训练，DPT 解决骨干特征怎么变成像素图。二者常成对出现，NBS 就是这个组合 |
| [NBS（No Bias Stereo）](./paper-nbs-no-bias-stereo.md) | 下游使用者：NBS 的主张是连 correlation volume 都不要，把匹配交给 attention、解码交给 DPT 头。读本页是为了读懂 NBS 的解码侧 |
| [立体匹配基础模型与基准生态](../methods/stereo-matching-foundation-models.md) | 谱系位置：DPT 不是一条立体匹配路线，而是多条路线共用的**解码组件**；在该页的方法谱系表里它出现在「骨干与头」而非方法行 |
| [Vision Transformer](../concepts/vision-transformer.md) | 上游机制底座：理解 DPT 为什么按层取特征，要先理解 ViT 各层 token 的性质差异 |

## 工程实践

| 项 | 建议 |
|----|------|
| **单目深度** | 直接用 `isl-org/DPT` 预训练与推理脚本 |
| **立体（NBS 式）** | 等 NBS 官方代码；结构为 ViT 特征 → DPT head → disparity |
| **与 DINOv2** | NBS 用 DINOv2 初始化 ViT-L，再接 DPT 式多尺度融合 |

## 结论

**总判：DPT 是「ViT 骨干 → 像素任务」之间的标准适配器，本库把它当组件读，而不是当一条可选的深度/立体路线。**

1. **选型层级要分清** — 换 DPT 头解决的是**解码**问题；如果误差来自匹配或骨干表征，换头不会有增益。
2. **与骨干成对决策** — DPT 的收益依赖骨干各层特征本身有层次差异，配 DINOv2 类自监督骨干是常见组合。
3. **数值别从本页引** — 归档为书目型，未收录原文指标；NBS 的 ETH3D 成绩是联合结果，不可拆给 DPT。
4. **可部署** — `isl-org/DPT` 已开源，单目深度可直接跑；立体视差用法仍要等 NBS 官方代码发布。

## 关联页面

- [NBS（No Bias Stereo）](./paper-nbs-no-bias-stereo.md) — 使用 DPT 视差头
- [DINOv2](./paper-dinov2.md) — 常见骨干组合
- [立体匹配基础模型](../methods/stereo-matching-foundation-models.md) — 谱系里的「骨干与头」位置
- [U-Net](../methods/unet.md) / [FCN 语义分割](../methods/fcn-semantic-segmentation.md) — 卷积式密集解码对照
- [Vision Transformer](../concepts/vision-transformer.md) — 上游机制底座

## 参考来源

- [DPT 论文摘录](../../sources/papers/dpt_arxiv_2103_13413.md)
- [DPT 官方仓](../../sources/repos/dpt.md)
- Ranftl et al., *Vision Transformers for Dense Prediction* — <https://arxiv.org/abs/2103.13413>

## 推荐继续阅读

- 官方仓库：<https://github.com/isl-org/DPT>
- NBS 项目页（DPT head 描述）：<https://intrinsic-experimental.github.io/nbs-website/>
