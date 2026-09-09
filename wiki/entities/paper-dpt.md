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
  - ../concepts/state-estimation.md
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

## 工程实践

| 项 | 建议 |
|----|------|
| **单目深度** | 直接用 `isl-org/DPT` 预训练与推理脚本 |
| **立体（NBS 式）** | 等 NBS 官方代码；结构为 ViT 特征 → DPT head → disparity |
| **与 DINOv2** | NBS 用 DINOv2 初始化 ViT-L，再接 DPT 式多尺度融合 |

## 关联页面

- [NBS（No Bias Stereo）](./paper-nbs-no-bias-stereo.md) — 使用 DPT 视差头
- [DINOv2](./paper-dinov2.md) — 常见骨干组合
- [立体匹配基础模型](../methods/stereo-matching-foundation-models.md)

## 参考来源

- [DPT 论文摘录](../../sources/papers/dpt_arxiv_2103_13413.md)
- [DPT 官方仓](../../sources/repos/dpt.md)
- Ranftl et al., *Vision Transformers for Dense Prediction* — <https://arxiv.org/abs/2103.13413>

## 推荐继续阅读

- 官方仓库：<https://github.com/isl-org/DPT>
- NBS 项目页（DPT head 描述）：<https://intrinsic-experimental.github.io/nbs-website/>
