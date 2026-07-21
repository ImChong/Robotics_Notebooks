---
type: entity
tags: [paper, vision-transformer, visual-encoder, vlm, efficiency, linear-attention, xpeng]
status: complete
updated: 2026-07-21
arxiv: "2606.24253"
related:
  - ../concepts/vision-transformer.md
  - ../concepts/vision-backbones.md
  - ../methods/vla.md
  - ../queries/perception-backbone-selection.md
  - ./paper-x-world.md
  - ./paper-x-foresight.md
sources:
  - ../../sources/papers/turingvit_arxiv_2606_24253.md
  - ../../sources/sites/turingvit-github-io.md
summary: "TuringViT（arXiv:2606.24253，小鹏）：VLM-native 高效 ViT，用 TLA+VISTA-Curation+动态分辨率四阶段预训练，以约 10% 数据量超越 SigLIP2 等开源骨干；项目页截至入库日未列代码/权重。"
---

# TuringViT（Making SOTA Vision Transformers Accessible to All）

**TuringViT**（arXiv:2606.24253）由[小鹏（XPeng）](https://www.xiaopeng.com/)提出：面向 VLM/VLA 时代的 **可定制 SOTA 视觉编码器**，沿 **架构 / 数据 / 训练** 三轴协同设计，使高分辨率与动态分辨率预训练在可控算力下可完成，并成为小鹏 AI 系统公开叙事中的统一视觉底座。

## 一句话定义

**用线性注意力主导的 Turing Block、策展式图视频监督与原生动态分辨率四阶段配方，把「定制 SOTA ViT」从超算专属降到可复现工程预算。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ViT | Vision Transformer | 图像分块 + Transformer 视觉编码器 |
| TLA | Turing Linear Attention | TuringViT 主导的近线性注意力层 |
| MHA | Multi-Head Attention | 块内保留的全局 softmax 注意力层 |
| VISTA | Vision Data Curation via Image–Video Scoring and Temporal Aggregation | 图/视频数据策展管线 |
| VLM | Vision-Language Model | 下游多模态大模型，本工作的原生对接目标 |
| MIM | Masked Image Modeling | 阶段一特征蒸馏式掩码重建初始化 |

## 为什么重要

- **打中社区真实瓶颈：** 下游要定制延迟/分辨率/时序，却买不起 SigLIP2 级数据与二次方注意力预训练。
- **VLM-native 而非事后贴：** 动态分辨率从预训练第一天对齐下游用法，避免「低分辨率训完再适配」。
- **与小鹏驾驶栈同源：** 同一机构后续 [X-World](./paper-x-world.md) / [X-Foresight](./paper-x-foresight.md) 等世界模型与 VLA 工作共享「统一视觉基础」语境。

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 小鹏（XPeng） |
| arXiv | [2606.24253](https://arxiv.org/abs/2606.24253) |
| 项目页 | <https://turingvit.github.io/> |
| 变体 | TuringViT-18L（3 Blocks）/ 24L（4 Blocks） |
| 开源状态 | **未开源**（截至 2026-07-21 项目页无 GitHub/权重） |

## 核心原理

### 方法栈

| 模块 | 角色 |
|------|------|
| **Turing Block** | 5× TLA + 1× MHA；2D-RoPE + RMSNorm + SwiGLU |
| **VISTA-Curation** | 多候选 recap、相对打分、视频时序聚合 → 更密监督 |
| **四阶段预训练** | MIM 初始化 → 有界动态图文 → 无界高分辨率 → 图视频混合 |
| **目标** | SigLIP + SuperClass；后期 LVT 对齐视频嵌入 |

### 流程总览

```mermaid
flowchart LR
  web[噪声网页图文/视频] --> vista[VISTA-Curation]
  vista --> s1[Stage1 MIM 蒸馏]
  s1 --> s2[Stage2 有界动态图文]
  s2 --> s3[Stage3 无界高分辨率]
  s3 --> s4[Stage4 图视频混合]
  s4 --> enc[TuringViT 编码器]
  enc --> vlm[下游 VLM / VLA]
```

## 源码运行时序图

**不适用** — 截至 2026-07-21，[项目页](https://turingvit.github.io/)仅提供 Technical Report（arXiv），未列出可运行训练/推理仓库或权重入口。

## 评测要点

| 设定 | 要点（项目页 / 摘要） |
|------|------------------------|
| 零样本分类 | TuringViT-24L ImageNet-1K ≈ **83.9**；ImageNet-A **89.7**（相对 SigLIP2-L +5.4） |
| 检索 | COCO / Flickr30K 均值优于 SigLIP2-L |
| 数据量 | 预训练约 **0.85B** 对 vs SigLIP2-L **10B** |
| 延迟 | 高分辨率侧相对标准 ViT 更平坦（TensorRT FP16 @ 3080 Ti） |

## 与其他工作对比

| 对照 | 差异 |
|------|------|
| **SigLIP2 / Seed1.5-ViT** | 同为强开源视觉塔；TuringViT 强调 **线性注意力主导 + VLM-native 动态分辨率** 与更低数据预算 |
| **标准 ViT softmax** | 二次方注意力；TuringViT 以 5×TLA+1×MHA 换近线性缩放 |
| **事后分辨率适配** | 本工作从预训练起对齐下游动态分辨率，避免低分训完再贴 |

## 工程实践

| 项 | 要点 |
|------|------|
| 选型信号 | 需要 **动态分辨率 + 高分辨率延迟可控** 的 VLM 视觉塔时，对照 SigLIP2 / Seed1.5-ViT 读本页数字 |
| 数据预算叙事 | 项目页：约 **0.85B** 对样本 vs SigLIP2-L **10B** |
| 延迟 | FP16 TensorRT @ RTX 3080 Ti：相对标准 ViT，高分辨率侧更平坦 |
| 复现边界 | **配方公开、资产未公开**；勿假设可直接下载权重 |

## 局限与风险

- **开源缺口：** 无代码/权重则社区「可及」主要停留在方法叙述与指标对照。
- **机构数据优势：** VISTA 与 850M 对依赖内部策展管线，外部难以 1:1 复刻。
- **误区：** 把 TLA 当成「免费线性注意力」——块内仍保留稀疏 MHA 做全局路由，工程上要按 5:1 配比调延迟/精度。

## 关联页面

- [Vision Transformer](../concepts/vision-transformer.md) — ViT 机制与二次方注意力瓶颈
- [视觉骨干](../concepts/vision-backbones.md) — 预训练→VLA 视觉塔链条
- [感知骨干选型 Query](../queries/perception-backbone-selection.md) — 任务导向选型树
- [VLA](../methods/vla.md) — 下游策略对视觉塔的依赖
- [X-World](./paper-x-world.md) — 同机构多摄世界模型底座
- [X-Foresight](./paper-x-foresight.md) — 同机构驾驶 VLA + 预测式世界建模

## 参考来源

- [TuringViT 论文摘录（arXiv:2606.24253）](../../sources/papers/turingvit_arxiv_2606_24253.md)
- [TuringViT 项目页归档](../../sources/sites/turingvit-github-io.md)

## 推荐继续阅读

- 论文 PDF：<https://arxiv.org/pdf/2606.24253>
- 项目主页：<https://turingvit.github.io/>
