---
type: entity
tags:
  - paper
  - vla
  - world-model
  - autonomous-driving
  - distillation
status: complete
updated: 2026-09-19
arxiv: "2608.01035"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md
sources:
  - ../../sources/papers/wam_diff2_arxiv_2608_01035.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md
summary: "层次 AR→扩散蒸馏的高效智驾 VLA：分块适配/蒸馏/跨尺度整模蒸馏，将自回归预训练转为离散扩散 VLA，并行解码减轻暴露偏差。"
---

# WAM-Diff2（arXiv:2608.01035）

**WAM-Diff2**（[arXiv:2608.01035](https://arxiv.org/abs/2608.01035)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第一篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) **世界模型** 段。

## 一句话定义

**层次 AR→扩散蒸馏的高效智驾 VLA：分块适配/蒸馏/跨尺度整模蒸馏，将自回归预训练转为离散扩散 VLA，并行解码减轻暴露偏差。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| TTT | Test-Time Training | 部署阶段无标注数据的在线适配 |

## 为什么重要

- 层次 AR→扩散蒸馏的高效智驾 VLA：分块适配/蒸馏/跨尺度整模蒸馏，将自回归预训练转为离散扩散 VLA，并行解码减轻暴露偏差。
- 开源状态：**待核实**（步骤 2.5，入库日 2026-09-19）。
- 与 [vla weekly trends 2026 08 10 part1 technology map](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md) 同批工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.01035](https://arxiv.org/abs/2608.01035) |
| **开源** | **待核实** |
| **文内评测** | NAVSIM、Bench2Drive、LingoQA、DriveBench、COCO |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

- **文内口径：** NAVSIM、Bench2Drive、LingoQA、DriveBench、COCO
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

> 下表只做**定位对照**：本页与下列同批各页均为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 的索引级摘要，未逐条核对原文实验表，彼此**不共享同一评测协议**，不可据此横比数字。

| 对照 | 差异读法 |
|------|----------|
| [JEPA-WAM](./paper-jepa-wam.md) | 同批「世界模型」段，改的位置不同：本文把自回归预训练**蒸馏成离散扩散**、靠并行解码减轻暴露偏差，JEPA-WAM 换的是预测所在的**表征空间**（V-JEPA 潜空间）。一个改解码，一个改表征 |
| [World Tokens](./paper-world-tokens-inference-trimmed-wam.md) | 同批同为降低世界模型的部署代价，手段不同：World Tokens 训练期加世界监督、推理期**裁掉生成分支**；本文保留生成但换成可并行的离散扩散。一个是不生成，一个是生成得更快 |
| [Depth-Wise Probing Driving VLA](./paper-depth-wise-probing-driving-vla.md) / [CMU-Drive / V2V-VLA](./paper-cmu-drive-v2v-vla.md) | 同批三条智驾 VLA 的不同取舍：本文改解码范式，Depth-Wise 剪层，CMU-Drive 加协同与通信头。选型先确认瓶颈落在暴露偏差、时延还是多车信息缺失 |
| [Diffusion Policy](../methods/diffusion-policy.md) / [Generative World Models](../methods/generative-world-models.md) | 前者给扩散式动作生成的机制背景，后者给「生成未来 → 驱动动作」一族的谱系；本文的特别之处是**离散**扩散且**由 AR 蒸馏而来**，不是从头训一个扩散头——分块 / 跨尺度整模蒸馏是这条路径的工程主体 |

## 结论

**WAM-Diff2 适合作为本期「世界模型」路线的快速索引页。**

1. 核心贡献：层次 AR→扩散蒸馏的高效智驾 VLA：分块适配/蒸馏/跨尺度整模蒸馏，将自回归预训练转为离散扩散 VLA，并行解码减轻暴露偏差。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part1.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md)
- [arXiv:2608.01035](https://arxiv.org/abs/2608.01035)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.01035)
