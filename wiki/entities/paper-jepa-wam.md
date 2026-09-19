---
type: entity
tags:
  - paper
  - vla
  - world-model
  - jepa
status: complete
updated: 2026-09-19
arxiv: "2608.09381"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md
sources:
  - ../../sources/papers/jepa_wam_arxiv_2608_09381.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md
summary: "V-JEPA 潜空间联合嵌入世界动作模型：共享预测器学习视觉变化与连续动作，可接入已有 VLA 而不改感知–动作通路。"
---

# JEPA-WAM（arXiv:2608.09381）

**JEPA-WAM**（[arXiv:2608.09381](https://arxiv.org/abs/2608.09381)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第一篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) **世界模型** 段。

## 一句话定义

**V-JEPA 潜空间联合嵌入世界动作模型：共享预测器学习视觉变化与连续动作，可接入已有 VLA 而不改感知–动作通路。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| TTT | Test-Time Training | 部署阶段无标注数据的在线适配 |

## 为什么重要

- V-JEPA 潜空间联合嵌入世界动作模型：共享预测器学习视觉变化与连续动作，可接入已有 VLA 而不改感知–动作通路。
- 开源状态：**待核实**（步骤 2.5，入库日 2026-09-19）。
- 与 [vla weekly trends 2026 08 10 part1 technology map](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md) 同批工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.09381](https://arxiv.org/abs/2608.09381) |
| **项目页** | https://spritewithoutice.github.io/JEPA_WAM |
| **开源** | **待核实** |
| **文内评测** | LIBERO、LIBERO-Plus、RoboTwin 2.0；实机 AgileX COBOT Magic 双臂 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

- **文内口径：** LIBERO、LIBERO-Plus、RoboTwin 2.0；实机 AgileX COBOT Magic 双臂
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

> 下表只做**定位对照**：本页与下列同批各页均为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 的索引级摘要，未逐条核对原文实验表，彼此**不共享同一评测协议**，不可据此横比数字。

| 对照 | 差异读法 |
|------|----------|
| [WAM-Diff2](./paper-wam-diff2.md) | 同批「世界模型」段，未来信号的处理位置相反：本文在 V-JEPA **潜空间**里用共享预测器同时学视觉变化与连续动作，WAM-Diff2 走的是把自回归蒸馏成离散扩散的**解码器**改造。一个改表征，一个改解码 |
| [World Tokens](./paper-world-tokens-inference-trimmed-wam.md) | 同批同为降低世界模型的部署代价：World Tokens 训练期加世界监督、推理期**裁掉**生成分支；本文干脆不生成像素，预测停在潜空间。目标一致，一个事后裁，一个事前不长 |
| [ω-0](./paper-omega-0.md) | 同批同为潜空间未来 embedding 条件的 WAM，落点不同：ω-0 面向人形全身 loco-manipulation，本文强调**可接入已有 VLA 而不改感知–动作通路**——那是一条集成成本主张，不是能力主张，选型时别当成能力对比 |
| [World Action Models](../concepts/world-action-models.md) | 该页给 WAM 的概念谱系；本文落在「联合嵌入预测 + 不改下游通路」这一支，与生成式像素 rollout 一支的取舍是**保真度 vs 墙钟与集成成本** |

## 结论

**JEPA-WAM 适合作为本期「世界模型」路线的快速索引页。**

1. 核心贡献：V-JEPA 潜空间联合嵌入世界动作模型：共享预测器学习视觉变化与连续动作，可接入已有 VLA 而不改感知–动作通路。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part1.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md)
- [arXiv:2608.09381](https://arxiv.org/abs/2608.09381)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.09381)
