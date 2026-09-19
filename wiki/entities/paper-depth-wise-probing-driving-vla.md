---
type: entity
tags:
  - paper
  - vla
  - autonomous-driving
  - efficiency
status: complete
updated: 2026-09-19
arxiv: "2608.07361"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md
sources:
  - ../../sources/papers/depth_wise_probing_driving_vla_arxiv_2608_07361.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md
summary: "智驾 VLA 规划 token 逐层探针：导航意图与规划信息在浅层已出现，深层层主要负责对齐规划器表示；学习式早读与剪层可提速且未见明确任务退化（DriveX@ECCV 2026）。"
---

# Depth-Wise Probing Driving VLA（arXiv:2608.07361）

**Depth-Wise Probing Driving VLA**（[arXiv:2608.07361](https://arxiv.org/abs/2608.07361)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第一篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) **性能提升** 段。

## 一句话定义

**智驾 VLA 规划 token 逐层探针：导航意图与规划信息在浅层已出现，深层层主要负责对齐规划器表示；学习式早读与剪层可提速且未见明确任务退化（DriveX@ECCV 2026）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| TTT | Test-Time Training | 部署阶段无标注数据的在线适配 |

## 为什么重要

- 智驾 VLA 规划 token 逐层探针：导航意图与规划信息在浅层已出现，深层层主要负责对齐规划器表示；学习式早读与剪层可提速且未见明确任务退化（DriveX@ECCV 2026）。
- 开源状态：**待核实**（步骤 2.5，入库日 2026-09-19）。
- 与 [vla weekly trends 2026 08 10 part1 technology map](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md) 同批工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.07361](https://arxiv.org/abs/2608.07361) |
| **开源** | **待核实** |
| **文内评测** | Bench2Drive |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

- **文内口径：** Bench2Drive
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

> 下表只做**定位对照**：本页与下列同批各页均为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 的索引级摘要，未逐条核对原文实验表，彼此**不共享同一评测协议**，不可据此横比数字。

| 对照 | 差异读法 |
|------|----------|
| [VLA Depth Decodability](./paper-vla-action-post-training-depth-decodability.md) | 同批两篇**逐层探针**诊断，结论指向相反的用法：本文探规划 token，发现导航意图与规划信息**浅层已出现**，指向剪层提速；那篇探深度信息，发现动作后训练令其全层退化，指向把被削掉的写入修回来。同一诊断工具，一个用于减，一个用于修 |
| [WA-SpecDec](./paper-wa-specdec.md) | 同批同为提速，省的维度不同：本文省**层数**（深度方向早读），WA-SpecDec 省**串行步数**（投机解码）。两条正交，可叠加 |
| [CMU-Drive / V2V-VLA](./paper-cmu-drive-v2v-vla.md) / [WAM-Diff2](./paper-wam-diff2.md) | 同批三条智驾 VLA 的不同取舍：本文做**减法**（剪层），CMU-Drive 做**加法**（多出语言推理与通信头），WAM-Diff2 改**解码范式**（AR→离散扩散并行解码）。选型先确认瓶颈是时延、协同信息还是暴露偏差 |
| [实时性 ↔ 泛化取舍](../concepts/embodied-fm-latency-generalization-tradeoff.md) | 该页给「规模 / 模态跨度换时延」这条边界；本文的读法是——若浅层已含规划信息，深层那部分算力就不落在这条取舍线上，剪掉不必然掉泛化。但「未见明确任务退化」只在 Bench2Drive 口径下成立，换域须重测 |

## 结论

**Depth-Wise Probing Driving VLA 适合作为本期「性能提升」路线的快速索引页。**

1. 核心贡献：智驾 VLA 规划 token 逐层探针：导航意图与规划信息在浅层已出现，深层层主要负责对齐规划器表示；学习式早读与剪层可提速且未见明确任务退化（DriveX@ECCV 2026）。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part1.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md)
- [arXiv:2608.07361](https://arxiv.org/abs/2608.07361)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.07361)
