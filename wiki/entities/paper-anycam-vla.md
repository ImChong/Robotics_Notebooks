---
type: entity
tags:
  - paper
  - vla
  - camera-robustness
  - novel-view
status: complete
updated: 2026-09-19
arxiv: "2603.05868"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md
sources:
  - ../../sources/papers/anycam_vla_arxiv_2603_05868.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md
summary: "零样本相机适配 VLA：测试时用前馈新视角合成将画面转为接近训练相机配置，无需增数据或改结构（IROS 2026）。"
---

# AnyCamVLA（arXiv:2603.05868）

**AnyCamVLA**（[arXiv:2603.05868](https://arxiv.org/abs/2603.05868)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第一篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) **空间感知** 段。

## 一句话定义

**零样本相机适配 VLA：测试时用前馈新视角合成将画面转为接近训练相机配置，无需增数据或改结构（IROS 2026）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| VLN | Vision-Language Navigation | 视觉–语言导航 |
| WAM | World Action Model | 联合未来与动作生成的具身策略 |
| TTT | Test-Time Training | 部署阶段无标注数据的在线适配 |

## 为什么重要

- 零样本相机适配 VLA：测试时用前馈新视角合成将画面转为接近训练相机配置，无需增数据或改结构（IROS 2026）。
- 开源状态：**待核实**（步骤 2.5，入库日 2026-09-19）。
- 与 [vla weekly trends 2026 08 10 part1 technology map](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md) 同批工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2603.05868](https://arxiv.org/abs/2603.05868) |
| **项目页** | https://heo0224.github.io/AnyCamVLA |
| **开源** | **待核实** |
| **文内评测** | LIBERO、LIBERO-Plus；实机 Franka Panda |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚未公开）。


## 实验与评测

- **文内口径：** LIBERO、LIBERO-Plus；实机 Franka Panda
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

> 下表只做**定位对照**：本页与下列同批各页均为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md) 的索引级摘要，未逐条核对原文实验表，彼此**不共享同一评测协议**，不可据此横比数字。

| 对照 | 差异读法 |
|------|----------|
| [Cross-View Action Consistency](./paper-cross-view-action-consistency-vla.md) | 同批同题（相机位姿一变策略就掉点），**介入时机相反**：本文在**测试期**把观测合成回训练相机配置，那条在**训练期**用跨视角流匹配一致性把策略练成视角不敏感。本文免训、可套在已有策略上，代价是推理多一次视角合成；那条要重训，但部署零额外开销 |
| [WNM-3D](./paper-wnm-3d-vln.md) | 同批「空间感知」段的另一条，抽象轴不同：本文修的是**图像层**的视角错配，WNM-3D 换的是**表征层**（3D 场景条件替代 2D 条件）。图像层改法部署更轻，表征层改法从根上不依赖某个相机摆位 |
| [VLA Depth Decodability](./paper-vla-action-post-training-depth-decodability.md) | 本文是在**补**几何一致性，那条诊断的是几何信息**为什么会缺**（动作后训练削弱深度可解码性）。合读的判据是：该在输入端补，还是该护住骨干里已有的几何表征 |
| [Sim2Real](../concepts/sim2real.md) | 相机内外参错配是「换场地就掉点」的常见落点；本文把它当**渲染域的可补偿偏移**处理，与视角随机化一族的取舍是「推理多一次合成」vs「训练多一批数据」 |

## 结论

**AnyCamVLA 适合作为本期「空间感知」路线的快速索引页。**

1. 核心贡献：零样本相机适配 VLA：测试时用前馈新视角合成将画面转为接近训练相机配置，无需增数据或改结构（IROS 2026）。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图](../overview/vla-weekly-trends-2026-08-10-part1-technology-map.md)
- [Generative World Models](../methods/generative-world-models.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part1.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part1.md)
- [arXiv:2603.05868](https://arxiv.org/abs/2603.05868)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2603.05868)
