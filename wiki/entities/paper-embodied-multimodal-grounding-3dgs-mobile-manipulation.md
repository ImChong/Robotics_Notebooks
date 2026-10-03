---
type: entity
tags:
  - paper
  - vla
  - 3dgs
  - mobile-manipulation
  - navigation
status: complete
updated: 2026-10-03
arxiv: "2608.10756"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md
sources:
  - ../../sources/papers/embodied-multimodal-grounding-3dgs-mobile-manipulation_arxiv_2608_10756.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md
summary: "Embodied MM Grounding（3DGS）（arXiv:2608.10756）：多角度更新带语义的三维高斯地图，用于开放词汇定位、避障与站位选择，再交给动作模型，缓解遮挡与视角变化。"
---

# Embodied MM Grounding（3DGS）（arXiv:2608.10756）

**Embodied MM Grounding（3DGS）**（*Embodied Multimodal Grounding for Open-Vocabulary Mobile Manipulation via Semantic 3D Gaussian Splatting*，[arXiv:2608.10756](https://arxiv.org/abs/2608.10756)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第二篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) **空间感知/类 Agent** 段。

## 一句话定义

**多角度更新带语义的三维高斯地图，用于开放词汇定位、避障与站位选择，再交给动作模型，缓解遮挡与视角变化。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| CoT | Chain-of-Thought | 链式推理 |

## 为什么重要

- 移动操作需语言–视觉–3D–可行性统一；单视角 VLA 易定位错误。
- 策展机构：香港科技大学（广州）；美的集团；香港科技大学
- 开源结论：**待核实**（步骤 2.5，2026-10-01）。

## 流程总览

以下按本页已归纳的机制与资料绘制，表示模块或阅读路径关系。

```mermaid
flowchart TD
    N0["多视角观测"]
    N1["语义三维高斯地图"]
    N2["开放词汇定位"]
    N3["避障与站位选择"]
    N4["动作模型"]
    N5["移动操作执行"]
    N6["新视角"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N1 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N1
```

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.10756](https://arxiv.org/abs/2608.10756) |
| **开源** | **待核实** |
| **文内评测** | Alicia-D 臂 + Unitree Go2 Edu 四足实机 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- **文内口径：** Alicia-D 臂 + Unitree Go2 Edu 四足实机
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md) | 同批 15 篇横向索引；本文属 **空间感知/类 Agent** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**Embodied MM Grounding（3DGS） 适合作为本期「空间感知/类 Agent」路线的快速索引页。**

1. 核心贡献：多角度更新带语义的三维高斯地图，用于开放词汇定位、避障与站位选择，再交给动作模型，缓解遮挡与视角变化。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第二篇）](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part2.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md)
- [arXiv:2608.10756](https://arxiv.org/abs/2608.10756)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.10756)
