---
type: entity
tags: ['paper', 'grasp', 'cross-embodiment', 'manipulation', 'ustc', 'szu']
status: complete
updated: 2026-09-17
arxiv: "2609.18504"
related:
  - ../tasks/manipulation.md
  - ../queries/cross-embodiment-transfer-strategy.md
  - ./paper-recmorph.md
  - ../overview/perception-action-transfer-9-papers-technology-map.md
sources:
  - ../../sources/papers/intermash_arxiv_2609_18504.md
  - ../../sources/sites/intermash.md
  - ../../sources/blogs/wechat_embodied_station_9_papers_perception_action_transfer_2026-09-17.md
summary: "InterMASH（arXiv:2609.18504）：球谐 anchor 统一人手/机械手局部几何；条件 DiT 联合生成手形与接触；Barrett 混合训练 90.30%；代码待发布。"
---

# InterMASH（arXiv:2609.18504）

**InterMASH**（*A Unified Geometric Representation for Grasp Synthesis*，[arXiv:2609.18504](https://arxiv.org/abs/2609.18504)，[项目页](https://inter-mash.github.io/)）来自 [具身智能小站 9 篇盘点](../../sources/blogs/wechat_embodied_station_9_papers_perception_action_transfer_2026-09-17.md)。

## 一句话定义

**用球固定 anchor + 低阶球谐把局部手–物–接触编码成跨 embodiment 共享 token，再经条件 Diffusion Transformer 联合生成手形与接触图。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DiT | Diffusion Transformer | 扩散 Transformer 生成骨干 |
| IK | Inverse Kinematics | 由 patch 几何恢复关节姿态 |
| MANO | hand Model with Articulated and Non-rigid defOrmations | 人手参数化模型 |

## 为什么重要

- 人手与机械手形态差异大，contact map  alone 难建立可迁移对应。
- **混合手型训练** 可提升 ShadowHand / Barrett 成功率（Barrett 90.30% on CMapDataset）。
- 开源结论：**待发布**（2026-09-17）。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.18504](https://arxiv.org/abs/2609.18504) |
| **开源** | **待发布** |
| **流程** | Align & encode → jointly denoise → patch-wise IK |
| **文内指标** | DexGraspNet Suc.1 91.9%、Pen. 16.2mm；人类 GRAB 微调提升 robotic 成功率 |

## 源码运行时序图

**不适用**（截至 2026-09-17 无官方代码仓库）。

## 实验与评测

| 设定 | 文内指标 | 读法 |
|------|----------|------|
| DexGraspNet | Suc.1 **91.9%**、Pen. **16.2 mm** | 成功率与穿透深度需成对读：穿透改善不等于六方向扰动全面领先 |
| CMapDataset · Barrett | **90.30%**（混合手型训练） | 主证据链——混合训练是 robotic 手成功率的主要杠杆 |
| 人手 → 机械手迁移 | GRAB 人手数据微调后 robotic 成功率上升 | 支撑「统一几何 token 可跨 embodiment 共享」的核心主张 |
| 质量–多样性 | 论文自述仍存在权衡 | 提升成功率的设置可能压缩生成多样性，选型时需按下游用途取舍 |

本页为清单摘要级；完整表格、扰动协议与消融以原文 PDF 为准（[参考来源](#参考来源)）。代码 **待发布**，暂不可独立复现。

## 与其他工作对比

| 对照对象 | 差异 |
|----------|------|
| 纯 contact map 条件生成 | contact map 只描述「哪里接触」，跨手形态差异大时难建立对应；InterMASH 用 **球固定 anchor + 低阶球谐** 编码局部手–物–接触几何 |
| 每种手单独训练的专用模型 | 数据不共享、换手即重训；本文用共享 token 支持 **混合手型训练**，且人手数据可反哺机械手 |
| DexGraspNet 系 baseline | 同数据集同指标对照（Suc.1 / Pen.）；InterMASH 的增量在跨 embodiment 共享而非单手精度极限 |
| [RecMorph](./paper-recmorph.md) | 同属跨 embodiment，但作用域不同：RecMorph 做 **locomotion 形态控制**，InterMASH 做 **抓取几何合成** |

## 结论

**InterMASH 把跨手抓取合成问题收敛到统一几何 token——质量–多样性权衡仍存在，混合训练是提升 robotic 成功率的关键杠杆。**

1. 与 [RecMorph](./paper-recmorph.md) 同属跨 embodiment，但 InterMASH 聚焦 **抓取几何** 而非 locomotion 控制。
2. 渗透深度指标改善不等于六方向扰动成功率全面领先。
3. 待代码发布后可与 DexGrasp 系列 baseline 复现对照。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [cross-embodiment-transfer-strategy](../queries/cross-embodiment-transfer-strategy.md)
- [RecMorph](./paper-recmorph.md)
- [9 篇技术地图](../overview/perception-action-transfer-9-papers-technology-map.md)

## 参考来源

- [intermash_arxiv_2609_18504.md](../../sources/papers/intermash_arxiv_2609_18504.md)
- [wechat_embodied_station_9_papers_perception_action_transfer_2026-09-17.md](../../sources/blogs/wechat_embodied_station_9_papers_perception_action_transfer_2026-09-17.md)

## 推荐继续阅读

- [InterMASH 项目页](https://inter-mash.github.io/)
- [arXiv PDF](https://arxiv.org/pdf/2609.18504)
