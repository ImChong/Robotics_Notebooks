---
type: entity
project_id: rangenav
tags:
  - paper
  - navigation
  - planning
status: complete
updated: 2026-10-29
arxiv: "2610.29845"
project: https://rangenav.github.io/
related:
  - ../tasks/locomotion.md
  - ../tasks/vision-language-navigation.md
  - ../concepts/sim2real.md
  - ../overview/embodied-station-2026-10-29-58-papers-technology-map.md
sources:
  - ../../sources/papers/rangenav_arxiv_2610_29845.md
  - ../../sources/sites/rangenav.md
  - ../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md
summary: "RangeNav（arXiv:2610.29845）：RangeNav在合成距离图上联合学习重建、语义和俯视预测，再冻结编码器训练导航策略；仿真及四足真机展示环视感知价值，比较需对应传感器视场条件。"
---

# RangeNav（arXiv:2610.29845）

**RangeNav**（*RangeNav: Task-Aligned LiDAR Range-Image Pretraining for Robot Navigation*，[arXiv:2610.29845](https://arxiv.org/abs/2610.29845)，[项目页](https://rangenav.github.io/)）来自 [具身智能小站 2026-10-29 盘点](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)，属「导航、规划与定位」组。

## 一句话定义

**RangeNav在合成距离图上联合学习重建、语义和俯视预测，再冻结编码器训练导航策略；仿真及四足真机展示环视感知价值，比较需对应传感器视场条件。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RangeNav | RangeNav: Task-Aligned LiDAR Range-Image Pretraining for Robot Navigation | 本文方法 / 基准的简称 |

## 为什么重要

- 用轻量激光雷达表征应对拥挤导航。
- 开源结论：**待发布**（步骤 2.5，2026-10-29）。项目页 HTTP 200，Code & models 标注 coming soon（脚本扫描）。
- 同期同组工作见 [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)。

## 流程总览

```mermaid
flowchart TD
    A[TartanGround 六向稠密深度图<br/>融合到共同传感器坐标系] --> B[映射到目标 LiDAR 球面距离图<br/>方位/俯仰分箱, 冲突取最小值]
    B --> C[增强与加噪<br/>随机环形滚动, 扫描环偏置, 高斯噪声, 丢点, 连续遮挡<br/>log 变换, 干净距离图作重建目标]
    C --> E[Encoder<br/>circular 卷积 + 各向异性卷积 + depthwise-separable 残差块<br/>输出空间 latent z_L]
    E --> D1[距离重建头<br/>Huber 重建 + 梯度损失]
    E --> D2[语义分割头<br/>8 类, CE + Lovász]
    E --> D3[BEV 头<br/>占据 + 地面高度]
    D1 --> L[联合预训练损失 L_pre<br/>训练后丢弃所有解码头, 仅保留 Encoder]
    D2 --> L
    D3 --> L
    L --> F[冻结 Encoder]
    F --> P[注意力融合带噪本体感觉与相对目标<br/>递归 SRU 策略]
    P --> Act[Actor 输出动作<br/>非对称 Critic 可用特权几何观测]
    Act --> R[MDPO 在 Isaac Lab 并行环境训练<br/>随机化地形 / 动力学 / 传感]
```

图依据：论文 §III-A–D / Fig.2 / Fig.3

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.29845](https://arxiv.org/abs/2610.29845) |
| **开源** | **待发布** |
| **要点** | RangeNav在合成距离图上联合学习重建、语义和俯视预测，再冻结编码器训练导航策略；仿真及四足真机展示环视感知价值，比较需对应传感器视场条件。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

**不适用（待发布）** — 项目页 HTTP 200，Code & models 标注 coming soon（脚本扫描）。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [多样成功示范、触觉工具链与预见式感知：具身智能小站 10-29 期 58 篇阅读坐标](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**用轻量激光雷达表征应对拥挤导航：RangeNav在合成距离图上联合学习重建、语义和俯视预测，再冻结编码器训练导航策略；仿真及四足真机展示环视感知价值，比较需对应传感器视场条件。**

1. 开源边界：**待发布** — 项目页 HTTP 200，Code & models 标注 coming soon（脚本扫描）。
2. 核心机制：RangeNav在合成距离图上联合学习重建、语义和俯视预测，再冻结编码器训练导航策略；仿真及四足真机展示环视感知价值，比较需对应传感器视场条件。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [vision-language-navigation](../tasks/vision-language-navigation.md)
- [sim2real](../concepts/sim2real.md)
- [embodied-station-2026-10-29-58-papers-technology-map](../overview/embodied-station-2026-10-29-58-papers-technology-map.md)

## 参考来源

- [rangenav_arxiv_2610_29845.md](../../sources/papers/rangenav_arxiv_2610_29845.md)
- [wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md](../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)
- [arXiv:2610.29845](https://arxiv.org/abs/2610.29845)
- [arXiv PDF](https://arxiv.org/pdf/2610.29845)：已通读方法部分（实验与附录以略读为准）：摘要、§I 引言、§II 相关工作（部分）、§III 方法（A–D）、§IV-A/B 实验设定

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.29845)
- [项目页](https://rangenav.github.io/)
