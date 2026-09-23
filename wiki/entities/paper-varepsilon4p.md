---
type: entity
tags:
  - paper
  - vla
  - flow-matching
  - manipulation
  - data-efficiency
status: complete
updated: 2026-09-23
arxiv: "2609.26672"
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../tasks/manipulation.md
  - ./paper-industrialvla-bench.md
  - ../overview/collab-wm-12-papers-technology-map.md
sources:
  - ../../sources/papers/varepsilon4p_arxiv_2609_26672.md
  - ../../sources/sites/varepsilon4p.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_collab_wm_2026-09-23.md
summary: "ε4P（arXiv:2609.26672）：按 flow-matching 噪声阶段分工：低精度目标任务数据在高噪声保留语境，高精度异任务数据在低噪声传递动作精度。"
---

# ε4P（arXiv:2609.26672）

**ε4P**（*Imperfection for Precision: Upcycling Imperfect Data for High-Precision Robotic Manipulation*，[arXiv:2609.26672](https://arxiv.org/abs/2609.26672)，[项目页](https://varepsilon4p.github.io/)）来自 [具身智能小站 12 篇盘点](../../sources/blogs/wechat_embodied_station_12_papers_collab_wm_2026-09-23.md)。

## 一句话定义

**按 flow-matching 噪声阶段分工：低精度目标任务数据在高噪声保留语境，高精度异任务数据在低噪声传递动作精度。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ε4P | Imperfection for Precision | 本文 imperfect 数据升级框架 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| FM | Flow Matching | 连续流匹配生成/策略训练 |
| UMI | Universal Manipulation Interface | 常见异任务高精度示范来源 |

## 为什么重要

- 亚毫米级精密操作常被昂贵任务专属遥操作数据卡住；ε4P 不粗暴混合 imperfect 源，而用 flow time 作 admission filter。
- 开源结论：**未开源**（步骤 2.5，2026-09-23）。
- 与 [12 篇技术地图](../overview/collab-wm-12-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.26672](https://arxiv.org/abs/2609.26672) |
| **开源** | **未开源** |
| **要点** | 离线估计 t_pm / t_tm 边界；训练时按源采样可接受 flow time；策略架构与目标不变。 |
| **文内指标** | ATX 插入 80.0%、两阶段线缆 88.3%、螺栓分拣 91.7%；相对 native co-training 最高 +31.7 pp。 |

## 源码运行时序图

**不适用**（入库日模型/训练权重未公开，或仅有 API/CLI 封装；无可运行官方训练/推理入口。）

## 实验与评测

- ATX 插入 80.0%、两阶段线缆 88.3%、螺栓分拣 91.7%；相对 native co-training 最高 +31.7 pp。
- **读法：** 索引级摘要；逐项对照与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [12 篇技术地图](../overview/collab-wm-12-papers-technology-map.md)；与同 arXiv 节点不重复造页。

## 结论

**ε4P 证明 imperfect 数据的价值在「何时贡献」而非「是否保留」；项目页入库日无 GitHub。**

1. 开源边界：**未开源** — 以项目页实际链接为准（入库日 2026-09-23）。
2. 核心机制：离线估计 t_pm / t_tm 边界；训练时按源采样可接受 flow time；策略架构与目标不变。…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [manipulation](../tasks/manipulation.md)
- [paper-industrialvla-bench](./paper-industrialvla-bench.md)

## 参考来源

- [varepsilon4p_arxiv_2609_26672.md](../../sources/papers/varepsilon4p_arxiv_2609_26672.md)
- [wechat_embodied_station_12_papers_collab_wm_2026-09-23.md](../../sources/blogs/wechat_embodied_station_12_papers_collab_wm_2026-09-23.md)
- [arXiv:2609.26672](https://arxiv.org/abs/2609.26672)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.26672)
- [项目页](https://varepsilon4p.github.io/)

