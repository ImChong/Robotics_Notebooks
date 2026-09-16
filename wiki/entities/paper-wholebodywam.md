---
type: entity
tags: ['paper', 'humanoid', 'wam', 'loco-manipulation', 'wbc']
status: complete
updated: 2026-09-16
arxiv: "2609.16644"
related:
  - ../concepts/world-action-models.md
  - ../concepts/whole-body-control.md
  - ../tasks/loco-manipulation.md
  - ./paper-dido-wam.md
  - ../overview/vla-deploy-12-papers-technology-map.md
sources:
  - ../../sources/papers/wholebodywam_arxiv_2609_16644.md
  - ../../sources/sites/wholebodywam.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md
summary: "WholeBodyWAM（arXiv:2609.16644）：保留预训练世界—动作先验，用 WBC 语义接地协调模块扩展到人形全身 loco-manipulation。"
---

# WholeBodyWAM（arXiv:2609.16644）

**WholeBodyWAM**（*WholeBodyWAM: Generalizing Pre-trained World-Action Priors to Humanoid Loco-Manipulation via WBC-Grounded Coordination*，[arXiv:2609.16644](https://arxiv.org/abs/2609.16644)，[项目页](https://wholebodywam.github.io/)）来自 [具身智能小站 12 篇盘点](../../sources/blogs/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md)。

## 一句话定义

**保留预训练世界—动作先验，用 WBC 语义接地协调模块扩展到人形全身 loco-manipulation。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界—动作联合模型 |
| WBC | Whole-Body Control | 全身控制 |
| Loco-Manip | Loco-Manipulation | 移动操作联合任务 |
| Prior | Pre-trained Prior | 预训练世界—动作先验 |

## 为什么重要

- 桌面 WAM 多；人形需同时协调行走、全身控制与手部动作，不宜从零重学全身行为。
- 开源结论：**待发布**（步骤 2.5，2026-09-16）。
- 与 [12 篇技术地图](../overview/vla-deploy-12-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.16644](https://arxiv.org/abs/2609.16644) |
| **开源** | **待发布** |
| **要点** | 预训练 WAM 先验 + 异构 WBC 语义协调模块；项目页入库日无 GitHub。 |
| **文内指标** | 项目页强调可扩展全身智能；具体数值以原文为准。 |


## 源码运行时序图

**不适用（待发布）** — 截至 2026-09-16 项目页未列可运行官方仓库。


## 实验与评测

- 项目页强调可扩展全身智能；具体数值以原文为准。
- **读法：** 索引级摘要；逐项对照与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [12 篇技术地图](../overview/vla-deploy-12-papers-technology-map.md)；与同 arXiv 节点不重复造页。

## 结论

**WholeBodyWAM 代表「先验复用 + WBC 接地」路线，工程复现需等官方代码。**

1. 开源边界：**待发布** — 以项目页实际链接为准（入库日 2026-09-16）。
2. 核心机制：预训练 WAM 先验 + 异构 WBC 语义协调模块；项目页入库日无 GitHub。…
3. 部署前核对任务协议与硬件条件，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [whole-body-control](../concepts/whole-body-control.md)
- [loco-manipulation](../tasks/loco-manipulation.md)
- [paper-dido-wam](./paper-dido-wam.md)

## 参考来源

- [wholebodywam_arxiv_2609_16644.md](../../sources/papers/wholebodywam_arxiv_2609_16644.md)
- [wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md](../../sources/blogs/wechat_embodied_station_12_papers_vla_deploy_2026-09-16.md)
- [arXiv:2609.16644](https://arxiv.org/abs/2609.16644)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.16644)
- [项目页](https://wholebodywam.github.io/)

