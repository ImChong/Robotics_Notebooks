---
type: entity
tags: [paper, curated-index, awesome-world-action-models-rcl, rcl-wam-catalog]
status: complete
updated: 2026-09-25
arxiv: "2609.03565"
venue: "2026"
summary: "SA+IDM trains an action-conditioned JEPA world model with two auxiliary heads: inverse dynamics recovers executed actions, while state alignment predicts measured physical state from consecutive image representations. De"
related:
  - ../entities/awesome-world-action-models-rcl.md
  - ../overview/rcl-awesome-wam-technology-map.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/papers/rcl_awesome_wam_2609_03565_toward-physically-grounded-jepa-world-mo.md
  - ../../sources/papers/rcl_awesome_wam_catalog.md
  - ../../sources/repos/awesome-world-action-models-rcl.md
---

# Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Pla...

**Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning** 收录于 [Awesome World-Action Models (RCL)](https://github.com/rcl-robotics/Awesome-World-Action-Models) **第 495/564** 篇，分组 **WAMs**。本页为知识库 **策展索引级** 详情节点；方法细节与量化指标以原文 PDF / 项目页为准。

## 一句话定义

SA+IDM trains an action-conditioned JEPA world model with two auxiliary heads: inverse dynamics recovers executed actions, while state alignment predicts measured physical state from consecutive image representations. Deployment uses only the encoder and latent predictor inside CEM planning. State alignment improves...

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界预测与动作生成耦合 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| IDM | Inverse Dynamics Model | 先预测未来再反推动作 |
| WM | World Model | 环境前向预测模型 |

## 为什么重要

- SA+IDM trains an action-conditioned JEPA world model with two auxiliary heads: inverse dynamics recovers executed actions, while state alignment predicts measured physical state from consecutive image representations. Deployment uses only the encoder and latent predictor inside CEM planning. State alignment improves...
- 在 [RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md) 中提供可点击的独立详情节点，避免清单条目无法落入知识图谱。
- 与列表实体 [Awesome World-Action Models](../entities/awesome-world-action-models-rcl.md) 及站内 WAM / VLA 方法页交叉，便于从策展索引跳转到学习主线。

## 核心信息（索引级）

| 字段 | 内容 |
|------|------|
| 编号 | 495/564 |
| 分组 | WAMs |
| 出处 | 2026 |
| 论文 | <https://arxiv.org/abs/2609.03565> |
| 子类 / 象限 | 潜空间预测与JEPA · 四象限外 |

## 核心机制（归纳）

### 策展导读要点

SA+IDM trains an action-conditioned JEPA world model with two auxiliary heads: inverse dynamics recovers executed actions, while state alignment predicts measured physical state from consecutive image representations. Deployment uses only the encoder and latent predictor inside CEM planning. State alignment improves...

本页不复述论文公式与完整实验表；若需工程落地，请回到原文并对照站内 [World Action Models（WAM）](../concepts/world-action-models.md) 等概念页。

## 评测与指标（索引级）

- 本条目为 RCL Awesome **索引级** 摘录，**未搬运** 原文量化 benchmark 与实机指标。
- 评测口径与具体数值以 [原文 / 项目页](https://arxiv.org/abs/2609.03565) 为准。
- 横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 同分组条目。

## 与其他工作对比（索引级）

- 本页 **不做** 与具体基线的逐项数值对比：索引级节点只保留清单坐标，同分组横向对照请回到 [技术地图](../overview/rcl-awesome-wam-technology-map.md) 的 **WAMs** 分组逐条展开。
- 与站内 **深度论文实体** 的分界：深度页承载机构、实验表与源码运行时序；本页只承载清单 Contribution 阅读锚点。同一 arXiv 若已存在深度页，应以深度页为准。
- 与清单内相邻条目孰优孰劣，本页不下结论：清单 Contribution 可能滞后于论文最新版本，差异应以各自原文的问题设定与评测口径为准。

## 结论

**本条目的站内价值是把「Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Pla...」从 RCL Awesome WAM 列表提升为可链接的知识节点，并保留清单 Contribution 作为阅读锚点。**

- 起作用的是策展坐标：列表分组 **WAMs** + Contribution 指出的问题设定，而不是本页自行推导的新算法结论。
- 适用边界：索引级页面不能替代 PDF；开源状态以项目页实际链接为准（清单可能滞后）。
- 若该工作成为学习主线，应再升格为深度论文实体（补机构、实验表、源码运行时序图或「不适用」说明）。

## 常见误区

1. 不要把 Awesome 条目的 Contribution 当成完整方法证明——它只是策展导读。
2. 同一 arXiv 在全库只允许一个 canonical 详情节点；若已有深度页，应以深度页为准。

## 关联页面

- 列表实体：[Awesome World-Action Models（RCL）](../entities/awesome-world-action-models-rcl.md)
- 技术地图：[RCL Awesome WAM 技术地图](../overview/rcl-awesome-wam-technology-map.md)
- 方法/任务：[generative-world-models.md](../methods/generative-world-models.md)、[manipulation.md](../tasks/manipulation.md)

## 参考来源

- [`sources/papers/rcl_awesome_wam_2609_03565_toward-physically-grounded-jepa-world-mo.md`](../../sources/papers/rcl_awesome_wam_2609_03565_toward-physically-grounded-jepa-world-mo.md) — 本条目策展摘录
- [`sources/papers/rcl_awesome_wam_catalog.md`](../../sources/papers/rcl_awesome_wam_catalog.md) — 列表总表
- [`sources/repos/awesome-world-action-models-rcl.md`](../../sources/repos/awesome-world-action-models-rcl.md)
- [`docs/PAPERS.md`](https://github.com/RCL-Robotics/Awesome-World-Action-Models/blob/main/docs/PAPERS.md) — 上游论文目录
- 论文：<https://arxiv.org/abs/2609.03565>

## 推荐继续阅读

- [Awesome World-Action Models (RCL) 仓库](https://github.com/rcl-robotics/Awesome-World-Action-Models)
- [原文](https://arxiv.org/abs/2609.03565)
