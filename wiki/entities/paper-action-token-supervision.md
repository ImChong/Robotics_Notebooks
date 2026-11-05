---
type: entity
project_id: action-token-supervision
tags:
  - paper
  - benchmark
  - evaluation
status: complete
updated: 2026-11-05
arxiv: "2611.05249"
project: https://mot-vla-gradient-routing.github.io/
related:
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
  - ../overview/hub-embodied-eval-benchmark.md
sources:
  - ../../sources/papers/action_token_supervision_arxiv_2611_05249.md
  - ../../sources/sites/action-token-supervision.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "Action-Token Supervision（arXiv:2611.05249）：作者分别开关动作标记监督和梯度回传，再比较驾驶与操作策略。"
---

# Action-Token Supervision（arXiv:2611.05249）

**Action-Token Supervision**（*Should the Action Expert Train the VLM? Action-Token Supervision and Gradient Routing in Mixture-of-Transformer VLAs*，[arXiv:2611.05249](https://arxiv.org/abs/2611.05249)，[项目页](https://mot-vla-gradient-routing.github.io/)）来自 [具身智能小站 2026-11-05 盘点](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)，属「策略泛化与评测」组。

## 一句话定义

**作者分别开关动作标记监督和梯度回传，再比较驾驶与操作策略。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Action-Token Supervision | Should the Action Expert Train the VLM? Action-Token Supervision and Gradient Routing in Mixture-of-Transformer VLAs | 本文方法 / 基准的简称 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉语言模型 |

## 为什么重要

- 动作预测的误差该传回视觉语言模型吗。一个值得复查的差别是，开环预测里两种训练方式接近时，闭环驾驶中车辆沿路线走出的距离仍可能不同。
- 开源结论：**项目页未列 GitHub**（步骤 2.5，2026-11-05）。项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
- 同期同组工作见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.05249](https://arxiv.org/abs/2611.05249) |
| **开源** | **项目页未列 GitHub** |
| **要点** | 作者分别开关动作标记监督和梯度回传，再比较驾驶与操作策略。 |
| **文内指标** | 公众号转述：一个值得复查的差别是，开环预测里两种训练方式接近时，闭环驾驶中车辆沿路线走出的距离仍可能不同。 |

## 源码运行时序图

**不适用（项目页未列 GitHub）** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号转述：一个值得复查的差别是，开环预测里两种训练方式接近时，闭环驾驶中车辆沿路线走出的距离仍可能不同。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**动作预测的误差该传回视觉语言模型吗：一个值得复查的差别是，开环预测里两种训练方式接近时，闭环驾驶中车辆沿路线走出的距离仍可能不同。**

1. 开源边界：**项目页未列 GitHub** — 项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。
2. 核心机制：作者分别开关动作标记监督和梯度回传，再比较驾驶与操作策略。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [simulation-evaluation-infrastructure](../concepts/simulation-evaluation-infrastructure.md)
- [vla](../methods/vla.md)
- [imitation-learning](../methods/imitation-learning.md)
- [embodied-station-2026-11-05-43-papers-technology-map](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)
- [具身评测基准选型闭环（知识链）](../overview/hub-embodied-eval-benchmark.md)

## 参考来源

- [action_token_supervision_arxiv_2611_05249.md](../../sources/papers/action_token_supervision_arxiv_2611_05249.md)
- [wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- [arXiv:2611.05249](https://arxiv.org/abs/2611.05249)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.05249)
- [项目页](https://mot-vla-gradient-routing.github.io/)
