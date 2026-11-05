---
type: entity
project_id: future-representation-distillation
tags:
  - paper
  - world-model
  - control
status: complete
updated: 2026-11-05
arxiv: "2611.04527"
project: https://harryjo97.github.io/dreamfree/
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
sources:
  - ../../sources/papers/future_representation_distillation_arxiv_2611_04527.md
  - ../../sources/sites/future-representation-distillation.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "Future Representation Distillation（arXiv:2611.04527）：DreamFree 训练视觉编码器从当前画面预测未来特征，部署时直接据此生成动作。"
---

# Future Representation Distillation（arXiv:2611.04527）

**Future Representation Distillation**（*Future Representation Distillation for Imagination-Free World Action Models*，[arXiv:2611.04527](https://arxiv.org/abs/2611.04527)，[项目页](https://harryjo97.github.io/dreamfree/)）来自 [具身智能小站 2026-11-05 盘点](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)，属「世界模型与控制」组。

## 一句话定义

**DreamFree 训练视觉编码器从当前画面预测未来特征，部署时直接据此生成动作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Future Representation Distillation | Future Representation Distillation for Imagination-Free World Action Models | 本文方法 / 基准的简称 |

## 为什么重要

- 省下生成未来视频的时间，让机器人更快出手。在作者的测试平台上，140 亿参数的视频骨干最高可做 11 赫兹闭环控制。
- 开源结论：**待发布**（步骤 2.5，2026-11-05）。项目页 HTTP 200，Code 入口标 soon；页面链接的 harryjo97/dreamfree 经 git ls-remote 可达，是否已含论文代码以仓库 README 为准。
- 同期同组工作见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.04527](https://arxiv.org/abs/2611.04527) |
| **开源** | **待发布** |
| **要点** | DreamFree 训练视觉编码器从当前画面预测未来特征，部署时直接据此生成动作。 |
| **文内指标** | 公众号转述：在作者的测试平台上，140 亿参数的视频骨干最高可做 11 赫兹闭环控制。 |

## 源码运行时序图

**不适用（待发布）** — 项目页 HTTP 200，Code 入口标 soon；页面链接的 harryjo97/dreamfree 经 git ls-remote 可达，是否已含论文代码以仓库 README 为准。 不编造运行脚本；代码公开后再补。

## 实验与评测

- 公众号转述：在作者的测试平台上，140 亿参数的视频骨干最高可做 11 赫兹闭环控制。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [接触丰富的手物重定向、分项泛化评测与手持夹爪示范：具身智能小站 11-05 期 43 篇阅读坐标](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**省下生成未来视频的时间，让机器人更快出手：在作者的测试平台上，140 亿参数的视频骨干最高可做 11 赫兹闭环控制。**

1. 开源边界：**待发布** — 项目页 HTTP 200，Code 入口标 soon；页面链接的 harryjo97/dreamfree 经 git ls-remote 可达，是否已含论文代码以仓库 README 为准。
2. 核心机制：DreamFree 训练视觉编码器从当前画面预测未来特征，部署时直接据此生成动作。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-05-43-papers-technology-map](../overview/embodied-station-2026-11-05-43-papers-technology-map.md)

## 参考来源

- [future_representation_distillation_arxiv_2611_04527.md](../../sources/papers/future_representation_distillation_arxiv_2611_04527.md)
- [wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md](../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- [arXiv:2611.04527](https://arxiv.org/abs/2611.04527)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.04527)
- [项目页](https://harryjo97.github.io/dreamfree/)
