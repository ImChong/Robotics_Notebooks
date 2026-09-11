---
type: entity
tags: [paper, benchmark, humanoid, mllm]
status: complete
updated: 2026-09-11
arxiv: "2609.10895"
code: https://huggingface.co/datasets/Alan123/reacthuman-benchmark-scaled
related:
  - ../methods/vla.md
  - ../methods/generative-world-models.md
  - ../tasks/manipulation.md
  - ../overview/dexterous-wm-humanoid-14-papers-technology-map.md
sources:
  - ../../sources/papers/reacthuman_arxiv_2609_10895.md
  - ../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md
summary: "17 类事件、1000+ 可复现情景；240 Hz 刚体仿真生成真值；五项诊断指标。"
---

# ReactHuman（arXiv:2609.10895）

**ReactHuman**（[ReactHuman: A Physics-Grounded Benchmark for Human-Like Reactive Decision-Making in Embodied Multimodal LLMs](https://arxiv.org/abs/2609.10895)）来自 [具身智能小站 14 篇盘点](../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md)。17 类事件、1000+ 可复现情景；240 Hz 刚体仿真生成真值；五项诊断指标。

## 一句话定义

**真正危险的反应要把每个动作执行出来测——突发家庭危险下的物理接地 MLLM 评测。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| VLM | Vision-Language Model | 视觉-语言多模态模型 |
| WM | World Model | 预测未来观测或表征的动力学模型 |
| IL | Imitation Learning | 模仿学习 |
| RL | Reinforcement Learning | 强化学习 |
| DoF | Degrees of Freedom | 自由度 |

## 为什么重要

- 纳入本期 **灵巧手 / 世界模型 / 人形控制 / VLA** 主线之一。
- 开源状态：**部分开源**（步骤 2.5 核查，2026-09-11）。
- 与 [14 篇技术地图](../overview/dexterous-wm-humanoid-14-papers-technology-map.md) 中同类工作可横向对照。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.10895](https://arxiv.org/abs/2609.10895) |
| **项目页** | https://arxiv.org/abs/2609.10895 |
| **代码/资源** | https://huggingface.co/datasets/Alan123/reacthuman-benchmark-scaled |
| **开源** | **部分开源** |
| **文内指标** | 17 类事件、1000+ 情景；五项诊断指标。 |


## 源码运行时序图

**不适用**（Hugging Face 数据集已发布；仿真/评测代码链以 arXiv 与数据集页为准。）

## 实验与评测

| 项 | 文内口径 |
|----|----------|
| 要点 | 17 类事件、1000+ 情景；五项诊断指标。 |

- **读法：** 本页为索引级摘要，上表取自 [公众号盘点](../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md) 与项目页；具体对照方法、任务集与逐项指标以 **原文 PDF** 为准（[参考来源](#参考来源)）。

## 结论

**ReactHuman 适合作为本期「部分开源」边界下的快速索引页，部署前请核对仓库/README 可运行性。**

1. 核心贡献：真正危险的反应要把每个动作执行出来测——突发家庭危险下的物理接地 MLLM 评测。
2. 开源结论：**部分开源** — 以项目页实际链接为准（入库日 2026-09-11）。
3. 横向对照见 [14 篇技术地图](../overview/dexterous-wm-humanoid-14-papers-technology-map.md)，避免与同 arXiv 重复造页。

## 关联页面

- [14 篇技术地图](../overview/dexterous-wm-humanoid-14-papers-technology-map.md)
- [VLA（Vision-Language-Action）](../methods/vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [Manipulation](../tasks/manipulation.md)

## 参考来源

- [reacthuman_arxiv_2609_10895.md](../../sources/papers/reacthuman_arxiv_2609_10895.md)
- [wechat 14篇盘点](../../sources/blogs/wechat_embodied_station_14_papers_dexterous_wm_humanoid_2026-09-11.md)
- [arXiv:2609.10895](https://arxiv.org/abs/2609.10895)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.10895)
- [项目页/资源](https://arxiv.org/abs/2609.10895)
