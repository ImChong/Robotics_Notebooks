---
type: entity
tags:
  - paper
  - wam
  - humanoid
  - xpeng
  - self-improvement
status: complete
updated: 2026-09-20
arxiv: "2609.17372"
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../entities/isaac-gr00t.md
sources:
  - ../../sources/papers/xpace_arxiv_2609_17372.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md
summary: "XPACE（arXiv:2609.17372）：WAM+Simulator 合一：联合预测视频与动作或给定动作预测视觉；无动作视频学动态；Simulator 生成 deviation→recovery 再微调 Policy；小鹏 IRON 真机。"
---

# XPACE（arXiv:2609.17372）

**XPACE**（*XPACE: Joint World and Action Modeling from Heterogeneous Experience*，[arXiv:2609.17372](https://arxiv.org/abs/2609.17372)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md)（2026-09-14–18）。

## 一句话定义

**WAM+Simulator 合一：联合预测视频与动作或给定动作预测视觉；无动作视频学动态；Simulator 生成 deviation→recovery 再微调 Policy；小鹏 IRON 真机。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界–动作模型 |
| VLA | Vision-Language-Action | 视觉–语言–动作 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

- 异构经验（人视频+机器人示范）需统一世界–动作模型；自生成恢复数据可闭环提升。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.17372](https://arxiv.org/abs/2609.17372) |
| **开源** | **待发布**（步骤 2.5，2026-09-20） |
| **方法摘要** | Shared video backbone; world-action joint prediction; simulator-driven recovery data self-improvement. |

## 源码运行时序图

**不适用**（截至 2026-09-20 未发布可运行官方代码或待核实）。

## 实验与评测

- XPENG IRON humanoid; human video skill transfer + recovery data gains（以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**XPACE 把 world simulator 用作 policy self-improvement 引擎，连接人视频与机器人示范。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [isaac-gr00t](../entities/isaac-gr00t.md)

## 参考来源

- [xpace_arxiv_2609_17372.md](../../sources/papers/xpace_arxiv_2609_17372.md)
- [wechat_senlanke_weekly_manipulation_2026-09-14_18.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-14_18.md)
- [arXiv:2609.17372](https://arxiv.org/abs/2609.17372)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.17372)
