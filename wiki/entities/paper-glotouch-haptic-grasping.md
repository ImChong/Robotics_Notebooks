---
type: entity
tags:
  - paper
  - haptic
  - grasping
  - manipulation
status: complete
updated: 2026-09-28
arxiv: "2609.27695"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/glotouch-haptic-grasping_arxiv_2609_27695.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md
summary: "GLoTouch（arXiv:2609.27695）：全局探针搜索+局部 visuotactile 与 3D 模型匹配；无外部视觉。"
---

# GLoTouch（arXiv:2609.27695）

**GLoTouch**（*GLoTouch: Global-to-Local Haptic Perception Using a Parallel Gripper for Object Search, Recognition, and Grasping Without External Vision*，[arXiv:2609.27695](https://arxiv.org/abs/2609.27695)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**全局探针搜索+局部 visuotactile 与 3D 模型匹配；无外部视觉。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 平行夹爪触觉范围小，难同时搜索与识别。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.27695](https://arxiv.org/abs/2609.27695) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Global probe search then local visuotactile model matching. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 黑暗/低光无视觉抓取（以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**GLoTouch 把全局搜索与局部触觉识别拆成两阶段。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [world-action-models](../concepts/world-action-models.md)

## 参考来源

- [glotouch-haptic-grasping_arxiv_2609_27695.md](../../sources/papers/glotouch-haptic-grasping_arxiv_2609_27695.md)
- [wechat_senlanke_weekly_manipulation_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)
- [arXiv:2609.27695](https://arxiv.org/abs/2609.27695)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.27695)
