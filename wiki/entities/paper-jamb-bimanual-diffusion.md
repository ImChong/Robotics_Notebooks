---
type: entity
tags:
  - paper
  - bimanual
  - diffusion
  - manipulation
status: complete
updated: 2026-09-28
arxiv: "2609.25322"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/jamb-bimanual-diffusion_arxiv_2609_25322.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md
summary: "JAMB（arXiv:2609.25322）：同一 Transformer 联合去噪双臂 action + 未来 3D point tracks，denoising 中互相修正。"
---

# JAMB（arXiv:2609.25322）

**JAMB**（*JAMB: Joint Action-Motion Diffusion for Bimanual Manipulation*，[arXiv:2609.25322](https://arxiv.org/abs/2609.25322)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**同一 Transformer 联合去噪双臂 action + 未来 3D point tracks，denoising 中互相修正。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 双臂 action-only 扩散不显式预测场景如何被改变。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.25322](https://arxiv.org/abs/2609.25322) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Joint denoising of bimanual actions and future point tracks. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- RoboTwin 2.0 16 任务 83.4%（CMU/密歇根，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**JAMB 用 future point tracks 把双臂交互写进扩散联合空间。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [world-action-models](../concepts/world-action-models.md)

## 参考来源

- [jamb-bimanual-diffusion_arxiv_2609_25322.md](../../sources/papers/jamb-bimanual-diffusion_arxiv_2609_25322.md)
- [wechat_senlanke_weekly_manipulation_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)
- [arXiv:2609.25322](https://arxiv.org/abs/2609.25322)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.25322)
