---
type: entity
tags:
  - paper
  - wam
  - manipulation
status: complete
updated: 2026-09-28
arxiv: "2609.27656"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/internw0-physical-world-model_arxiv_2609_27656.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md
summary: "InternW0（arXiv:2609.27656）：非对称 Video Expert 低频预测 + Action Expert 高频动作；缓存 layer-wise K/V + context routing 修正，解耦世界预测与控制频率。"
---

# InternW0（arXiv:2609.27656）

**InternW0**（*InternW0: A Foundational Physical World Model for Efficient Real-World Interactions*，[arXiv:2609.27656](https://arxiv.org/abs/2609.27656)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**非对称 Video Expert 低频预测 + Action Expert 高频动作；缓存 layer-wise K/V + context routing 修正，解耦世界预测与控制频率。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- WAM 每步重生成未来视频难实时。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.27656](https://arxiv.org/abs/2609.27656) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Asymmetric video/action experts with cached KV routing. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 7200 h 异构数据；15 阶段实验室+移液真机（上海 AI Lab 等，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**InternW0 用 KV 缓存把 WAM 推到可部署频率。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [world-action-models](../concepts/world-action-models.md)

## 参考来源

- [internw0-physical-world-model_arxiv_2609_27656.md](../../sources/papers/internw0-physical-world-model_arxiv_2609_27656.md)
- [wechat_senlanke_weekly_manipulation_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)
- [arXiv:2609.27656](https://arxiv.org/abs/2609.27656)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.27656)
