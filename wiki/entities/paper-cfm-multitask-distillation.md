---
type: entity
tags:
  - paper
  - flow-matching
  - manipulation
status: complete
updated: 2026-10-03
arxiv: "2609.28107"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/cfm-multitask-distillation_arxiv_2609_28107.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md
summary: "CFM 多任务蒸馏（arXiv:2609.28107）：多 single-task CFM Expert→蒸馏 velocity field 到共享 Multi-Task CFM，保留 FM demo objective。"
---

# CFM 多任务蒸馏（arXiv:2609.28107）

**CFM 多任务蒸馏**（*Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching*，[arXiv:2609.28107](https://arxiv.org/abs/2609.28107)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**多 single-task CFM Expert→蒸馏 velocity field 到共享 Multi-Task CFM，保留 FM demo objective。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 每任务单独 CFM 成本高；混合训练易干扰。

## 流程总览

以下按本页已归纳的机制与资料绘制，表示模块或阅读路径关系。

```mermaid
flowchart TD
    N0["多任务示范"]
    N1["单任务 CFM 专家"]
    N2["专家 velocity fields"]
    N3["共享 Multi-Task CFM"]
    N4["示范 FM objective"]
    N5["多任务动作生成"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N0 --> N4
    N4 --> N3
    N3 --> N5
```

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.28107](https://arxiv.org/abs/2609.28107) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Distill expert velocity fields into shared multitask CFM. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- RLBench 优于直接混合多任务训练（弗莱堡，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**蒸馏 velocity field 比蒸馏最终动作更适合多任务 CFM。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [world-action-models](../concepts/world-action-models.md)

## 参考来源

- [cfm-multitask-distillation_arxiv_2609_28107.md](../../sources/papers/cfm-multitask-distillation_arxiv_2609_28107.md)
- [wechat_senlanke_weekly_manipulation_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)
- [arXiv:2609.28107](https://arxiv.org/abs/2609.28107)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.28107)
