---
type: entity
tags:
  - paper
  - humanoid
  - vla
  - force-control
status: complete
updated: 2026-10-03
arxiv: "2609.23968"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/opt2vla-force-aware-humanoid_arxiv_2609_23968.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "Opt2VLA（arXiv:2609.23968）：多任务 VLA 同时输出几何目标与连续接触力参考；RL WBC 跟踪；WTO 自动生成带力标签数据。"
---

# Opt2VLA（arXiv:2609.23968）

**Opt2VLA**（*Opt2VLA: Force-Aware Vision-Language-Action for Contact-Rich Humanoid Whole-Body Manipulation*，[arXiv:2609.23968](https://arxiv.org/abs/2609.23968)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**多任务 VLA 同时输出几何目标与连续接触力参考；RL WBC 跟踪；WTO 自动生成带力标签数据。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 几何相同但接触力需求不同的操作无法仅靠视觉区分。

## 流程总览

以下按本页已归纳的机制与资料绘制，表示模块或阅读路径关系。

```mermaid
flowchart TD
    N0["WTO 优化"]
    N1["带接触力标签数据"]
    N2["VLA 训练"]
    N3["几何目标"]
    N4["连续接触力参考"]
    N5["RL 全身跟踪"]
    N6["机器人执行"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N2 --> N4
    N3 --> N5
    N4 --> N5
    N5 --> N6
```

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.23968](https://arxiv.org/abs/2609.23968) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | VLA predicts motion + force reference; task-specific RL WBC tracking. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 三个 contact-rich 人形任务仿真+真机（Georgia Tech，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**Opt2VLA 把力参考纳入 VLA 动作接口，连接语义与接触力控制。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [world-action-models](../concepts/world-action-models.md)

## 参考来源

- [opt2vla-force-aware-humanoid_arxiv_2609_23968.md](../../sources/papers/opt2vla-force-aware-humanoid_arxiv_2609_23968.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.23968](https://arxiv.org/abs/2609.23968)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.23968)
