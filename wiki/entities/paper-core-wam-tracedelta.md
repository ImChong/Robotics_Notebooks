---
type: entity
tags:
  - paper
  - wam
  - manipulation
status: complete
updated: 2026-09-28
arxiv: "2609.27314"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/core-wam-tracedelta_arxiv_2609_27314.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md
summary: "CoRe-WAM（arXiv:2609.27314）：TraceDelta：tracking 对齐历史特征到当前位置，signed feature difference 作 temporal residual；冻结 Motus 仅训 1.59M。"
---

# CoRe-WAM（arXiv:2609.27314）

**CoRe-WAM**（*CoRe-WAM: Correspondence-Aligned Temporal Residuals for World Action Models*，[arXiv:2609.27314](https://arxiv.org/abs/2609.27314)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**TraceDelta：tracking 对齐历史特征到当前位置，signed feature difference 作 temporal residual；冻结 Motus 仅训 1.59M。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 相机/物体运动使同像素比较失效。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.27314](https://arxiv.org/abs/2609.27314) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Correspondence-aligned TraceDelta adapter on frozen WAM. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- RoboTwin 2.0 50 任务 92.22% clean；可接 StarVLA（HKUST，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**TraceDelta 是参数高效的 WAM 时序对齐插件。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [world-action-models](../concepts/world-action-models.md)

## 参考来源

- [core-wam-tracedelta_arxiv_2609_27314.md](../../sources/papers/core-wam-tracedelta_arxiv_2609_27314.md)
- [wechat_senlanke_weekly_manipulation_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)
- [arXiv:2609.27314](https://arxiv.org/abs/2609.27314)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.27314)
