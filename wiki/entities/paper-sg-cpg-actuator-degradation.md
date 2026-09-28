---
type: entity
tags:
  - paper
  - quadruped
  - cpg
  - rl
status: complete
updated: 2026-09-28
arxiv: "2609.25687"
related:
  - ../tasks/locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/sg-cpg-actuator-degradation_arxiv_2609_25687.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "SG-CPG（arXiv:2609.25687）：冻结健康 CPG + 严重度门控残差协调 + 弱腿振幅门；Go2 最高 93% 小腿力矩退化仍多数通过。"
---

# SG-CPG（arXiv:2609.25687）

**SG-CPG**（*SG-CPG: Severity-Gated Central Pattern Generators for Adaptive Quadruped Locomotion under Continuous Actuator Degradation*，[arXiv:2609.25687](https://arxiv.org/abs/2609.25687)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**冻结健康 CPG + 严重度门控残差协调 + 弱腿振幅门；Go2 最高 93% 小腿力矩退化仍多数通过。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 容错常把关节二值化为正常/失效。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.25687](https://arxiv.org/abs/2609.25687) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Frozen healthy CPG + severity-gated residual and amplitude gates. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- Go2 仿真 95% 强度损失仍 100% survival；真机 28/29 trials（Purdue，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**SG-CPG 用连续严重度门控延长健康 CPG 到渐进退化。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [sg-cpg-actuator-degradation_arxiv_2609_25687.md](../../sources/papers/sg-cpg-actuator-degradation_arxiv_2609_25687.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.25687](https://arxiv.org/abs/2609.25687)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.25687)
