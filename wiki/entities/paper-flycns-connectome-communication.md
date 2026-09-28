---
type: entity
tags:
  - paper
  - locomotion
  - rl
  - communication
status: complete
updated: 2026-09-28
arxiv: "2609.28816"
related:
  - ../tasks/locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/flycns-connectome-communication_arxiv_2609_28816.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "FlyCNS（arXiv:2609.28816）：果蝇连接组启发局部模块+上下行路径；RL 联合学信息内容与发送时机；~21% 通信量保跟踪性能。"
---

# FlyCNS（arXiv:2609.28816）

**FlyCNS**（*FlyCNS: Connectome-Grounded Information Organization for Communication-Constrained Embodied Control*，[arXiv:2609.28816](https://arxiv.org/abs/2609.28816)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**果蝇连接组启发局部模块+上下行路径；RL 联合学信息内容与发送时机；~21% 通信量保跟踪性能。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 分布式腿控不能把所有传感持续传到中央。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.28816](https://arxiv.org/abs/2609.28816) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Connectome-inspired comm paths; RL co-learns content and timing. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 约 21% 通信量接近全通信策略（印第安纳大学，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**FlyCNS 把通信预算作为一等设计变量。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [flycns-connectome-communication_arxiv_2609_28816.md](../../sources/papers/flycns-connectome-communication_arxiv_2609_28816.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.28816](https://arxiv.org/abs/2609.28816)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.28816)
