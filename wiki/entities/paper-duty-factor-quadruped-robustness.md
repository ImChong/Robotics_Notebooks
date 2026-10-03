---
type: entity
tags:
  - paper
  - quadruped
  - locomotion
status: complete
updated: 2026-10-03
arxiv: "2609.22073"
related:
  - ../tasks/locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/duty-factor-quadruped-robustness_arxiv_2609_22073.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "占空比预测鲁棒性（arXiv:2609.22073）：在 TO+LQR、学习控制、质心 MPC 中统一分析；占空比比步态名更能预测窄梁/扰动稳定性。"
---

# 占空比预测鲁棒性（arXiv:2609.22073）

**占空比预测鲁棒性**（*Duty Factor Predicts Robust Constrained Quadrupedal Locomotion Across Gait Types*，[arXiv:2609.22073](https://arxiv.org/abs/2609.22073)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**在 TO+LQR、学习控制、质心 MPC 中统一分析；占空比比步态名更能预测窄梁/扰动稳定性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 步态类别不足以解释受限环境稳定性。

## 流程总览

以下按本页已归纳的机制与资料绘制，表示模块或阅读路径关系。

```mermaid
flowchart TD
    N0["TO 与 LQR"]
    N1["学习控制"]
    N2["质心 MPC"]
    N3["占空比测量"]
    N4["窄梁与扰动测试"]
    N5["稳定性对照"]
    N6["地形宽度条件"]
    N0 --> N3
    N1 --> N3
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N6 --> N3
```

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.22073](https://arxiv.org/abs/2609.22073) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | Duty factor as cross-framework stability predictor; terrain-width-conditioned DF. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 四足窄梁与扰动（迈阿密/CMU，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**占空比是比名义步态更通用的鲁棒性旋钮。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [locomotion](../tasks/locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [duty-factor-quadruped-robustness_arxiv_2609_22073.md](../../sources/papers/duty-factor-quadruped-robustness_arxiv_2609_22073.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.22073](https://arxiv.org/abs/2609.22073)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.22073)
