---
type: entity
tags:
  - paper
  - vla
  - manipulation
  - safety
status: complete
updated: 2026-10-03
arxiv: "2609.26313"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../concepts/world-action-models.md
sources:
  - ../../sources/papers/safeloop-vla-rollback_arxiv_2609_26313.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md
summary: "SafeLoop（arXiv:2609.26313）：外部 risk predictor 预测碰撞/物体失败概率与时间；noop/record/rollback；回到安全关节态再查 VLA。"
---

# SafeLoop（arXiv:2609.26313）

**SafeLoop**（*SafeLoop: Risk-Aware Rollback for Vision-Language-Action Manipulation*，[arXiv:2609.26313](https://arxiv.org/abs/2609.26313)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**外部 risk predictor 预测碰撞/物体失败概率与时间；noop/record/rollback；回到安全关节态再查 VLA。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- 长程 VLA 小误差累积成不可逆失败。

## 流程总览

以下按本页已归纳的机制与资料绘制，表示模块或阅读路径关系。

```mermaid
flowchart TD
    N0["VLA 候选动作"]
    N1["外部风险预测"]
    N2["noop 或执行"]
    N3["安全状态记录"]
    N4["风险触发 rollback"]
    N5["回到安全关节态"]
    N6["重新查询 VLA"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N1 --> N4
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 --> N0
```

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.26313](https://arxiv.org/abs/2609.26313) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | External risk predictor + checkpoint rollback without fine-tuning VLA. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- LIBERO 24 任务 + 3 真机；危险事件约降 70%（南大/港科广/北理工，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**SafeLoop 是不改 VLA 参数的安全回滚包装层。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)
- [world-action-models](../concepts/world-action-models.md)

## 参考来源

- [safeloop-vla-rollback_arxiv_2609_26313.md](../../sources/papers/safeloop-vla-rollback_arxiv_2609_26313.md)
- [wechat_senlanke_weekly_manipulation_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-09-21_25.md)
- [arXiv:2609.26313](https://arxiv.org/abs/2609.26313)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.26313)
