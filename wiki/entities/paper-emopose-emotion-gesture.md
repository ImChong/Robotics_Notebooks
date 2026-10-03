---
type: entity
tags:
  - paper
  - humanoid
  - social-hri
  - vlm
status: complete
updated: 2026-10-03
arxiv: "2609.23414"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/emopose-emotion-gesture_arxiv_2609_23414.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
summary: "EmoPose（arXiv:2609.23414）：VLM 选手势类/版本/强度/语音触发点；本地 14-DoF 动作库生成验证调度。"
---

# EmoPose（arXiv:2609.23414）

**EmoPose**（*EmoPose: Vision-Language Model Guided Emotion-Aware Gesture Generation for Humanoid Robots*，[arXiv:2609.23414](https://arxiv.org/abs/2609.23414)）来自 [senlanke 具身运控lab 周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)（2026-09-21–25）。

## 一句话定义

**VLM 选手势类/版本/强度/语音触发点；本地 14-DoF 动作库生成验证调度。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |
| MPC | Model Predictive Control | 模型预测控制 |

## 为什么重要

- VLM 直接出关节难保证可执行。

## 流程总览

以下按本页已归纳的机制与资料绘制，表示模块或阅读路径关系。

```mermaid
flowchart TD
    N0["语音与语义"]
    N1["VLM 手势规划"]
    N2["类别版本强度"]
    N3["语音触发时间"]
    N4["本地动作库"]
    N5["验证与调度"]
    N6["14-DoF 动作执行"]
    N0 --> N1
    N1 --> N2
    N1 --> N3
    N2 --> N4
    N4 --> N5
    N3 --> N5
    N5 --> N6
```

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.23414](https://arxiv.org/abs/2609.23414) |
| **开源** | **待发布**（步骤 2.5，2026-09-28） |
| **方法摘要** | VLM semantic planning + local verified gesture library. |

## 源码运行时序图

**不适用**（截至 2026-09-28 未发布可运行官方代码或待核实）。

## 实验与评测

- 开放式语言交互手势（港科广等，以 PDF 为准）。
- 读法：先对齐任务设定、传感器与成功定义，再解读 headline 数字。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见对应 [周更盘点](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md) 映射表，勿跨任务直接比 SR |
| **开源状态** | **待发布** — 部署前以项目页/arXiv 为准 |

## 结论

**EmoPose 分离语义规划与安全轨迹执行。**

1. 开源：**待发布**；勿凭公众号摘要臆断可复现性。
2. 指标须连同实验条件解读（仿真/真机、平台、成功阈值）。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [sim2real](../concepts/sim2real.md)

## 参考来源

- [emopose-emotion-gesture_arxiv_2609_23414.md](../../sources/papers/emopose-emotion-gesture_arxiv_2609_23414.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md)
- [arXiv:2609.23414](https://arxiv.org/abs/2609.23414)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2609.23414)
