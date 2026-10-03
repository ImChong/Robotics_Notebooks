---
type: entity
tags: [paper, manipulation, synthetic-data, sim2real, visual-perception]
status: complete
updated: 2026-10-02
arxiv: "2609.25653"
related:
  - ../tasks/manipulation.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day1_data_retargeting_2026_10_02.md
summary: "PhyVisGen 联合软夹爪接触仿真和路径追踪视觉渲染，从 RGB-D 场景造双臂操作示范，ACT 零真实示范迁移到 Franka。"
---

# PhyVisGen：Physically and Visually High-Fidelity Robotic Manipulation Data Generation

## 一句话定义

**PhyVisGen** 在同一合成数据流程中兼顾夹爪—物体的**物理接触**和相机看到的**视觉外观**，训练双臂操作策略。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RGB-D | Red Green Blue and Depth | 用于重建初始真实场景的彩色与深度观测 |
| ACT | Action Chunking with Transformers | 消费合成示范的下游模仿策略 |
| Sim2Real | Simulation to Reality | 合成训练到 Franka 真机执行 |

## 为什么重要

模拟抓取如果接触不真实，动作错；只把动作模拟好而透明物体、阴影不真实，视觉策略仍会错。两条误差要一起控制。

## 方法

从真实 RGB-D 场景建立资产，让软指尖、夹爪、物体在仿真中共同求解接触；将仿真状态交给路径追踪渲染，生成形变、透明外观、阴影和动作配对的数据，再以 ACT 训练。

## 实验与评测

文章报告只用合成示范训练的 ACT 在 Franka 真机五项任务成功率 **65%–95%**。透明物体任务凸显视觉渲染的重要性；该范围不表示任意环境与物体的稳定成功率。

## 与其他工作对比

相较纯 [Sim2Real](../concepts/sim2real.md) 动力学随机化，PhyVisGen 同时改善**视觉域**与**接触域**；不是人体到机器人骨架重定向方法。

## 结论

**合成操作数据的动作和画面必须共享一致的几何与接触状态，否则提高样本数也可能放大错误监督。**

1. 单独检查软指尖压缩和物体运动。
2. 透明外观、投影阴影应纳入视觉域验收。
3. 真机五项任务结果应逐项报告，不能只引用区间。

## 工程实践

对比真实/合成图像下的物体定位及闭环抓取误差，保持动作、渲染、时间戳一致。本次来源仅给出 [arXiv](https://arxiv.org/abs/2609.25653)，尚未确认可运行官方代码。

## 局限与风险

资产重建、软物理和路径追踪成本较高；Franka 任务不等于人形全身操作验证。

## 源码运行时序图

**不适用**：官方可运行实现未获核实。

## 关联页面

- [操作](../tasks/manipulation.md)
- [Sim2Real](../concepts/sim2real.md)
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md)

## 参考来源

- [Day 1 文章逐篇索引](../../sources/blogs/humanoid_motion_intelligence_day1_data_retargeting_2026_10_02.md)
- [论文](https://arxiv.org/abs/2609.25653)

## 推荐继续阅读

- [PhyVisGen arXiv](https://arxiv.org/abs/2609.25653)
