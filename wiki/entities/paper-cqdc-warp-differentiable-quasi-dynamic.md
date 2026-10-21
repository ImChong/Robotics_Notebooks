---
type: entity
tags: [paper, simulation, differentiable-simulation, quasi-dynamic, gpu, model-predictive-control, contact-rich]
status: complete
updated: 2026-10-21
project_id: cqdc-warp
arxiv: "2610.21349"
related:
  - ../concepts/differentiable-simulation.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/cqdc-warp_arxiv_2610_21349.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "CQDC-Warp（arXiv:2610.21349）：GPU 加速的可微准动态仿真器，用于批量 MPC 与接触引导的轨迹优化；摘要报告在 Allegro 手内旋转基准的 17,408 次试验中成功率 79.8%，高于 CFree（73.7%）与 IMPACT（65.2%）。"
---

# CQDC-Warp：GPU 加速的可微准动态仿真

**CQDC-Warp 把准动态接触仿真放到 GPU 上并保持可微，使接触引导的轨迹优化可以批量运行。**

## 英文缩写速查

| 缩写 | English | 简要说明 |
|------|---------|----------|
| MPC | Model Predictive Control | 模型预测控制，滚动优化未来动作序列 |
| QDC | Quasi-Dynamic Contact | 准动态接触，在时间步之间近似接触与摩擦的仿真方式（推测：CQDC 的全称按此理解） |
| GPU | Graphics Processing Unit | 图形处理器，用于批量并行计算 |

## 为什么重要

摘要指出，基于梯度的规划有前景，但同时优化运动与接触决策很难；许多规划器运行在 CPU 上，吞吐量有限。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.21349](https://arxiv.org/abs/2610.21349) |
| 作者 | Yanchen Ding, Jiayi Chen, Jianfeng Xu, Wei Zhao, Li Yi, He Wang（arXiv 作者列表） |
| 机构 | 未核实（公众号标注"待核实"） |
| 开源状态 | **未核查代码**（摘要页未列代码链接） |

## 核心原理

- **批量可微仿真：** GPU 上并行运行多条候选轨迹。
- **接触引导：** 在规划前指定意图接触，并以软代价鼓励其出现；接触几何与摩擦响应由仿真动力学决定，不预设接触模式时间表。

## 工程实践

- 基准为 17 个物体的 Allegro 手内旋转任务（摘要）。
- 源码运行时序图：**不适用**（未核查到官方代码）。

## 结论

**一句话总判：把准动态接触仿真做成可微的批量 GPU 计算，可以让接触引导的 MPC 在同一基准上优于两种对照方法（据摘要）。**

1. **接触意图可以软约束** — 摘要不预设接触模式时间表。
2. **吞吐量是 MPC 的实际瓶颈之一** — 方法以 GPU 批量计算为出发点。
3. **对比结论限于单一基准** — 17,408 次试验集中在 Allegro 手内旋转。

## 局限与风险

- 准动态近似对高速动力学的适用范围，摘要未讨论。
- 结论局限于手内旋转任务，对其他接触任务的泛化未知。

## 关联页面

- [可微仿真](../concepts/differentiable-simulation.md)
- [接触丰富操作](../concepts/contact-rich-manipulation.md)
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md)

## 参考来源

- [论文归档](../../sources/papers/cqdc-warp_arxiv_2610_21349.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.21349](https://arxiv.org/abs/2610.21349)
