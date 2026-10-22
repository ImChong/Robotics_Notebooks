---
type: entity
tags: [paper, deformable-object, cloth-simulation, neural-ode, physics-informed, continuous-time]
status: complete
updated: 2026-10-21
project_id: continuous-cloth-pinn
arxiv: "2610.14342"
related:
  - ../concepts/differentiable-simulation.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/continuous-cloth-pinn_arxiv_2610_14342.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "ContinuousCloth（arXiv:2610.14342）：在神经轨迹模型中嵌入弹簧–质点 ODE 的物理信息神经求解器，可在任意时间戳查询布料状态；摘要称纯物理设定下运行时间最多降低 17.4 倍，并给出插值与参数估计的收益。"
---

# 连续时间布料动力学：物理信息神经常微分方程

**ContinuousCloth 把布料的弹簧–质点方程写进神经轨迹模型，使轨迹可以在任意时刻查询，并可通过自动微分得到速度与加速度。**

## 英文缩写速查

| 缩写 | English | 简要说明 |
|------|---------|----------|
| ODE | Ordinary Differential Equation | 常微分方程 |
| PINN / Physics-informed | Physics-Informed Neural Network | 物理信息神经网络，在训练中加入物理约束 |
| — | Spring–mass model | 弹簧–质点模型，布料的常用离散物理模型 |

## 为什么重要

摘要指出，经典的时间步进模拟器只在离散积分网格上给出轨迹；离散时间的预测模型也难以描述不同时间间隔下的布料运动。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.14342](https://arxiv.org/abs/2610.14342) |
| 作者 | Yingdong Ru, Lipeng Zhuang, Zhuo He, Gerardo Aragon Camarasa（arXiv 作者列表） |
| 机构 | 未核实（公众号标注"待核实"） |
| 开源状态 | **未核查代码**（摘要页未列代码链接） |

## 核心原理

- **连续时间：** 在任意时间戳查询位移、速度与加速度（通过自动微分）。
- **物理约束：** 弹簧–质点 ODE 嵌入神经轨迹模型。
- **稀疏观测：** 与稀疏、带噪声的观测融合时，该 ODE 作为归纳偏置用于插值与参数估计。

## 工程实践

- 未发现官方代码入口；源码运行时序图：**不适用**。

## 结论

**一句话总判：把布料的物理 ODE 作为神经模型的约束，可以同时获得连续查询能力与更少的计算时间，但结论限于摘要中的基准设定。**

1. **连续查询是离散求解器的补充** — 摘要强调可在任意时刻取值。
2. **物理先验有助于稀疏观测** — 插值与参数估计是摘要给出的两类下游任务。
3. **运行时间收益依赖基准** — 17.4 倍是摘要中的上限表述，不宜外推。

## 局限与风险

- 六个边界条件场景的结论，未必覆盖复杂接触与多物体交互（推测）。
- 与高精度求解器相比的精度差距，需以原文表格为准。

## 关联页面

- [可微仿真](../concepts/differentiable-simulation.md)
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md)
- [操作任务](../tasks/manipulation.md)

## 参考来源

- [论文归档](../../sources/papers/continuous-cloth-pinn_arxiv_2610_14342.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.14342](https://arxiv.org/abs/2610.14342)
