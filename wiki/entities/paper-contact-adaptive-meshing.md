---
type: entity
tags: [paper, deformable-object, mesh-refinement, contact-simulation, gpu, adaptive-resolution]
status: complete
updated: 2026-10-21
project_id: contact-adaptive-mesh
arxiv: "2610.14142"
related:
  - ../concepts/differentiable-simulation.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/simulation-evaluation-infrastructure.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/contact-adaptive-mesh_arxiv_2610_14142.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "接触自适应网格细化（arXiv:2610.14142）：利用求解器已有的几何、弹性与接触量为候选边打分，在 GPU 上并行选择互不冲突的边进行细化，只在接触与形变关键区域增加分辨率。"
---

# 按需细化：可变形物体操作的接触自适应网格划分

**这篇工作在仿真循环内按接触情况局部细化四面体网格，把自由度集中在发生接触与形变的区域。**

## 英文缩写速查

| 缩写 | English | 简要说明 |
|------|---------|----------|
| — | Tetrahedral mesh | 四面体网格，可变形体仿真的常用离散结构 |
| — | Conflict-free subset | 无冲突子集，指同一轮中互不相互影响的细化边 |
| GPU | Graphics Processing Unit | 图形处理器，用于并行计算 |

## 为什么重要

摘要指出，均匀的精细网格会在整个物体上引入不必要的自由度；粗网格计算快，但可能无法解析接触引起的形变。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.14142](https://arxiv.org/abs/2610.14142) |
| 作者 | Gabriel Johnson, Tom Olesch, Fei Liu（arXiv 作者列表） |
| 机构 | 未核实（公众号标注"待核实"） |
| 开源状态 | **未核查代码**（摘要页未列代码链接） |

## 核心原理

- **打分：** 用求解器已有的几何、弹性与接触量为候选边打分。
- **并行选择：** GPU 上选出一组互不冲突的边。
- **原地更新：** 在仿真循环中同时更新网格拓扑与物理状态。

## 工程实践

- 摘要在真实可变形物体操作序列上做了评测。
- 源码运行时序图：**不适用**（未核查到官方代码）。

## 结论

**一句话总判：接触驱动的局部细化可以在不牺牲关键接触解析的前提下降低可变形体仿真的自由度，但收益需要在目标任务上核对。**

1. **复用求解器已有量来决定细化** — 摘要强调不引入额外的昂贵计算。
2. **并行无冲突选择是工程关键** — 决定了方法能否在仿真循环中运行。
3. **效率与精度需成对报告** — 摘要中的加速比以原文为准。

## 局限与风险

- 细化只增加分辨率，粗化策略与误差控制在摘要中未展开（推测）。
- 结论依赖具体求解器与材料模型。

## 关联页面

- [可微仿真](../concepts/differentiable-simulation.md)
- [接触丰富操作](../concepts/contact-rich-manipulation.md)
- [仿真评测基础设施](../concepts/simulation-evaluation-infrastructure.md)

## 参考来源

- [论文归档](../../sources/papers/contact-adaptive-mesh_arxiv_2610_14142.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.14142](https://arxiv.org/abs/2610.14142)
