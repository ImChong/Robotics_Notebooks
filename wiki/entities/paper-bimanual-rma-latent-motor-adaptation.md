---
type: entity
tags: [paper, bimanual-manipulation, rapid-motor-adaptation, latent-adaptation, sim-to-real, contact-rich]
status: complete
updated: 2026-10-21
project_id: bimanual-rma
arxiv: "2610.14833"
related:
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
  - ../concepts/contact-rich-manipulation.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/bimanual-rma_arxiv_2610_14833.md
  - ../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md
summary: "Scaling Latent Motor Adaptation（arXiv:2610.14833）：把 RMA 式潜在运动适应扩展到双臂协调操作，在 ManiSkill 三个接触丰富任务上以观测–动作历史与双腕深度图保持特权策略的性能（据摘要）。"
---

# 将潜在运动适应扩展到协调双臂操作

**这篇工作检验 RMA 式的历史推断能否从单臂扩展到双臂：双臂策略需要从交互历史中推断物体质量、摩擦与形状等未观测属性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RMA | Rapid Motor Adaptation | 通过交互历史推断环境潜在变量并据此调整动作的方法 |
| YCB | Yale-CMU-Berkeley Object Set | 常用的物体数据集，本文用于随机化物体属性 |
| DoF | Degree of Freedom | 自由度 |

## 为什么重要

摘要指出，双臂协作带来更高维控制、耦合接触和共享物体受力，RMA 能否在这种设定下成立需要验证。

## 核心信息

| 项目 | 内容 |
|------|------|
| arXiv | [2610.14833](https://arxiv.org/abs/2610.14833) |
| 作者 | Seung-Bin Joo, Alessandro Albini, Luca Wiehe, Perla Maiolino, Daniel Olmeda Reino, João F. Henriques（arXiv 作者列表） |
| 机构 | 未核实（公众号标注"待核实"） |
| 会议 | 据公众号转述：IROS 2026 Bimanual Manipulation Workshop 接收 |
| 开源状态 | **未核查代码**（摘要页未列代码链接） |

## 核心原理

- **设定：** 三个 ManiSkill 接触丰富任务，物体质量、摩擦、尺度与形状随机化（YCB 物体）。
- **输入：** 观测–动作历史与双腕深度图。
- **结论（摘要）：** 双臂 RMA 在双臂抓取与抬起上保持特权策略性能；摘要还与一个缺少历史推断的对照做了比较（摘要截断，具体结果以原文为准）。

## 工程实践

- 仿真实验基于 ManiSkill；源码运行时序图：**不适用**（未核查到官方代码）。

## 结论

**一句话总判：潜在运动适应可以从单臂扩展到双臂协调操作，但依赖交互历史与深度观测的组合。**

1. **历史推断仍是核心** — 双臂设定中，历史信息用于估计未观测的物体属性。
2. **腕部深度有实际作用** — 摘要把双腕深度图列为输入之一。
3. **仿真结论需真机验证** — 实验主要在 ManiSkill 中完成。

## 局限与风险

- 摘要未给出具体数字，本页不引用。
- 随机化范围之外的物体与接触模型，泛化性未知。

## 关联页面

- [强化学习](../methods/reinforcement-learning.md)
- [Sim2Real](../concepts/sim2real.md)
- [接触丰富操作](../concepts/contact-rich-manipulation.md)

## 参考来源

- [论文归档](../../sources/papers/bimanual-rma_arxiv_2610_14833.md)
- [公众号综述摘录（33 篇 Manipulation）](../../sources/raw/wechat_senlanke_manipulation_33_papers_2026-10-12_16.md)

## 推荐继续阅读

- [arXiv:2610.14833](https://arxiv.org/abs/2610.14833)
