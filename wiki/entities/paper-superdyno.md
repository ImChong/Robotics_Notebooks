---
type: entity
project_id: superdyno-humanoid-control
project: https://dynamics-humanoid.github.io/dynamics-humanoid/
openreview: https://openreview.net/forum?id=FPgKt7bA8w
venue: "ICLR 2025 Workshop on World Models"
tags: [paper, humanoid, whole-body-control, differentiable-simulation, learned-dynamics, reinforcement-learning, cmu]
status: complete
updated: 2026-10-23
related:
  - ../concepts/differentiable-simulation.md
  - ../methods/reinforcement-learning.md
  - ../concepts/behavior-foundation-model.md
  - ./paper-shac.md
sources:
  - ../../sources/papers/superdyno_iclr_2025_workshop.md
  - ../../sources/sites/dynamics-humanoid-project.md
summary: "SuperDyno（ICLR 2025 Workshop）学习可微神经动力学，并结合监督学习与 RL 扩展人形全身跟踪到数万段动作；论文所列项目页当前不可访问，未找到官方代码。"
---

# SuperDyno：可微神经动力学驱动的人形全身控制

**一句话定义：** SuperDyno 从模拟轨迹学习可微神经动力学模型，再结合监督学习与强化学习进行大规模人形全身跟踪。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SuperDyno | Scalable Humanoid Whole-Body Control via Differentiable Neural Network Dynamics | 论文提出的可微神经动力学控制框架 |
| DNN | Differentiable Neural Network | 近似人形系统状态转移的可微网络 |
| RL | Reinforcement Learning | 与监督学习结合训练全身控制器 |
| WBC | Whole-Body Control | 全身动作跟踪控制任务 |

## 为什么重要

SuperDyno 将可微世界模型用于人形控制：神经动力学预测未来状态，使跟踪目标的梯度能够作用于策略。论文瞄准动作库扩张后 model-free 强化学习的样本成本。

## 方法栈

仿真状态/动作数据 → 监督拟合神经动力学 → 可微模型预测与 rollout → 结合监督学习和 RL 优化策略 → 全身跟踪评测。代理模型的精度决定梯度可信度；面对训练分布之外的状态和接触模式，需回到原始仿真器检查偏差。

## 源码运行时序图

**不适用** — 截至 2026-10-23 未找到可访问的官方项目页或论文配套仓库；OpenReview 论文所列项目地址返回 404，暂无可对齐的可运行入口。

## 实验与评测

论文摘要报告较先前方法更高的样本效率和更低跟踪误差，并扩展到包含数万段动作片段的数据集。作者还报告中等规模神经动力学模型用于预测与策略优化，以及稀疏奖励和跨任务迁移实验。摘要没有列出完整数值，定量比较需核对论文正文表格。

## 与其他工作对比

| 路线 | 被求导模型 | 主要风险 |
|------|----------|----------|
| [SHAC](./paper-shac.md) | 显式可微物理仿真器 | 接触近似与梯度稳定性 |
| SuperDyno | 从仿真轨迹拟合的神经动力学 | 域外模型误差可能误导策略梯度 |
| Model-Free RL | 不要求模型可微 | 样本效率可能随动作库规模上升 |

## 结论

**SuperDyno 将可微 learned dynamics 与监督学习、强化学习组合成大规模人形动作跟踪流程；其收益取决于神经模型在策略访问分布中的精度。**

1. 方法结合模型学习和控制学习，而非只对固定物理模型求导。
2. 数万动作片段的扩展性是论文报告的结果；外推需核对数据集、机器人和训练预算。
3. 代理模型可用于预测与策略优化，但预测精度不自动保证策略梯度可靠。
4. 所列项目页当前 404，未找到官方源码，暂无法依官方实现复现。

## 局限与风险

- 学习的动力学近似受训练覆盖、接触分布和仿真随机化影响。
- 策略可能利用代理模型误差，需在原始仿真器复评。
- 当前可核验材料以论文为主，源码状态待后续复核。

## 关联页面

- [可微仿真](../concepts/differentiable-simulation.md)
- [SHAC](./paper-shac.md)
- [强化学习](../methods/reinforcement-learning.md)
- [行为基础模型](../concepts/behavior-foundation-model.md)

## 参考来源

- [SuperDyno 论文归档](../../sources/papers/superdyno_iclr_2025_workshop.md)
- [项目页核查](../../sources/sites/dynamics-humanoid-project.md)
- [OpenReview 论文](https://openreview.net/forum?id=FPgKt7bA8w)

## 推荐继续阅读

- [SuperDyno OpenReview 页面](https://openreview.net/forum?id=FPgKt7bA8w)
- [SuperDyno 论文 PDF](https://openreview.net/pdf?id=FPgKt7bA8w)
