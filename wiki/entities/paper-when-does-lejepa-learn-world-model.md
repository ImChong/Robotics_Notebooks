---
type: entity
tags: [paper, jepa, world-model, representation-learning, cshl, nyu, brown]
status: complete
updated: 2026-10-06
arxiv: "2605.26379"
venue: "arXiv 2026"
related:
  - ./article-videodb-jepa-world-models.md
  - ./paper-lejepa.md
  - ./paper-lewm.md
  - ../methods/model-based-rl.md
sources:
  - ../../sources/papers/when_does_lejepa_learn_world_model_arxiv_2605_26379.md
summary: "研究 LeJEPA 何时能学习可用于规划的世界表征；在线性可辨识性假设下建立理论结果，并报告像素输入潜空间控制实验。"
---

# When Does LeJEPA Learn a World Model?

**When Does LeJEPA Learn a World Model?**（arXiv:2605.26379）由 David Klindt、Yann LeCun、Randall Balestriero 提出。论文的问题不是“所有 embedding 是否都是 world model”，而是：在什么动力学和潜变量分布假设下，学习到的表征能恢复环境的潜在自由度，从而支持规划？

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| JEPA | Joint-Embedding Predictive Architecture | 在联合表征空间预测目标表征 |
| LeJEPA | LeJEPA | 以 alignment 与高斯分布正则为核心的 JEPA 变体 |
| OU | Ornstein–Uhlenbeck process | 具有均值回复特性的随机过程，论文用于表征状态转移条件 |
| MPC | Model Predictive Control | 用动力学模型评估动作序列并滚动重规划 |

## 核心原理与理论假设

在论文分析的世界类别中，潜变量服从高斯分布并经平稳加性噪声动力学演化时，alignment + Gaussian regularization 的 LeJEPA 表征可线性恢复真实潜变量（至正交变换）。作者证明此条件下的 Gaussian 分布具有特定唯一性，并给出近似可辨识性分析，连接到 latent-space planning。

```mermaid
flowchart TB
  World["潜变量与状态转移"] --> Observe["非线性观测"]
  Observe --> Encode["LeJEPA 编码"]
  Encode --> Align["alignment + Gaussian regularization"]
  Align --> Recover["线性可辨识的 latent 状态"]
  Recover --> Plan["潜空间规划实验"]
```

理论结论受作者明确设定约束。论文也讨论潜变量并非高斯、维度选择、有限样本和优化等限制；不能将定理解读为任意视觉表示必然等同真实世界状态。

## 实验与含义

论文报告从二维合成示例扩展到高维 latent，并进行像素输入机器人控制相关的潜空间实验。阅读实验时应区分：

- **表示识别：** 线性映射是否能恢复环境潜变量；
- **模型预测：** latent 动力学是否能在时间上准确预测；
- **规划控制：** 以模型优化动作后，闭环机器人任务是否成功。

前两者是第三者的必要参考，但不自动推出普遍的机器人任务成功率。

## 与相关工作的对比与关系

- [LeJEPA](./paper-lejepa.md) 提出 alignment 与 SIGReg 的表征学习配方；本论文研究其何时能恢复世界结构。
- [LeWorldModel](./paper-lewm.md) 进一步学习像素到未来 latent 的 action-conditioned dynamics。
- [VideoDB JEPA 长文](./article-videodb-jepa-world-models.md) 将本论文用于说明 latent 几何与规划的关系。

## 结论

**在潜变量服从高斯分布且经平稳加性噪声动力学演化的特定世界类别下，alignment + Gaussian regularization 的 LeJEPA 表征可线性恢复真实潜变量（至正交变换）。**

- 理论可辨识性受高斯潜变量与平稳噪声等前提约束，不可外推为任意视觉编码均自动恢复真实物理状态。
- 阅读像素控制与潜空间规划实验时，应拆开考察「表示线性恢复」「动力学预测」与「闭环规划成功率」三层指标。
- 潜变量维度选择、非高斯分布与有限样本优化误差仍会限制模型在复杂机器人控制任务上的迁移上界。

## 关联页面

- [Model-Based RL（基于模型的强化学习）](../methods/model-based-rl.md)：论文分析像素输入下所学世界表征与潜空间动力学模型能否支撑基于模型的强化学习与闭环规划控制。
- [Model Predictive Control (MPC，模型预测控制)](../methods/model-predictive-control.md)：论文将线性可辨识潜状态与动力学预测进一步连接到潜空间滚动优化与 MPC 动作规划实验。

## 参考来源

- [论文来源归档](../../sources/papers/when_does_lejepa_learn_world_model_arxiv_2605_26379.md)
- [arXiv:2605.26379](https://arxiv.org/abs/2605.26379)
- [LeJEPA 详情页](./paper-lejepa.md)
