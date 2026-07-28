# residualrl.github.io（Residual RL for Robot Control 项目页）

- **标题：** Residual Reinforcement Learning for Robot Control
- **类型：** site / project-page
- **URL：** <https://residualrl.github.io/>
- **配套论文：** [Residual Reinforcement Learning for Robot Control（arXiv:1812.03201）](https://arxiv.org/abs/1812.03201)
- **代码：** 页面仅提供 preprint 与视频链接，**未列代码仓库**（截至 2026-07-28 核查）
- **入库日期：** 2026-07-28

## 一句话摘要

Johannink et al.（Siemens / UC Berkeley / Hamburg TU，ICRA 2019）的项目页：Residual RL 将传统反馈控制器与 RL 叠加（$u=\pi_H(s_m)+\pi_\theta(s_m,s_o)$），在 Sawyer 真机积木装配任务上约 3 小时学会抗初姿扰动的插入技能；页面提供论文与演示视频。

## 公开信息要点（截至入库日）

- **机构：** Siemens Corporation；UC Berkeley；Hamburg University of Technology（前三作者共同一作）。
- **页面内容：** 摘要、preprint 链接、任务演示视频；无 Code / Resources 区。
- **视频要点：** 真机积木插入（两站立积木之间），积木朝向随机（正直 / ±20° 倾斜）。
- **与 RPL 的关系：** 与 Silver et al. 的 Residual Policy Learning（arXiv:1812.06298）**同期独立提出**；思想同为「base + 残差」， Johannink 侧重真机接触任务，Silver 侧重仿真长视野稀疏奖励；代码托管在 RPL 侧（见 [`sources/repos/residual-policy-learning.md`](../repos/residual-policy-learning.md)）。

## 为何值得保留

Residual RL 是「控制器打底 + RL 补偿」范式最具代表性的早期真机证据：3 小时 / 约 8k 样本即可在真机上学会接触丰富的插入修正；项目页是核查其开源状态（未开源）的一手依据。

## 对 wiki 的映射

- 实体页：[paper-residual-rl-robot-control](../../wiki/entities/paper-residual-rl-robot-control.md)
- 方法页：[residual-policy-learning](../../wiki/methods/residual-policy-learning.md)
