---
type: entity
tags: [paper, project, humanoid, locomotion, flow-matching, reinforcement-learning, policy-distillation, unitree-go2, unitree-g1]
status: complete
updated: 2026-10-09
project_id: rfpo-rectified-flow-policy-optimization
arxiv: "2610.10453"
project: "https://aigeeksgroup.github.io/RFPO/"
repo: "https://github.com/AIGeeksGroup/RFPO"
related:
  - ../tasks/humanoid-locomotion.md
  - ../tasks/loco-manipulation.md
  - ../concepts/flow-matching-embodied-policy.md
sources:
  - ../../sources/papers/rfpo_arxiv_2610_10453.md
  - ../../sources/repos/rfpo.md
summary: "RFPO：面向具身控制的 reward-aware online reflow 与多预算蒸馏，把 64 步 flow policy 压到单次 Euler 推理；论文报告 Go2 单步保留 98.5% reward、推理延迟 4.39 ms→0.08 ms；官方仓库主要含 Go2/G1 Isaac Lab 实验，未含部署包。"
---

# RFPO：把具身 Flow policy 压到单步推理

**RFPO**（*Rectified Flow Policy Optimization for Embodied Control*，[arXiv:2610.10453](https://arxiv.org/abs/2610.10453)，[项目页](https://aigeeksgroup.github.io/RFPO/)，[GitHub](https://github.com/AIGeeksGroup/RFPO)）研究如何在不明显损失控制回报的情况下，把需要多步积分的 flow policy 变成单步控制器。论文不是泛化地“减少采样步数”，而是通过 reward-aware online reflow 与多预算动作蒸馏，显式训练一个能在不同积分预算下工作的学生策略。

## 一句话理解

**把 64 步生成式动作积分成本，转化为训练期的策略蒸馏问题，让部署时只执行一次 Euler 更新。**

## 英文缩写速查

| 缩写 | 英文全称 | 本文含义 |
|---|---|---|
| RFPO | Rectified Flow Policy Optimization | 面向具身控制的流策略优化方法 |
| ODE | Ordinary Differential Equation | flow policy 采样所需求解的常微分方程 |
| PPO | Proximal Policy Optimization | 论文中的冻结 Gaussian 控制器 / reward teacher |
| Reflow | Rectified Flow | 重新学习更直、更易少步积分的流轨迹 |
| Isaac Lab | Isaac Lab simulation framework | 官方代码提供的主要并行仿真实验环境 |

## 问题与方法

标准 flow policy 通过沿连续时间向量场积分生成动作。训练时若用较多积分步，动作质量可以较好，但实时机器人控制会付出逐步推理成本；直接把推理步数砍到 1，则会出现论文所称的 few-step discretization gap。RFPO 用在线奖励与策略动作作为监督，重整学生流轨迹，而不是直接把 64 步模型硬截成一步。

~~~mermaid
flowchart TB
  gaussian["冻结的 Gaussian PPO 控制器
提供动作与奖励参照"]
  student["Flow 学生策略
不同 Euler 预算共同训练"]
  reflow["Reward-aware online reflow
对齐策略动作与奖励"]
  regularize["自适应计算正则
约束动作质量 / 推理预算"]
  deploy["部署
单次 Euler 更新"]
  eval["Go2 / G1 等具身控制评测"]
  gaussian --> reflow
  student --> reflow
  reflow --> student
  student --> regularize
  regularize --> deploy
  deploy --> eval
~~~

### 核心训练设计

1. **Online reflow：** 用策略执行和回报驱动学生 flow 轨迹重整，使少步离散积分更接近完整轨迹的控制效果。
2. **多预算动作蒸馏：** 不只在单一积分步数上监督；学生在多个推理预算下与冻结的 Gaussian PPO 控制器对齐，论文也评估从 64、32、16、8、4 到 1 步的性能。
3. **自适应计算正则：** 将推理预算与动作质量一并纳入优化，鼓励策略在降低计算量时仍保持有用控制输出。
4. **单步部署：** 训练完成后只需一次 Euler 更新生成动作，不再执行 64 次积分。

## 结果

- **评测平台：** 论文覆盖 Unitree Go2、Spot、H1、G1 的控制评估；报告在不同初始噪声和积分预算下的回报。
- **Go2 单步结果：** 单步策略保留 64 步基线 **98.5%** 的 reward。
- **推理延迟：** Go2 mean onboard inference latency 从 **4.39 ms 降到 0.08 ms**，约 **54.9×** 加速。
- **总体比较：** 论文报告四种机器人上的 one-step return 与 64-step 值相差不超过 **2.4%**；包含真实机器人单步 locomotion 稳定性演示。

以上数值是论文指定平台与协议下的结果，不应直接外推为所有 flow policy 或控制任务都能无损单步化。

## 代码与复现状态

官方 [GitHub 仓库](https://github.com/AIGeeksGroup/RFPO)提供 Isaac Lab locomotion 与 manipulation 实验代码；README 将其描述为 FPO++ 实验代码，含 Go2/G1 方法实现。仓库文档给出 submodule 初始化、环境安装、训练和不同 Euler 步数评估的入口。**论文覆盖 Spot/H1 的结果不等于仓库已提供这些本体的实现；README 明确说明部署包未包含。**因此，本页把代码定位为训练 / 仿真复现实验，而非可直接刷入机器人运行的完整部署栈。

## 局限与阅读边界

- 论文强调的是受控任务内的少步流策略优化；具体泛化能力仍取决于任务、控制器及训练数据。
- “单步”指一次 Euler 更新，并不表示整个机器人系统只进行一次计算或不需要底层伺服控制。
- 官方仓库含有实验代码，但未提供 deployment packages；复现实机延迟与真机部署仍需额外系统集成。
- 论文对四个本体报告的性能接近是该实验集合上的结论，不能推断任意任务、噪声和硬件都保持相同比例。

## 关联页面

- [人形 locomotion](../tasks/humanoid-locomotion.md) — 具身运动控制任务入口。
- [Loco-Manipulation](../tasks/loco-manipulation.md) — RFPO 论文同时报告 manipulation 实验。
- [Flow Matching](../methods/flow-matching.md) — 连续流生成及其数值积分背景。

## 参考来源

- [RFPO 论文归档](../../sources/papers/rfpo_arxiv_2610_10453.md)
- [RFPO 官方代码归档](../../sources/repos/rfpo.md)
