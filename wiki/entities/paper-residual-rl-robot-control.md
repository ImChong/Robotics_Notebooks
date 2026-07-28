---
type: entity
tags: [paper, residual-learning, reinforcement-learning, td3, manipulation, assembly, real-world-rl, siemens, berkeley]
status: complete
updated: 2026-07-28
arxiv: "1812.03201"
related:
  - ../methods/residual-policy-learning.md
  - ./paper-residual-policy-learning.md
  - ../methods/reinforcement-learning.md
  - ../tasks/manipulation.md
  - ../concepts/contact-rich-manipulation.md
  - ./paper-reskill-residual-skill-policies.md
sources:
  - ../../sources/personal/residual-policy-reading-list.md
  - ../../sources/sites/residualrl-github-io.md
summary: "Residual Reinforcement Learning for Robot Control（ICRA 2019，Siemens/UC Berkeley/Hamburg TU）：u=π_H(s_m)+π_θ(s_m,s_o)，传统反馈控制器打底、TD3 残差处理接触与摩擦；Sawyer 真机积木插入约 3 小时学会，初姿扰动下 15/20 vs 手工控制器 2/20；官方代码未开源。"
---

# Residual Reinforcement Learning for Robot Control（Residual RL，ICRA 2019）

**Residual Reinforcement Learning for Robot Control**（Tobias Johannink, Shikhar Bahl, Ashvin Nair 共同一作；Siemens Corporation / UC Berkeley / Hamburg University of Technology，ICRA 2019，[arXiv:1812.03201](https://arxiv.org/abs/1812.03201)，[项目页](https://residualrl.github.io/)）把控制问题分解为**传统反馈控制器可解的部分**与**RL 学习的残差部分**，最终控制量为两者叠加。在 Sawyer 真机积木装配任务上，Residual RL 用约 **3 小时（约 8k 样本）** 学会在积木朝向随机扰动下完成插入，而手工控制器在同样扰动下仅 **2/20** 成功。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 本文中只训练残差策略，不从头学控制 |
| TD3 | Twin Delayed Deep Deterministic Policy Gradient | 底层 off-policy 算法（rlkit 实现），样本效率支撑真机训练 |
| MDP | Markov Decision Process | 问题建模框架；机器人状态 $s_m$ 与物体状态 $s_o$ 耦合 |
| DoF | Degrees of Freedom | Sawyer 7 自由度机械臂 + 平行夹爪 |
| MuJoCo | Multi-Joint dynamics with Contact | 仿真验证平台；真机实验与之对照 |
| Sim2Real | Simulation to Real | 仿真预训练策略初始化真机 Residual RL 的实验（VI-D 节） |

## 为什么重要

- **Residual 谱系的真机起点**：与 Silver et al. RPL **同期独立**提出「base + 残差」叠加控制，本文专攻**真实世界**接触丰富任务，证明残差分解能把真机 RL 的样本需求压到小时级。
- **制造业语境的清晰问题定义**：奖励分解 $r=f(s_m)+g(s_o)$——$f$（机器人几何目标）由传统控制器离线高效优化，$g$（物体接触/摩擦目标）由 RL 在线学习；这是理解「为什么不让 RL 从零学」最清楚的论述之一。
- **安全性论据**：纯 RL 训练期空间探索范围大，真机上危险；Residual RL 的探索被 base 约束在任务相关流形附近。

## 核心原理（方法）

### 系统论视角的问题分解

机器人（全驱动，状态 $s_m$）与环境物体（欠驱动，状态 $s_o$）耦合：$s_m$ 通过耦合矩阵 $B(s_m,s_o)$ 间接驱动物体。刚体项 $A,C$ 大致已知，**接触/摩擦主导的 $B$ 未知**——这正是传统反馈控制失效、需要试错调参的部分。

### 叠加控制律

$$u = \pi_H(s_m) + \pi_\theta(s_m, s_o)$$

- $\pi_H$：人工设计控制器（仿真用笛卡尔位置控制；真机用柔顺关节阻抗控制），提供 $s_m$ 子空间的指数稳定误差动态；
- $\pi_\theta$：TD3 训练的残差策略，输入含**物体状态**（积木位姿、夹爪力），优化含接触项的总奖励；
- 探索噪声只加在残差上：$u'_t = \pi_\theta(s_t) + \mathcal{N}_t + \pi_H(s_t)$。

### 真机设置

- **任务**：把手持积木插入桌面上两站立积木之间；积木可平面滑动、可绕 y 轴倾倒；站立积木必须保持 upright。
- **观测**：夹爪位置 + z 向力、视觉跟踪系统估计的积木位姿、目标位置。
- **奖励**：$r_t=-\|x_g-x_t\|_2-\lambda(\|\theta_l\|_1+\|\theta_r\|_1)-\mu\|X_g-X_t\|_2-\beta(\|\phi_l\|_1+\|\phi_r\|_1)$（位置误差 + 倾倒/偏航惩罚）。

## 实验与评测

| 实验 | 设置 | 结果 |
|------|------|------|
| 样本效率 | 残差 vs 纯 TD3（仿真+真机） | 残差收敛更快、最终回报更高；纯 RL 探索范围更大（真机危险） |
| 初姿扰动（真机） | 积木 ±20° 随机倾斜 | **Residual RL 15/20 vs 手工控制器 2/20**；约 8k 样本（≈3 h）学会轻推积木纠偏的接触行为 |
| 控制噪声（仿真） | $u'=u+\mathcal{N}(\mu,\sigma^2)$ | 残差对零均值噪声与**递增偏置**均保持性能；手工控制器随偏置增大急剧退化 |
| Sim2Real 初始化 | 仿真预训练策略当初始化 | **<1000 步**真机交互即解决任务（3 seeds） |

## 源码运行时序图

**不适用**（截至 2026-07-28 项目页 [residualrl.github.io](https://residualrl.github.io/) 仅提供论文与视频，未发布官方代码）。思想同源的同期工作 [Residual Policy Learning](./paper-residual-policy-learning.md) 有官方开源实现（[k-r-allen/residual-policy-learning](https://github.com/k-r-allen/residual-policy-learning)，归档见 [`sources/repos/residual-policy-learning.md`](../../sources/repos/residual-policy-learning.md)），可作为残差训练管线的可运行参照。

## 结论

**接触/摩擦主导的真机任务上，「传统控制器打底 + TD3 残差」把小时级真机训练变成现实；纯 RL 与纯手工控制器都被明显超越。**

1. **奖励分解即分工** — $f(s_m)$ 给控制器、$g(s_o)$ 给 RL；写不出 $g$ 的解析形式恰是用 RL 的理由。
2. **3 小时真机训练锚点** — 约 8k 样本学会 ±20° 初姿扰动下的插入（15/20 vs 2/20），这是 Residual 谱系最有代表性的真机数字。
3. **残差可吸收控制偏置** — 控制器带偏置噪声时 RL 学会反向补偿，对应现实中传感器漂移/标定误差场景。
4. **Sim 初始化 + 真机残差 = 高效迁移** — 仿真预训练做 base，真机残差 <1000 步收敛，是后续 sim2real 残差工作（ASAP、ResMimic）的先声。
5. **复现门槛** — 官方未开源；TD3（rlkit）+ 阻抗控制 + 视觉跟踪的组合需要自行搭建，环境建模成本主要在接触丰富的积木系统。

## 常见误区或局限

- **依赖定制视觉跟踪**：真机观测依赖外部相机跟踪积木位姿/角度，端到端视觉残差留作未来工作。
- **不是通用装配方案**：任务为单一积木插入族；多步装配序列未涉及。
- **base 设计仍要领域知识**：阻抗控制器与奖励塑形都需人工设计，只是「比从零 RL 便宜」而非「免设计」。

## 与其他工作对比

| 维度 | 本文（Residual RL） | [Silver RPL](./paper-residual-policy-learning.md) | 纯 TD3 | 手工控制器 |
|------|---------------------|---------------------------------------------------|--------|------------|
| 提出关系 | 同期独立（互引） | 同期独立（互引） | — | — |
| 主战场 | **真机**接触任务 | 仿真长视野稀疏奖励 | — | — |
| base 来源 | 人工反馈控制器 | 人工控制器 / MPC（含 CachedPETS） | 无 | 自身 |
| RL 算法 | TD3 | DDPG + HER | TD3 | — |
| 开源 | 未开源 | 已开源 | — | — |

## 关联页面

- [Residual Policy Learning 方法页](../methods/residual-policy-learning.md)
- [Residual Policy Learning（Silver）](./paper-residual-policy-learning.md)
- [ReSkill](./paper-reskill-residual-skill-policies.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)

## 推荐继续阅读

- 项目页与真机视频：<https://residualrl.github.io/>
- 同期 RPL 开源实现：<https://github.com/k-r-allen/residual-policy-learning>
- rlkit（TD3 实现基础）：<https://github.com/rail-berkeley/rlkit>

## 参考来源

- [Residual Policy / Residual RL 论文精读清单摘录](../../sources/personal/residual-policy-reading-list.md)
- [residualrl.github.io 项目页归档](../../sources/sites/residualrl-github-io.md)
- Johannink et al., *Residual Reinforcement Learning for Robot Control*, ICRA 2019. <https://arxiv.org/abs/1812.03201>
