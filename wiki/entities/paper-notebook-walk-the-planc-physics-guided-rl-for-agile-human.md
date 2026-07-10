---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2601.06286"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_walk-the-planc.md
summary: "踏脚石/稀疏落脚点上的人形行走，最难的是「敏捷」和「精准落脚」要同时满足：纯 model-free RL 在这种离散、受约束地形上很难学，常常退化成原地站着不动；纯模型法（落脚规划）落脚精准但动作保守、对未建模动力学不鲁棒。PLANC 把两者缝起来——用一个 降阶 LIP 落脚规划器实时生成「动力学一致」的全状态参考轨迹，再用 控制李雅普诺夫函数（CLF）奖励把 RL 策略引导到这条物理可行的参考上，最终在 Unitree G1 上实现既快又准、可真机部署的踏脚石行走。"
---

# Walk the PLANC

**Walk the PLANC: Physics-Guided RL for Agile Humanoid Locomotion on Constrained Footholds** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：05_Locomotion），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

踏脚石/稀疏落脚点上的人形行走，最难的是「敏捷」和「精准落脚」要同时满足：纯 model-free RL 在这种离散、受约束地形上很难学，常常退化成原地站着不动；纯模型法（落脚规划）落脚精准但动作保守、对未建模动力学不鲁棒。PLANC 把两者缝起来——用一个 降阶 LIP 落脚规划器实时生成「动力学一致」的全状态参考轨迹，再用 控制李雅普诺夫函数（CLF）奖励把 RL 策略引导到这条物理可行的参考上，最终在 Unitree G1 上实现既快又准、可真机部署的踏脚石行走。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| LIP | Linear Inverted Pendulum | 线性倒立摆，双足步态规划的经典降阶模型 |
| HLIP | Hybrid / Horizontal LIP | LIP 的混合系统变体，用于步态周期落脚预测 |
| RoM | Reduced-order Model | 降阶模型，相对全身动力学的简化表示 |
| CLF | Control Lyapunov Function | 控制李雅普诺夫函数，用 V(η) 度量输出跟踪误差并约束其衰减 |
| PPO | Proximal Policy Optimization | 近端策略优化，本文 RL 训练算法 |
| MPC | Model Predictive Control | 模型预测控制，对比的传统模型法 |
| DoF | Degree of Freedom | 自由度；G1 共 21 个驱动自由度 |

## 为什么重要

- **「规划器当参考 + CLF 当引导」是可迁移范式**：把降阶模型的物理结构作为 RL 的软引导，而非硬约束，适用于踏脚石、沟壑、稀疏落脚点等强离散接触地形。
- **CLF 奖励提供物理可解释的塑形**：相比堆砌手工奖励项，CLF 衰减条件给出明确的稳定性目标，训练更稳、更不易退化。
- **师生蒸馏打通部署**：特权教师 → 无特权学生 → PPO 抗噪微调，是从仿真特权信息走向真机的成熟工程路线。
- **限制**：LIP 是降阶近似，复杂富接触/三维地形下可能损失最优性；真机依赖动捕 + 高程图获取落脚点，野外感知尚未完全自洽；评测以踏脚石/稀疏落脚为主，更广义地形泛化待验证。

## 解决什么问题

1. **受约束落脚点难学**：踏脚石、稀疏/离散落脚点要求脚必须落在指定可行区域，纯 model-free RL 探索效率低，常退化为保守的「原地踏步/站立」。 2. **敏捷 vs. 精准的矛盾**：既要动态、敏捷地迈步跨越，又要每一步都精准落到目标点，奖励整形很难两全。 3. **模型法太保守、RL 太脆弱**：传统降阶模型规划落脚精准但对未建模动力学鲁棒性差、动作保守；端到端 RL 鲁棒但缺乏物理结构、训练不稳定。

**目标**：让降阶模型的「物理结构与落脚精度」去引导 RL 的「鲁棒性与动态能力」，在受约束地形上又快又准地行走。

## 核心机制

1. **物理引导而非纯奖励整形**：用降阶 LIP 规划器在线生成动力学一致的参考，再以 CLF 奖励把 RL 引导到该参考上，兼得模型法的落脚精度与 RL 的鲁棒性。
2. **CLF 作为可学习的稳定性塑形信号**：把控制李雅普诺夫函数的衰减条件写进奖励，使训练围绕物理一致轨迹收敛，显著优于纯 model-free 基线（后者在难地形上退化为站立）。
3. **师生蒸馏 + PPO 微调**：从特权教师蒸馏到无特权学生，再在噪声/部分可观测下微调，打通从特权训练到真机部署的链路。
4. **真机验证的受约束行走**：Unitree G1 上实现踏脚石/稀疏落脚点的敏捷精准行走，对未见石块深度零样本泛化，并能抵抗 ±100 Nm / 0.2s 的外部推力扰动。

方法拆解（深读笔记小节）：核心：降阶落脚规划器 + CLF 奖励引导 RL；师生蒸馏三阶段管线；训练与部署设置。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Walk_the_PLANC__Physics-Guided_RL_for_Agile_Humanoid_Locomotion_on_Constrained_Footholds/Walk_the_PLANC__Physics-Guided_RL_for_Agile_Humanoid_Locomotion_on_Constrained_Footholds.html> |
| arXiv | <https://arxiv.org/abs/2601.06286> |
| 机构 | Caltech AMBER Lab（加州理工） |
| 作者 | Min Dai, William D. Compton, Junheng Li, Lizhi Yang, Aaron D. Ames |
| 发表 | 2026-01-09 (arXiv) |
| 项目主页 | [caltech-amber.github.io/planc](https://caltech-amber.github.io/planc/) |
| 源码 | [匿名仓库 anonymous.4open.science](https://anonymous.4open.science/r/robot_rl-E4FF)（以项目页后续正式开源为准） |
| 笔记阅读日期 | 2026-06-30 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_walk-the-planc.md](../../sources/papers/humanoid_pnb_walk-the-planc.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Walk_the_PLANC__Physics-Guided_RL_for_Agile_Humanoid_Locomotion_on_Constrained_Footholds/Walk_the_PLANC__Physics-Guided_RL_for_Agile_Humanoid_Locomotion_on_Constrained_Footholds.html>
- 论文：<https://arxiv.org/abs/2601.06286>

## 推荐继续阅读

- [机器人论文阅读笔记：Walk the PLANC](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/05_Locomotion/Walk_the_PLANC__Physics-Guided_RL_for_Agile_Humanoid_Locomotion_on_Constrained_Footholds/Walk_the_PLANC__Physics-Guided_RL_for_Agile_Humanoid_Locomotion_on_Constrained_Footholds.html)
