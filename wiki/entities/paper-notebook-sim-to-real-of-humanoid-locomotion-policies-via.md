---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2504.06585"
related:
  - ../overview/paper-notebook-category-10-sim-to-real.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_sim-to-real-humanoid-locomotion-via-joint-torque.md
summary: "把策略从仿真搬到真机失败，根子在现实差（reality gap）——仿真没建模的非线性执行器动力学、关节柔顺、接触顺应等。主流做法是域随机化（DR）：把摩擦、质量、电机常数等十几个参数在区间里乱抽。但 DR 的表达力被「参数化」框死——它只能在预先选定的参数维度上抖动，抖不出那些状态相关、非参数化的复杂偏差。本文换个空间下手：在关节力矩上直接加一个状态相关的扰动项 τ_φ(s)，由一张随机初始化、训练全程不更新的小 MLP 生成，每个 episode 重抽一次权重。这样注入的扰动天然依赖机器人当前状态（姿态、速度、接触力……），能逼真模拟「在某些位形下执行器更软、某些接触下力更偏」这类 DR 给不出的差异，从而把策略训得对没见过的现实差更鲁棒。仿真里遇到未见的执行器刚度和软地面，DR/ERFI 全军覆没而本法照走；真机 TOCABI 上 3/3 成功步行，DR 2/3、ERFI 0/3。"
---

# Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection

**Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：10_Sim-to-Real），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

把策略从仿真搬到真机失败，根子在现实差（reality gap）——仿真没建模的非线性执行器动力学、关节柔顺、接触顺应等。主流做法是域随机化（DR）：把摩擦、质量、电机常数等十几个参数在区间里乱抽。但 DR 的表达力被「参数化」框死——它只能在预先选定的参数维度上抖动，抖不出那些状态相关、非参数化的复杂偏差。本文换个空间下手：在关节力矩上直接加一个状态相关的扰动项 τ_φ(s)，由一张随机初始化、训练全程不更新的小 MLP 生成，每个 episode 重抽一次权重。这样注入的扰动天然依赖机器人当前状态（姿态、速度、接触力……），能逼真模拟「在某些位形下执行器更软、某些接触下力更偏」这类 DR 给不出的差异，从而把策略训得对没见过的现实差更鲁棒。仿真里遇到未见的执行器刚度和软地面，DR/ERFI 全军覆没而本法照走；真机 TOCABI 上 3/3 成功步行，DR 2/3、ERFI 0/3。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| DR | Domain Randomization | 域随机化：训练时随机抖动仿真物理参数以覆盖现实差 |
| ERFI | Enhanced Random Force Injection | 增强随机力注入，给执行器加随机力/力矩偏置的对照基线 |
| PPO | Proximal Policy Optimization | 近端策略优化，本文 RL 主干 |
| AMP | Adversarial Motion Prior | 对抗式动作先验，用判别器约束动作像参考运动 |
| MLP | Multi-Layer Perceptron | 多层感知机，扰动网络 τ_φ 的结构 |
| Reality Gap | 现实差 | 仿真与真机之间未建模的动力学/接触/执行器差异 |

## 为什么重要

- **超越参数化 DR**：给「DR 调不出鲁棒性」的工程困境一条新思路：换扰动空间，而非继续堆参数区间
- **状态相关性是关键**：比 ERFI 的状态无关随机力更强，说明现实差的「结构」值得显式建模
- **零额外硬件/标定**：不需力矩传感器、不需真机系统辨识，纯训练时技巧，落地成本低
- **可与现有管线叠加**：τ_φ 注入与 PPO+AMP 正交，可加在大多数 RL 步行/全身控制训练上

## 解决什么问题

Sim-to-Real 的核心矛盾：**仿真训得好 ≠ 真机走得动**，因为仿真器没法精确建模真实执行器（高减速比齿轮的非线性、摩擦、延迟）和接触柔顺。两条已有路线各有短板：

1. **域随机化（DR）**：把质量、摩擦、电机常数等**预选参数**在区间里随机化。问题在于它**只能在参数化维度上抖**——现实差里那些**状态相关、非参数化**的成分（比如某个位形下齿轮回差骤增、某种接触下力矩明显偏移）根本不在它的表达空间里，覆盖不到。 2. **随机力注入（ERFI 等）**：给执行器加**随机但与状态无关**的力/力矩偏置。它打破了参数化的束缚，但偏置是「白噪声」式的，不随机器人状态变化，仍模拟不出结构化的状态相关偏差。

## 核心机制

1. **换空间**：把 Sim-to-Real 的扰动从「参数空间」搬到「关节力矩空间」，突破参数化域随机化的表达上限。
2. **状态相关随机扰动**：用随机初始化、不更新的 MLP 生成依赖状态的力矩偏移，模拟非线性执行器、接触柔顺等结构化现实差。
3. **零偏置先验**：静止无驱动 → 零扰动的设计，把「现实差随驱动增大」的物理直觉写进结构。
4. **真机验证**：在全尺寸人形 TOCABI 上零样本步行，鲁棒性显著优于 DR 与 ERFI。

方法拆解（深读笔记小节）：在「力矩空间」注入扰动，而不是在「参数空间」抖动；扰动网络 τ_φ：一张随机初始化、永不更新的小 MLP；与策略联合训练（PPO + AMP）；直觉：为什么这招比 DR 更广。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 10_Sim-to-Real |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Sim-to-Real_Humanoid_Locomotion_via_Joint_Torque_Space_Perturbation_Injection/Sim-to-Real_Humanoid_Locomotion_via_Joint_Torque_Space_Perturbation_Injection.html> |
| arXiv | <https://arxiv.org/abs/2504.06585> |
| 机构 | Woohyun Cha, Junhyeok Cha, Jaeyong Shin, Donghyeon Kim, Jaeheung Park（首尔国立大学 智能信息系 / AICT 水原 / 1X Technologies） |
| 发表 | 2025-04-09（arXiv v1） |
| 源码 | 论文未公开代码 ❌ |
| 笔记阅读日期 | 2026-06-17 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-10-sim-to-real](../overview/paper-notebook-category-10-sim-to-real.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_sim-to-real-humanoid-locomotion-via-joint-torque.md](../../sources/papers/humanoid_pnb_sim-to-real-humanoid-locomotion-via-joint-torque.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Sim-to-Real_Humanoid_Locomotion_via_Joint_Torque_Space_Perturbation_Injection/Sim-to-Real_Humanoid_Locomotion_via_Joint_Torque_Space_Perturbation_Injection.html>
- 论文：<https://arxiv.org/abs/2504.06585>

## 推荐继续阅读

- [机器人论文阅读笔记：Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Sim-to-Real_Humanoid_Locomotion_via_Joint_Torque_Space_Perturbation_Injection/Sim-to-Real_Humanoid_Locomotion_via_Joint_Torque_Space_Perturbation_Injection.html)
