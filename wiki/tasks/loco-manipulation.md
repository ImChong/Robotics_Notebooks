---
type: task
tags: [loco-manipulation, humanoid, whole-body, manipulation, locomotion]
status: complete
summary: "Loco-Manipulation 关注机器人边移动边操作的全身协调问题，是 locomotion 与 manipulation 的耦合任务。"
---

# Loco-Manipulation

**移动操作（Loco-Manipulation）**：机器人在运动（行走/移动）的同时执行操作任务（抓取/推动/交互），要求同时具备行走能力和上肢操作能力。

## 一句话定义

让机器人**边走边动手**——不是先停下来再操作，而是行走和操作同时发生、相互协调。

## 为什么重要

真实世界的很多任务，**不允许机器人停下来再操作**：
- 从移动的传送带上取物
- 在行走时打开门或按电梯
- 在非结构化环境中拿递物品
- 和人交互时保持跟随并执行任务

人形机器人的终极目标是在复杂环境中替代人类完成通用任务，loco-manipulation 是走向这一目标必须解决的核心能力。

它也是当前人形机器人最难的开放问题之一：
- **行走**和**操作**各自都很难
- 两者同时做，会在动力学、控制、感知层面产生强耦合

## 核心挑战

### 1. 全身动力学耦合
手臂的运动会影响质心位置和角动量，从而影响行走稳定性。
反过来，行走时的步态节律和地面力也会传导到手臂末端，影响操作精度。

这意味着：**独立优化行走和操作，合并起来通常是不对的。**

### 2. 高自由度全身协调
人形机器人 30+ 自由度：
- 下肢负责行走和平衡
- 上肢负责操作
- 躯干需要在两者之间协调
- 没有全身协调控制器，很难保证稳定性

### 3. 接触丰富（Contact-Rich）
操作本质上是**接触任务**：
- 手和物体的接触
- 脚和地面的接触
- 两类接触同时变化，需要同时管理

### 4. 感知与规划
- 需要同时感知环境（地形 + 操作对象）
- 需要规划行走路径 + 操作动作
- 实时性要求高

### 5. Sim2Real Gap 更严峻
比纯 locomotion 或纯 manipulation 更难做 sim2real：
- 两类 gap 叠加
- 物体交互的接触模型更难精确建模
- 视觉感知在运动中更不稳定

## 常见方法路线

### 路线 A：全身 WBC + MPC
- 用 centroidal dynamics MPC 做高层规划
- 用 TSID / WBC 统一处理腿部和手臂的力分配
- 优势：动力学一致，理论扎实
- 劣势：建模和调参复杂，难以泛化

### 路线 B：分层控制（Hierarchical）
- 上层：操作策略（给出末端轨迹/接触计划）
- 下层：locomotion controller（执行行走 + 接触力）
- 优势：模块化，可复用已有 locomotion controller
- 劣势：层间接口设计复杂，耦合处理不够优雅

### 路线 C：端到端 RL
- 直接用 RL 训练一个统一的全身策略
- 输入：视觉 + 状态；输出：全身关节动作
- 优势：自动发现耦合动作
- 劣势：训练极难，奖励设计复杂，sim2real 挑战大

### 路线 D：IL（模仿学习）+ 全身控制
- 用人类遥操作或 MoCap 数据收集全身动作演示
- 用 Diffusion Policy / ACT 等方法训练策略
- 结合 WBC 做执行
- 优势：数据驱动，可以捕获自然的人类协调模式

### 近期代表工作（2024-2026）

| 工作 | 路线 | 核心思路 |
|------|------|---------|
| ULTRA (UIUC, 2026) | RL + 统一控制 | 多模态统一 loco-manipulation 控制器 |
| Mobile ALOHA (Stanford, 2024) | 遥操作 + BC | 在移动底盘上做双手操作 |
| HumanoidBench (2024) | benchmark | 标准化评测 loco-manipulation 任务 |

## 评价指标

- **任务成功率**：在目标操作任务上的成功率
- **行走稳定性**：操作过程中摔倒频率、步态质量
- **操作精度**：末端定位精度、抓取成功率
- **泛化能力**：对新地形、新物体的适应能力
- **实时性**：控制频率能否满足实际部署要求

## 关联系统/方法

- [Locomotion](./locomotion.md)
- [Manipulation](./manipulation.md)
- [VLA](../methods/vla.md) — 移动操作中的语义任务条件与 action chunk 生成新路线
- [Teleoperation](./teleoperation.md) — 遥操作是 loco-manipulation 数据采集的主要方式
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md)
- [TSID](../concepts/tsid.md)
- [Diffusion Policy](../methods/diffusion-policy.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [ULTRA Survey](./ultra-survey.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md) — 接触丰富型操作是 loco-manip 上肢控制的核心子问题

## 参考来源

- Cheng et al., *Expressive Whole-Body Control for Humanoid Robots* (2024) — 全身运动控制与操作
- Fu et al., *Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation* (2024) — 移动双手操作代表
- [ULTRA survey](./ultra-survey.md) — 统一多模态 loco-manipulation 综述（2026）
- **ingest 档案：** [sources/papers/teleoperation.md](../../sources/papers/teleoperation.md) — ALOHA / OmniH2O / UMI 遥操作系统
- **ingest 档案：** [sources/papers/diffusion_and_gen.md](../../sources/papers/diffusion_and_gen.md) — ACT / Diffusion Policy / π₀（loco-manip 策略学习）
- **ingest 档案：** [sources/papers/gentlehumanoid_upper_body_compliance.md](../../sources/papers/gentlehumanoid_upper_body_compliance.md) — GentleHumanoid：上半身柔顺与 contact-rich human/object interaction
- [机器人论文阅读笔记：GentleHumanoid](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_Loco-Manipulation_and_WBC/GentleHumanoid__Learning_Upper-body_Compliance_for_Contact-rich_Human_and_Object/GentleHumanoid__Learning_Upper-body_Compliance_for_Contact-rich_Human_and_Object.html)

## 推荐继续阅读

- Cheng et al., [*Expressive Whole-Body Control*](https://arxiv.org/abs/2402.16796)
- Fu et al., [*Mobile ALOHA*](https://mobile-aloha.github.io/)
- Duan et al., [*Humanoid Locomotion as Next Token Prediction*](https://arxiv.org/abs/2402.19469) — 统一行走与操作的 token 预测路线

## 一句话记忆

> Loco-Manipulation 要机器人边走边动手，是行走和操作能力的叠加，也是比两者单独都难一个量级的全身协调控制问题，是当前人形机器人最前沿的开放挑战之一。
