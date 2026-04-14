---
type: concept
tags: [rl, sim2real, training, humanoid, policy-optimization]
status: complete
---

# Privileged Training（特权信息训练）

**特权训练**（Privileged Training / Teacher-Student Training）：训练阶段提供给策略额外的、在真实部署时无法获取的信息，再通过知识蒸馏将能力迁移给仅使用可观测信息的部署策略。

## 一句话定义

> 让老师策略用"作弊信息"学会技能，然后教会学生策略只用真实传感器也能做到。

---

## 为什么重要

人形/足式机器人的核心矛盾：

- **仿真里**：可以访问完整物理状态（地形高度、质量参数、摩擦系数、所有接触力……）
- **真实里**：只有 IMU、关节编码器、摄像头等有限传感器

直接只用可观测信息训练：样本效率低、策略难以收敛（状态不可完整观测）。

特权训练的价值：
1. **训练效率**：Teacher 策略用完整状态收敛快，Student 从 Teacher 蒸馏，比 Student 直接从头练快得多
2. **性能上界**：Teacher 给了 Student 一个可靠的行为模板，而不只是 reward 信号
3. **Sim2Real 关键**：避免策略依赖仿真才有的状态，天然对齐 sim2real 需求

---

## 核心机制

### Teacher-Student 两阶段流程

```
阶段 1：训练 Teacher 策略
  输入：完整特权状态 s_priv（包含仿真才有的信息）
  训练方法：标准 RL（PPO/SAC 等）
  目标：在仿真中最大化 reward，尽量学好

阶段 2：训练 Student 策略
  输入：仅可观测状态 s_obs（传感器数据）
  目标：模仿 Teacher 的行为（行为克隆 + DAgger 变体）
  损失：KL 散度 / 动作回归 / 特征对齐
```

### Asymmetric Actor-Critic（非对称 Actor-Critic）

单阶段变体：Actor 和 Critic 使用不同的状态空间。

- **Actor**：只使用可观测状态 $s_{obs}$（直接生成部署策略）
- **Critic**：使用完整特权状态 $s_{priv}$（更准确的价值估计，指导 Actor 更新）

$$L_{actor} = -\mathbb{E}[\log \pi_\theta(a|s_{obs}) \cdot A(s_{priv}, a)]$$

优点：单阶段即可，无需两步训练；Critic 不部署，可以用所有信息。

---

## 特权信息的类型

| 类别 | 示例 | 真实部署能否获取 |
|------|------|---------------|
| 地形信息 | 高度图、斜度、障碍位置 | 需要感知（激光/深度相机），有延迟 |
| 物理参数 | 质量、摩擦系数、关节刚度 | 基本不可知，RMA 用自适应估计 |
| 接触状态 | 精确接触力、接触点位置 | 需要力传感器（部分机器人无） |
| 对手/目标状态 | 物体精确位置/速度 | 视觉估计有噪声 |
| 全局状态 | 机器人世界坐标（GPS-like） | 通常无法精确获取 |

---

## 典型算法与实现

### RMA（Rapid Motor Adaptation，Kumar et al. 2021）

最具代表性的 Teacher-Student + Sim2Real 框架。

**阶段 1**：训练 Teacher（base policy）
- 输入：机器人状态 + 环境参数（摩擦、质量等特权信息）
- PPO 训练，学会适应各种参数

**阶段 2**：训练 Adaptation Module
- 输入：过去 $n$ 步的关节状态历史
- 目标：从历史轨迹中隐式估计 Teacher 用的特权信息
- 损失：预测 Teacher 所用的隐变量

部署时：Base policy + Adaptation Module，无需特权信息。

### Learning to Walk in Minutes（ETH Zurich）

- Teacher：完整地形信息 → PPO
- Student：只用 proprioception（本体感知）→ 行为克隆
- 关键发现：高度仅凭步态历史即可隐式估计

### AMP + Teacher-Student

- Teacher：用高质量参考动作 + 判别器训练
- Student：蒸馏 Teacher 策略到本体感知策略

---

## 和标准 Sim2Real 的关系

```
标准 Sim2Real 路线：
  Domain Randomization → 在仿真中训练鲁棒策略 → 直接部署

特权训练路线：
  Teacher（特权信息）→ 高效收敛 →
  Student 蒸馏 → 仅用可观测状态 →
  Domain Randomization + 鲁棒训练 → 部署
```

两者不互斥——特权训练常和域随机化结合：Teacher 在随机参数空间训练，Student 从 Teacher 蒸馏同时也在随机参数下训练。

---

## 常见优点

- 训练效率高：Teacher 利用完整状态快速收敛
- 部署简洁：Student 只需传感器输入，不依赖特权信息
- 自然解耦：仿真能力和真实部署能力分开优化
- 可组合：与域随机化、课程学习等方法无缝结合

## 常见局限

- **两阶段训练成本**：需要训练两个网络，调试工作量翻倍
- **Student 上界有限**：Student 只能逼近 Teacher，不能超越它
- **特权信息选择**：哪些信息作为特权信息需要领域知识判断
- **Adaptation 泛化性**：Adaptation Module 对分布外情况可能失效

---

## 参考来源

- [sources/papers/privileged_training.md](../../sources/papers/privileged_training.md) — ingest 档案（Kumar RMA 2021 / Lee Science Robotics 2020 / Ji 并发训练 2022）
- Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021) — 最经典的 Teacher-Student sim2real 实现
- Zhuang et al., *Robot Parkour Learning* (2023) — Teacher-Student + 视觉输入扩展
- Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain* (Science Robotics, 2020) — 非对称 Actor-Critic 在足式机器人上的应用
- Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning* (2018) — 非对称 AC 理论基础

---

## 关联页面

- [Sim2Real](./sim2real.md) — 特权训练是 sim2real 的核心技术之一，解决训练-部署感知差异
- [Reinforcement Learning](../methods/reinforcement-learning.md) — Teacher 阶段用标准 RL 训练
- [Imitation Learning](../methods/imitation-learning.md) — Student 阶段本质上是模仿 Teacher 的行为克隆
- [Domain Randomization](./domain-randomization.md) — 常与特权训练结合，增强策略鲁棒性
- [Loco-Manipulation](../tasks/loco-manipulation.md) — 复杂操作任务需要特权训练处理感知遮挡

## 一句话记忆

> 特权训练让策略在仿真里"作弊"学技能，再通过蒸馏让真实部署的策略继承这些技能——训练时用全知，部署时用感知。
