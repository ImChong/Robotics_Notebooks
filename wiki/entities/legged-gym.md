---
type: entity
---

# legged_gym

**legged_gym** 是 ETH Zurich RSL（Robotic Systems Lab）开源的足式机器人强化学习训练框架，建立在 **Isaac Gym** 之上。

## 一句话定义

如果说 Isaac Gym 提供的是高并行 GPU 仿真底座，那 **legged_gym** 提供的就是：

> 一套已经帮你把足式机器人 RL 训练流程、reward 设计、domain randomization 和 sim2real 工程习惯打包好的经典训练栈。

## 为什么重要

很多人第一次真正把“机器人 RL 训练”跑起来，不是从零自己写环境，而是直接从 `legged_gym` 这种现成训练框架开始。

它重要的原因不是“它最先进”，而是：

- 它是足式机器人 RL 里最经典的开源基线之一
- 大量四足 / locomotion 项目都受它影响
- 它把 reward、observation、randomization、训练 loop 这些最容易踩坑的部分整理成了可复用工程模板
- 它是理解“机器人 RL 训练工程化长什么样”的最好窗口之一

一句话：

> `legged_gym` 的价值，不只是跑一个四足策略，而是让你看见足式机器人 RL 工程栈是怎么组织起来的。

## 它到底是什么

### 1. 不是物理引擎
`legged_gym` 本身不是 MuJoCo、也不是 Isaac Gym 那样的物理引擎。

它更像：
- 一个训练框架
- 一套环境定义和任务模板
- 一套机器人 RL 的工程实践集合

底层物理仿真还是依赖 Isaac Gym。

### 2. 它更像“足式 RL 训练工作台”
它帮你组织：
- 机器人模型加载
- observation space 定义
- action space 定义
- reward 设计
- command sampling
- domain randomization
- curriculum
- 训练脚本与超参数管理

所以它在研究和复现里非常高效。

## 为什么它在机器人 RL 里很有代表性

### 1. 让 RL 训练更像工程流程，而不是论文伪代码
很多论文只说：
- 训练 PPO
- 做 DR
- 调一下 reward

但真正落到工程上，问题是：
- reward 怎么拆项
- command 怎么采样
- observation 哪些该进、哪些不该进
- DR 怎么加在动力学、摩擦、质量、延迟上
- curriculum 怎么设计

`legged_gym` 把这些东西写成了具体可跑的结构。

### 2. 足式机器人社区的公共语境
只要你看四足 / humanoid locomotion RL 的代码生态，很多人都绕不开：
- ETH 系路线
- ANYmal / legged robotics 研究
- PPO + DR + privileged information 这套经验范式

`legged_gym` 正是这条路线最有代表性的开源实现之一。

### 3. 它把 sim2real 思路工程化了
`legged_gym` 最大的一个历史贡献是：
- 不只是训出一个策略
- 而是把 **domain randomization、teacher-student、privileged training、鲁棒 locomotion** 这些思路放进训练框架

这让它和纯 benchmark 环境不一样。

## 它在解决什么问题

### 1. 足式 locomotion 的训练问题
最典型：
- 四足机器人站立、行走、跑动
- terrain adaptation
- robust locomotion

### 2. sim2real 前的仿真训练问题
你可以用它：
- 训练 locomotion policy
- 加大范围 domain randomization
- 提升策略对扰动和模型误差的鲁棒性

### 3. RL 工程组织问题
它还在解决一个更实际的问题：

> 当你不是只想跑一个 demo，而是想系统地管理 reward、环境、训练参数、模型输出、评估流程时，该怎么组织代码。

## legged_gym 的典型结构价值

### 1. Reward 设计模板
足式机器人 RL 最大的坑之一是 reward。

`legged_gym` 让你看到一个典型 locomotion reward 会怎么拆：
- 速度跟踪
- 姿态稳定
- 能量惩罚
- 平滑项
- 足端相关项

这对初学者非常重要。

### 2. Observation 设计模板
它让你看到：
- base velocity
- projected gravity
- joint position / velocity
- previous action
- command

这些变量为什么会被放进 observation。

### 3. Domain Randomization 模板
这是它最值得学的一块之一：
- 摩擦随机化
- 质量随机化
- 推力扰动
- 控制延迟近似
- 观测噪声

这些都不是“为了学术好看”，而是为了 sim2real 更稳。

### 4. Terrain 与 Curriculum 设计
足式 locomotion 的难点不是平地，而是：
- 崎岖地形
- 斜坡
- 台阶
- 随机障碍

`legged_gym` 在 terrain generation 和 curriculum 组织上很有代表性。

## 它和 Isaac Gym / Isaac Lab 的关系

### 和 Isaac Gym 的关系
`legged_gym` 是典型的 **Isaac Gym 生态项目**。

也就是说：
- 它的历史价值很高
- 但技术语境偏旧一代 NVIDIA robot RL 栈

见：[Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)

### 和 Isaac Lab 的关系
现在如果做新项目：
- 你仍然很值得学习 `legged_gym` 的 reward、DR、训练组织思路
- 但未必要继续把它当成长期主训练主线
- 更合理的做法是：**学 `legged_gym` 的工程经验，实验平台优先看 Isaac Lab**

一句话：

> `legged_gym` 更像旧经典工程模板，Isaac Lab 更像当前官方主线平台。

## 它和当前项目主线的关系

### 和 Reinforcement Learning 的关系
它是机器人 RL 在足式 locomotion 上最典型的工程实现之一。

见：[Reinforcement Learning](../methods/reinforcement-learning.md)

### 和 Locomotion 的关系
`legged_gym` 本质上就是把 locomotion 任务组织成可训练环境。

见：[Locomotion](../tasks/locomotion.md)

### 和 Sim2Real 的关系
`legged_gym` 很值得学的一点，是它把 sim2real 里的 DR 和鲁棒训练写成了代码结构，而不是停留在概念层。

见：[Sim2Real](../concepts/sim2real.md)

### 和 Domain Randomization 的关系
如果你想看 DR 在机器人里是怎么真正落地的，`legged_gym` 是非常典型的例子。

见：[Domain Randomization](../concepts/domain-randomization.md)

## 常见误区

### 1. 以为 `legged_gym` 是物理引擎
不是，它是训练框架，底层依赖 Isaac Gym。

### 2. 以为 `legged_gym` 过时了就完全没必要看
错。它在工程组织和 RL 训练经验上仍然很有学习价值。

### 3. 以为把 `legged_gym` 跑通就等于搞懂了 locomotion RL
远远不够。你还得理解：
- reward 为什么这么设计
- observation 为什么这么选
- DR 为什么这么加
- 这些和真实机器人有什么关系

### 4. 以为四足经验能无脑迁移到人形
不行。很多经验能复用，但人形在平衡、接触切换、自由度耦合上更难。

## 推荐使用建议

### 如果你是初学者
- 强烈建议读它、跑它
- 尤其要看 reward / observation / DR / curriculum 这几块怎么组织

### 如果你做人形机器人 RL
- 可以把 `legged_gym` 当“四足 RL 工程范式样板”
- 真正做人形时，再结合 Isaac Lab、新的人形环境和更复杂控制结构

### 如果你做新项目
- 学它的工程思路
- 平台选型上优先考虑 Isaac Lab 或更新框架

## 推荐继续阅读

- 官方仓库：<https://github.com/leggedrobotics/legged_gym>
- 论文：Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*
- [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)

## 参考来源

- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning* (2021) — legged_gym 原论文
- 官方仓库：<https://github.com/leggedrobotics/legged_gym>
- **ingest 档案：** [sources/papers/simulation_tools.md](../../sources/papers/simulation_tools.md) — Isaac Gym 原论文（legged_gym 的基础平台）
- **ingest 档案：** [sources/papers/policy_optimization.md](../../sources/papers/policy_optimization.md) — Rudin et al. 2022 详细摘录

## 关联页面

- [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Sim2Real](../concepts/sim2real.md)

## 一句话记忆

> `legged_gym` 是足式机器人 RL 里最经典的开源训练框架之一，它不是物理引擎，而是一套把 locomotion、reward、domain randomization 和 sim2real 工程实践组织起来的旧经典训练栈。
