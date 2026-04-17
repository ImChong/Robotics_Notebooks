---
type: entity
---

# MuJoCo

**MuJoCo（Multi-Joint dynamics with Contact）** 是机器人与控制领域最经典的物理引擎之一，现在由 **Google DeepMind** 维护并开源。

## 一句话定义

MuJoCo 是一个强调**刚体动力学、接触仿真、控制与优化友好性**的机器人物理引擎。

一句话说白了：

> 如果你在做机器人控制、强化学习、模仿学习、动作生成，MuJoCo 基本是你绕不过去的一套经典底座。

## 先说结论

这几个点最容易被写错，先说清楚：

- MuJoCo 现在是 **Google DeepMind 维护的开源项目**
- 官方仓库是 `google-deepmind/mujoco`
- **`mujoco-py` 已经 deprecated**，新项目应优先用官方 Python bindings
- MuJoCo 不只是“老牌仿真器”，它现在在机器人学习、动作控制、基准环境这条线上仍然很强

## 为什么重要

很多物理引擎都能“把机器人放进去跑起来”，但 MuJoCo 的特别之处在于：

- 对控制研究非常友好
- 刚体动力学与接触仿真表现稳定
- 可微 / 优化友好思路强
- 和 OpenAI Gym / Gymnasium / DM Control / RL 基准生态历史关系深
- 在 locomotion、manipulation、motion imitation 里长期是标配之一

它的重要性不是“它最炫”，而是：

> 它几乎是机器人控制和 RL 社区共同默认的一套语言环境。

## MuJoCo 在解决什么问题

### 1. 机器人动力学仿真
你给它：
- 机器人模型
- 环境几何
- 关节驱动
- 接触条件

它负责算：
- 状态怎么演化
- 接触力怎么产生
- 动作怎么影响系统

### 2. 强化学习训练环境底座
MuJoCo 最广泛的历史影响之一，是成为大量 RL benchmark 的底层仿真器。

例如：
- Hopper
- Walker2d
- HalfCheetah
- Humanoid
- Ant

这些环境几乎是 RL 论文里一代人的共同语言。

### 3. 控制与动作生成实验平台
如果你做：
- model-based control
- trajectory optimization
- motion retargeting
- locomotion
- manipulation

MuJoCo 都是非常常见的实验底座。

## MuJoCo 的典型特征

### 1. 接触与动力学表现强
MuJoCo 长期受欢迎，一个关键原因就是：
- 接触仿真稳定
- 机器人动力学计算效率高
- 对足式 / 人形 / 机械臂控制都比较顺手

### 2. XML 模型定义体系成熟
MuJoCo 使用 MJCF（XML）来定义：
- 刚体
- 关节
- actuator
- sensor
- 接触几何
- 场景结构

这让它在研究里非常可控，也很适合快速试验。

### 3. 和控制研究很贴
MuJoCo 的气质一直偏：
- 控制
- 优化
- dynamics-based planning
- RL benchmark

所以它不是“偏渲染”的仿真器，而是偏研究和控制。

## 现在该怎么用 MuJoCo

### 正确做法
新项目里应该优先：
- 使用 **官方 MuJoCo Python bindings**
- 使用当前维护中的官方仓库与文档

### 不推荐做法
- 不要再把 `mujoco-py` 当主线
- 不要沿着很多旧博客里那套安装流程走

`mujoco-py` 在历史上很重要，但现在已经是 deprecated 状态。

## MuJoCo 和 Isaac Gym / Isaac Lab 的关系

这两个很容易被放在一起比较，但它们不完全是同一种定位。

### MuJoCo 更像
- 经典控制 / RL 社区通用底座
- 研究实验环境
- benchmark 生态核心仿真器

### Isaac Gym / Isaac Lab 更像
- NVIDIA 系高并行 GPU RL 训练栈
- 更偏大规模并行训练、人形/足式大 rollout
- Isaac Lab 是当前 NVIDIA 新主线

一句话：
- **MuJoCo**：经典、稳定、研究友好
- **Isaac Lab**：当前 NVIDIA 官方主线、面向大规模 robot learning workflow

它们不是互相淘汰关系，而是不同生态里的核心底座。

见：[Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)

## MuJoCo 在人形机器人里的作用

### 1. 人形 locomotion baseline
很多 humanoid locomotion RL 和控制基准，首先会在 MuJoCo Humanoid 或自定义人形模型上验证。

### 2. 模仿学习与动作生成
MuJoCo 在：
- motion imitation
- retargeting
- humanoid motion prior
- skill learning

这几个方向里非常常见。

### 3. 控制与优化实验
MuJoCo 适合做：
- trajectory optimization
- MPC baseline
- system identification 仿真验证
- control policy benchmark

## 和当前项目主线的关系

### 和 Reinforcement Learning 的关系
MuJoCo 是大量 RL benchmark 和机器人训练环境的底层仿真器。

见：[Reinforcement Learning](../methods/reinforcement-learning.md)

### 和 Imitation Learning 的关系
在动作模仿、人形动作生成、技能迁移研究里，MuJoCo 是常见平台。

见：[Imitation Learning](../methods/imitation-learning.md)

### 和 Locomotion 的关系
MuJoCo 在足式 / 人形 locomotion 研究里长期是主流基准环境之一。

见：[Locomotion](../tasks/locomotion.md)

### 和 Trajectory Optimization / MPC 的关系
MuJoCo 不只是拿来做 model-free RL，也很适合控制和优化实验。

见：[Trajectory Optimization](../methods/trajectory-optimization.md)

见：[Model Predictive Control (MPC)](../methods/model-predictive-control.md)

## 常见误区

### 1. 以为 MuJoCo 只是老 RL benchmark 引擎
错。它今天仍然是控制、学习、动作生成的重要平台。

### 2. 以为 MuJoCo 只能做 model-free RL
错。它其实很适合控制与优化研究。

### 3. 还把 `mujoco-py` 当默认入口
这已经过时了，新用户应该直接用官方 bindings。

### 4. 以为选了 MuJoCo，sim2real 就自然更容易
仿真器只是底座。sim2real 成败还取决于状态估计、系统辨识、执行器建模、观测设计等。

## 推荐使用建议

### 如果你是初学者
- MuJoCo 非常值得学
- 先学会官方 bindings 和基本模型定义
- 用它跑通一个 locomotion 或 manipulation baseline

### 如果你是做 RL 研究
- MuJoCo 仍然是最值得掌握的 benchmark 平台之一
- 读老 RL 论文时基本都绕不开

### 如果你是做控制 / 优化研究
- MuJoCo 特别适合做 dynamics-based baseline
- 拿来做 MPC / trajectory optimization / system identification 验证很顺手

## 继续深挖入口

如果你想沿着仿真平台继续往下挖，建议从这里进入：

### Repo / 平台入口
- [Simulation](../../references/repos/simulation.md)
- [Locomotion Benchmarks](../../references/benchmarks/locomotion-benchmarks.md)

## 推荐继续阅读

- MuJoCo 官网：<https://mujoco.org/>
- 官方仓库：<https://github.com/google-deepmind/mujoco>
- DeepMind 开源说明：<https://deepmind.google/blog/open-sourcing-mujoco>
- `mujoco-py` 仓库（deprecated 说明）：<https://github.com/openai/mujoco-py>
- MuJoCo Playground：<https://playground.mujoco.org/>

## 参考来源

- [sources/papers/simulation_tools.md](../../sources/papers/simulation_tools.md) — ingest 档案（MuJoCo 2012 / Isaac Gym 2021 / Genesis 2024）
- Todorov et al., *MuJoCo: A physics engine for model-based control* (2012) — MuJoCo 原论文
- 官方网站：<https://mujoco.org/>
- 官方仓库：<https://github.com/google-deepmind/mujoco>

## 关联页面

- [Pinocchio](./pinocchio.md)
- [Isaac Gym / Isaac Lab](./isaac-gym-isaac-lab.md)
- [Simulation](../../references/repos/simulation.md)

## 一句话记忆

> MuJoCo 是 Google DeepMind 维护的开源机器人物理引擎，是控制、强化学习、模仿学习和动作生成研究里最经典也最耐用的实验底座之一。
