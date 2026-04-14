# Isaac Gym / Isaac Lab

**Isaac Gym** 和 **Isaac Lab** 是 NVIDIA 机器人强化学习生态里的两代核心框架。

## 一句话定义

- **Isaac Gym**：NVIDIA 早期的 GPU 加速机器人 RL 仿真框架，主打大规模并行训练。
- **Isaac Lab**：当前 NVIDIA 官方主推的机器人学习框架，建立在 Isaac Sim 之上，用于 robot learning、locomotion、manipulation 和 sim2real 研究。

一句话说白了：

> 如果你在看老论文和老开源项目，会不断遇到 Isaac Gym；如果你要搭现在的新实验栈，应该优先看 Isaac Lab。

## 先说结论

这是最容易把人带偏的地方：

- **Isaac Gym 现在已经是 deprecated / legacy software**
- **NVIDIA 官方建议迁移到 Isaac Lab**
- 但大量 2021–2024 的机器人 RL 项目、代码库、论文 baseline 仍然基于 Isaac Gym / IsaacGymEnvs

所以正确心态不是“Gym 已死不用看”，也不是“继续把 Gym 当主线”，而是：

> **理解 Isaac Gym 的历史地位，实际项目优先使用 Isaac Lab。**

## 为什么它们重要

在人形 / 足式机器人 RL 里，训练成本一直是大问题。

传统 CPU 仿真慢，机器人状态维度高，接触计算复杂，训练一个可用策略很耗时。

Isaac Gym 当年为什么火：
- PhysX 仿真 + GPU tensor API
- 大规模并行环境 rollout
- 很适合 PPO 这类 on-policy 训练
- 一下把“几千到几万个环境同时跑”这件事变得工程上可行

Isaac Lab 为什么重要：
- 它接住了 Isaac Gym 这条线
- 建立在更完整的 Isaac Sim 生态之上
- 官方持续维护、文档更系统、迁移路径更明确
- 对 robot learning / manipulation / locomotion 的支持更现代

## Isaac Gym 是什么

### 它解决什么问题

Isaac Gym 的核心价值在于：

- 快速并行仿真机器人环境
- 在 GPU 上直接处理观测、动作和物理状态
- 让 RL 训练速度足够快，适合足式 / 人形机器人训练

这套框架在很多机器人 RL 工作里几乎是默认训练底座。

### 它的典型特征

- GPU 加速物理仿真
- GPU tensor API
- 支持 URDF / MJCF 导入
- 支持位置、速度、力矩等传感器观测
- 支持 runtime domain randomization
- 和 `IsaacGymEnvs`、`legged_gym` 等代码库一起构成经典研究栈

### 它的问题

Isaac Gym 虽然强，但它现在有三个现实问题：

1. **官方已停止支持**
2. **文档和生态会越来越偏向迁移**
3. **新项目如果还把 Gym 当主线，后面维护成本会变高**

所以你现在看它，更像是在理解“旧主流基线”。

## Isaac Lab 是什么

### 它解决什么问题

Isaac Lab 不是单纯替代 Isaac Gym 的 API 包装，而是 NVIDIA 当前 robot learning 的官方主线框架。

它建立在 Isaac Sim 之上，目标是：
- 提供现代化 robot learning workflow
- 接住旧的 IsaacGymEnvs / OmniIsaacGymEnvs / Orbit 用户
- 支持训练、迁移、任务定义、环境注册、仿真管理

### 它的典型特征

- 建立在 **Isaac Sim** 上
- 支持强化学习、模仿学习、locomotion、manipulation
- 提供从旧框架迁移的官方文档
- 有更清晰的任务组织与环境注册方式
- 是 NVIDIA 现在推荐的主线

### 什么时候优先用 Isaac Lab

如果你：
- 正在搭建新的人形 / 足式 RL 项目
- 想用 NVIDIA 官方当前支持的方案
- 想减少以后迁移成本

那就直接优先 Isaac Lab。

## Isaac Gym 和 Isaac Lab 的关系

最容易混的点在这里。

### 不是简单的“新版本号关系”
它们不是“Isaac Gym 2.0 = Isaac Lab”。

更准确地说：
- Isaac Gym 是早期独立 GPU RL 仿真框架
- Isaac Lab 是后续基于 Isaac Sim 的新主线 robot learning 框架

### 研究语境里的关系

如果你在读论文或开源项目：
- 2021–2024 很多足式 / 人形 RL baseline 还在用 Isaac Gym
- 现在新项目越来越往 Isaac Lab 迁移

所以你需要能“读得懂 Gym 论文与代码”，但“新实验优先用 Lab”。

## Isaac Gym / Isaac Lab 在人形机器人里的作用

### 1. 人形 locomotion RL 训练底座
最典型用途：
- PPO 训练 humanoid locomotion
- domain randomization
- 大规模 parallel rollout

### 2. 操作与 loco-manipulation 研究
新一代框架（特别是 Isaac Lab）更适合接 manipulation 和复杂任务。

### 3. sim2real 的仿真前端
虽然 sim2real 不只取决于仿真器，但：
- 仿真器决定环境 fidelity
- 环境定义决定训练流程
- domain randomization 和 observation pipeline 也深受框架影响

## 常见搭配关系

### 经典旧栈

```text
Isaac Gym
  + IsaacGymEnvs
  + legged_gym
  + PPO / rl-games
```

### 当前推荐新栈

```text
Isaac Sim
  + Isaac Lab
  + 任务定义 / 环境注册
  + RL / IL / manipulation / locomotion
```

## 它和当前项目主线的关系

### 和 Reinforcement Learning 的关系
Isaac Gym / Isaac Lab 是 RL 训练的“基础设施层”，不是 RL 算法本身。

见：[Reinforcement Learning](../methods/reinforcement-learning.md)

### 和 Locomotion 的关系
在人形和足式 locomotion 研究里，它们常常是训练环境和 benchmark 平台。

见：[Locomotion](../tasks/locomotion.md)

### 和 Sim2Real 的关系
它们提供仿真训练和 domain randomization 的主要工作台，但 sim2real 成功与否还取决于状态估计、系统辨识、观测设计等。

见：[Sim2Real](../concepts/sim2real.md)

### 和 Domain Randomization 的关系
Isaac Gym 当年就因为易于做大规模随机化而很受欢迎；Isaac Lab 也延续了这条能力路线。

见：[Domain Randomization](../concepts/domain-randomization.md)

## 常见误区

### 1. 以为 Isaac Gym 还是官方主线
不是，它现在是 legacy。

### 2. 以为看老论文就没必要学 Isaac Lab
错。读老论文要懂 Gym，新实验要优先 Lab。

### 3. 以为换成 Isaac Lab，旧经验都作废
也不对。训练逻辑、任务构造、reward 设计、DR 思路很多是继承下来的。

### 4. 以为仿真器选对了，sim2real 就稳了
远远不够。状态估计、系统辨识、执行器建模、观测延迟同样关键。

## 推荐使用建议

### 如果你是初学者
- 先理解 Isaac Gym 在机器人 RL 里的历史地位
- 真正开始新项目时优先直接上 Isaac Lab

### 如果你是复现旧工作
- 大概率还要会看 Isaac Gym / IsaacGymEnvs / legged_gym
- 但最好顺手了解一下迁移思路

### 如果你是做新项目
- 直接优先 Isaac Lab
- 把 Gym 当成旧基线和历史语境

## 继续深挖入口

如果你想沿着仿真与训练平台继续往下挖，建议从这里进入：

### Repo / 平台入口
- [Simulation](../../references/repos/simulation.md)
- [RL Frameworks](../../references/repos/rl-frameworks.md)

## 推荐继续阅读

- NVIDIA Isaac Gym 页面：<https://developer.nvidia.com/isaac-gym>
- Isaac Lab 文档首页：<https://isaac-sim.github.io/IsaacLab/v2.1.0/>
- Isaac Lab 迁移指南：<https://isaac-sim.github.io/IsaacLab/v1.0.0/source/migration/index.html>

## 参考来源

- Makoviychuk et al., *Isaac Gym: High Performance GPU Based Physics Simulation For Robot Learning* (2021) — Isaac Gym 原论文
- 官方文档：<https://isaac-sim.github.io/IsaacLab/v2.1.0/>

## 关联页面

- [legged_gym](./legged-gym.md)
- [MuJoCo](./mujoco.md)
- [RL Frameworks](../../references/repos/rl-frameworks.md)

## 一句话记忆

> Isaac Gym 是旧一代高性能机器人 RL 仿真框架，Isaac Lab 是当前 NVIDIA 官方推荐的新主线。读旧工作要懂 Gym，做新项目优先 Lab。
