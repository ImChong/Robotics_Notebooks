---
type: method
tags: [imitation-learning, privileged-training, dagger, teacher-student, sim2real]
status: complete
updated: 2026-07-14
summary: "飞书 Know-How 模块：Teacher-Student 用仿真特权信息训练教师，再蒸馏或 DAgger 聚合数据训练可部署学生，缓解 BC 分布偏移与 sim2real 观测差距。"
related:
  - ./dagger.md
  - ../concepts/privileged-training.md
  - ./behavior-cloning.md
  - ./imitation-learning.md
  - ../concepts/sim2real.md
  - ../overview/humanoid-rl-motion-control-methods.md
sources:
  - ../../sources/papers/humanoid_motion_control_know_how.md
  - ../../sources/papers/privileged_training.md
---

# Teacher-Student 与 DAgger 训练

RoboParty 飞书 Know-How 将 **Teacher-Student 模型** 与 **DAgger（Dataset Aggregation）** 列为同一教学模块：在仿真中用**特权观测**训练强教师，再把能力迁移给仅具可部署传感器的**学生**；DAgger 进一步让学生访问自身诱导状态并由专家补标，缓解行为克隆的分布漂移。

## 一句话定义

教师「作弊」学会难技能，学生只带真机传感器跟师；DAgger 则让学生在自己会犯错的状态上向专家补课。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TS | Teacher-Student | 特权教师向可部署学生蒸馏 |
| DAgger | Dataset Aggregation | 迭代用当前策略采集状态并由专家标注 |
| BC | Behavior Cloning | 纯监督模仿，易分布偏移 |
| RL | Reinforcement Learning | 可与 TS 混合（RL 探索 + 专家修正） |
| Sim2Real | Simulation to Real | TS/DAgger 主要缩小观测与动力学 gap |
| IL | Imitation Learning | 本模块所属范式 |

## 为什么重要

- **人形部署常态**：仿真有完美速度、高度图、接触标志；真机没有。
- **飞书代码块定位**：与 DreamWaQ、PIE 等「单阶段非对称 AC」并列，代表**两阶段或迭代 IL** 路线。
- **与 BFM 多技能 TS** 区分：本页是**通用训练范式**；多动作 BFM 见 [teacher-student-multi-skill-bfm](./teacher-student-multi-skill-bfm.md)。

## 核心原理

**Teacher-Student（蒸馏）：**
1. 训练 $\pi_{\mathrm{teacher}}(a \mid s_{\mathrm{priv}})$
2. 收集 $(s_{\mathrm{deploy}}, a^*)$ 其中 $a^* \sim \pi_{\mathrm{teacher}}(\cdot \mid s_{\mathrm{priv}}(s_{\mathrm{deploy}}))$
3. 监督训练 $\pi_{\mathrm{student}}(a \mid s_{\mathrm{deploy}})$

**DAgger：**
- 每轮用 $\pi_i$ rollout → 专家标注访问状态 → 聚合数据集 → 重训 $\pi_{i+1}$（详见 [DAgger](./dagger.md)）。

## 主要技术路线

| 路线 | 代表链接 | 说明 |
|------|----------|------|
| 迭代 IL | [DAgger](./dagger.md) | 分布聚合纠偏 |
| 特权信息 | [Privileged Training](../concepts/privileged-training.md) | 仿真作弊观测 |
| Sim2Real | [Sim2Real](../concepts/sim2real.md) | 部署 gap 主线 |

## 工程实践

- 明确 **特权列表**（地形高度、真实摩擦、外力）与 **部署列表**（IMU、关节、可选深度）。
- 记录每轮 DAgger 的 **状态覆盖** 与失败模式，避免只在专家分布附近重复采样。
- 与单阶段非对称 AC 对比样本效率与调试成本（飞书强调两种路线并存）。

## 局限与风险

- **专家质量上限**：教师本身不强，学生无法超越。
- **仿真特权泄露**：若特权与部署观测存在确定性映射漏洞，可能导致过拟合作弊特征。
- **工程迭代成本**：DAgger 需专家在线或自动标注管线。

## 关联页面

- [DAgger](./dagger.md)、[Privileged Training](../concepts/privileged-training.md)
- [Teacher-Student 多技能 BFM](./teacher-student-multi-skill-bfm.md)
- [Know-How 技术地图](../overview/humanoid-motion-control-know-how-technology-map.md)

## 参考来源

- [humanoid_motion_control_know_how.md](../../sources/papers/humanoid_motion_control_know_how.md)
- [privileged_training.md](../../sources/papers/privileged_training.md)

## 推荐继续阅读

- Ross et al., *DAgger* (2011)
- [Sim2Real](../concepts/sim2real.md)
