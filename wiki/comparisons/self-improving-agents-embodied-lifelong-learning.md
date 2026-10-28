---
type: comparison
title: 自我改进智能体与具身终身学习
updated: 2026-10-28
tags: [self-improvement, lifelong-learning, robot-learning, reinforcement-learning]
summary: "比较智能体代码、记忆、策略及仿真资产的更新闭环，区分真机交互、想象训练与人类示范。"
related:
  - ../concepts/recursive-self-improvement.md
  - ../concepts/data-flywheel.md
  - ../entities/paper-arcadia.md
  - ../entities/paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md
  - ../entities/paper-rcl-2401-16013-serl-a-software-suite-for-sample-efficient-robot.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/papers/self_improving_agents_embodied_lifelong_learning.md
  - ../../sources/papers/arcadia_arxiv_2512_00076.md
---

# 自我改进智能体与具身终身学习

**自我改进闭环**将交互经验或评测反馈用于更新智能体；具身场景还必须处理真实机器人采集、复位、硬件约束和部署验证。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RSI | Recursive Self-Improvement | 系统迭代改进自身的能力 |
| RL | Reinforcement Learning | 从环境反馈优化策略 |
| VLA | Vision-Language-Action | 将视觉、语言和动作关联的模型 |
| CQL | Conservative Q-Learning | 用保守价值估计学习策略的离线 RL 方法 |
| DGM | Darwin Gödel Machine | 以代码修改与经验评测探索智能体改进 |

## 先区分更新的对象

Fang 等以「输入—智能体—环境—优化器」组织反馈闭环；Ren 等进一步区分**基础模型参数**与**外围结构**（提示词、记忆、工具、控制逻辑）。两种分类有助于判断改进究竟发生在何处；系统拥有记忆或循环执行，不足以证明它能持续提高。

| 工作 | 更新对象与反馈 | 更新过程 | 实验或工程边界 |
|------|----------------|----------|----------------|
| [DGM](https://arxiv.org/abs/2505.22954) | 智能体代码；编程基准反馈 | 档案采样 → 改代码 → 评测 → 扩充档案 | 评测得分是经验验证，非改动有益的形式证明；不是机器人实验证据 |
| [RoboCat](https://arxiv.org/abs/2306.11706) | 通才策略与训练数据；任务轨迹 | 人类示范 → 任务微调 → 自采轨迹 → 通才再训练 | 新任务仍需示范；任务与机器人分布限制结论 |
| [RISE](../entities/paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md) | 机器人策略；世界模型想象轨迹及进度价值 | 动力学预测 → 价值评估 → 想象空间策略更新 → 真机验证 | 想象训练降低交互负担，但模型误差与真实部署差距仍需测量 |
| [SERL](../entities/paper-rcl-2401-16013-serl-a-software-suite-for-sample-efficient-robot.md) | 真机策略；奖励及交互数据 | off-policy RL 与奖励、复位、控制基础设施 | 是样本高效训练工具，不单独构成跨任务终身学习系统 |
| [Arcadia](../entities/paper-arcadia.md) | 场景资产与共享表示；部署反馈 | 采集 → 场景生成 → 共享学习 → sim-from-real 反馈 | 部分代码开放；完整闭环和真机性能须按现有实体页的边界解读 |
| [AllDayNav](https://arxiv.org/abs/2606.10927) | 多模态记忆和导航策略；生成目标与奖励 | 记忆累积 → 自生成指令和目标 → CQL 策略更新 | 项目页接近 100% 指特定仿真场景；截至 2026-10-28 未列代码下载链接 |

## 面向机器人的使用判断

1. **先定义反馈来源。** 成功判定、奖励、人工接管和环境复位都需要工程实现；仅保存数据不产生可靠监督。
2. **选择训练环境。** 真机 RL（SERL）得到直接交互，但要计入硬件、复位和安全成本；RISE 用世界模型生成训练轨迹，需用独立真机试验检查模型偏差。
3. **区分适应和累积。** RoboCat 的新任务需要示范且要再训练；AllDayNav 同时更新记忆和策略；Arcadia 还将场景资产写回。它们改变的系统部件不同。
4. **保留基线和旧任务。** 比较更新前后同条件成功率、轨迹效率、样本及人力成本，并复测旧任务，才能发现遗忘或性能回退；单次新任务提升不能证明无限持续改进。
5. **隔离候选版本。** 借鉴 DGM 的版本档案与独立评测，在仿真、离线日志和受控真机分阶段验证后再部署；编程基准结果不能直接推断机器人安全性。

## 关联页面

- [递归自我改进](../concepts/recursive-self-improvement.md)：自我修改与评估的更一般视角。
- [具身数据飞轮](../concepts/data-flywheel.md)：数据采集、训练、部署与反馈。
- [强化学习](../methods/reinforcement-learning.md)：策略、奖励和交互基础。

## 参考来源

- [论文组归档：自我改进智能体与具身终身学习](../../sources/papers/self_improving_agents_embodied_lifelong_learning.md)
- [Arcadia 论文归档](../../sources/papers/arcadia_arxiv_2512_00076.md)
- [AllDayNav 项目页归档](../../sources/sites/alldaynav.md)

## 推荐继续阅读

- [Ren 等：Self-Improvements in Modern Agentic Systems](https://arxiv.org/abs/2607.13104)
- [Fang 等：A Comprehensive Survey of Self-Evolving AI Agents](https://arxiv.org/abs/2508.07407)
