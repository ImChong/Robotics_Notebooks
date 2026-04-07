# Robot Learning Overview

**机器人学习**：让机器人通过数据学会完成复杂任务的方法集合，核心是把“如何做”从人工编程转向从经验中学习。

## 为什么重要

传统机器人依赖精确建模和手工控制，缺点：

- 对复杂环境建模困难
- 手工调参成本高
- 泛化能力差

机器人学习的优势：

- 从数据中自动提取有效策略
- 能处理高维状态/动作空间
- 在仿真中训练，迁移到真实机器人

## 核心组成

### 1. 强化学习（RL）
从环境交互中学习最优策略。核心是 reward signal。

### 2. 模仿学习（IL）
从专家演示中学习策略。核心是 demonstration。

### 3. 预测模型
学习环境动态模型，用于 planning 或 model-based RL。

### 4. 感知-动作联合学习
端到端学习直接从 raw sensor 到 action 的映射。

## 和其他领域的关系

- **控制理论**：提供稳定性、收敛性理论支撑
- **优化**：RL 训练本质是优化问题
- **计算机视觉**：感知模块的基础
- **运动控制**：具体执行层面的基础

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Locomotion](../tasks/locomotion.md)
- [Whole-Body Control](../concepts/whole-body-control.md)

## 推荐继续阅读

- Sutton & Barto, *Reinforcement Learning: An Introduction*
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com)
