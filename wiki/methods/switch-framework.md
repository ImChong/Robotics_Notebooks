---
type: method
tags: [reinforcement-learning, skill-switching, humanoid, skill-graph, motion-imitation, unitree-g1]
status: complete
updated: 2026-04-30
related:
  - ./imitation-learning.md
  - ./any2track.md
  - ./ams.md
  - ../entities/unitree-g1.md
  - ../concepts/whole-body-control.md
sources:
  - ../../sources/papers/switch_humanoid.md
summary: "Switch 是一种分层人形机器人控制框架，通过「增强技能图」与「缓冲节点」设计，实现了多样化敏捷技能之间的 100% 成功切换与极强的扰动鲁棒性。"
---

# Switch: 敏捷技能切换框架

> **Query 产物**：本页由「人形机器人如何实现不同技能间的平滑切换」研究触发。

**Switch** 是由香港科技大学等机构提出的一种针对人形机器人的技能切换方案。它解决了传统运动模仿方法在处理非连续、大跨度动作转换时容易失稳的问题，通过将高层图规划与底层强化学习（RL）控制相结合，实现了极其稳健的动作衔接。

## 核心技术路线

### 1. 增强技能图 (Augmented Skill Graph)
Switch 的核心数据结构是一个高度互联的动作图：
- **节点**：动捕数据中的每一帧。
- **边**：除了原始视频中的时序边外，通过计算节点间的运动学相似度（姿态 $q$ 和速度 $\dot{q}$），自动添加**跨技能跳转边**。
- **缓冲节点 (Buffer Nodes)**：在原本不连续的技能之间插入“虚拟缓冲区”，这些区域不强制跟踪参考轨迹，而是给 RL 策略留出探索空间。

### 2. 缓冲感知强化学习 (Buffer-aware RL)
Switch 采用 [PPO](./ppo-vs-sac.md) 算法训练底层控制策略，其奖励函数具有自适应性：
- **在常规节点**：执行标准轨迹模仿。
- **在缓冲节点**：模仿权重降低，转而通过「目标可达性奖励」引导机器人向下一个技能的入口状态靠拢。

### 3. FGR (Foot-Ground Contact Reward)
为了解决 [Sim2Real](../concepts/sim2real.md) 中常见的滑步（Foot Skating）问题，Switch 引入了接触一致性正则项：
$$r_{FGR} = \exp(-\omega \|v_{foot} \cdot c_{foot}\|)$$
该奖励强制机器人在足端与地面接触（$c_{foot}=1$）时速度趋于零，显著提升了动作的物理真实感。

## 主要技术特色

| 特性 | 传统方法 (如 GMT/Any2Track) | Switch 框架 |
|------|---------------------------|-------------|
| **切换机制** | 依赖神经网络隐式学习过渡 | 显式图搜索 + 缓冲节点引导 |
| **切换成功率** | 困难任务下低至 < 10% | **100% (简单/中等/困难任务)** |
| **抗扰动能力** | 易崩溃，依赖 reset | 可承受 500N 冲击，自动规划起身路径 |
| **物理一致性** | 易出现滑步、穿模 | FGR 奖励保证高质量足地交互 |

## 跌倒恢复与在线规划

Switch 的高层调度器不仅仅是简单的顺序执行器：
- **实时重规划**：当检测到机器人由于外力偏离当前轨迹时，调度器会在技能图中实时搜索距离当前状态最近的节点。
- **闭环恢复**：如果机器人跌倒，调度器会连接到“起身”技能段，将机器人引导回稳定状态，无需人工干预。

## 关联页面

- [模仿学习 (Imitation Learning)](./imitation-learning.md) — Switch 的基础训练范式。
- [Any2Track & RGMT](./any2track.md) — 基准对比方法，侧重于通用跟踪。
- [AMS](./ams.md) — 侧重于物理可行性过滤。
- [Unitree G1](../entities/unitree-g1.md) — Switch 的实验验证平台。

## 参考来源

- [sources/papers/switch_humanoid.md](../../sources/papers/switch_humanoid.md)
- Lau et al., *Switch: Learning Agile Skills Switching for Humanoid Robots*, 2026.
