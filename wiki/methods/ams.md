---
type: method
tags: [rl, sampling, data-generation, humanoid, physics-feasibility]
status: complete
updated: 2026-04-27
related:
  - ./imitation-learning.md
  - ./beyondmimic.md
  - ../concepts/reward-design.md
sources:
  - ../../sources/papers/motion_control_projects.md
summary: "AMS (Adaptive Motion Synthesis) 提出了一种物理可行性过滤与混合奖励机制，通过在极端动作处引入先验来提升复杂平衡任务的训练成功率。"
---

# AMS: 物理可行性过滤与混合奖励

在人形机器人动作合成与学习中，很多参考动捕数据（MoCap）由于人体与机器人的动力学差异，直接模仿会导致物理不可行。**AMS** 提供了一套系统化的数据清洗与奖励设计方法。

## 核心挑战：物理不一致性

人体的质量分布、接触面积和平衡策略与金属构成的机器人截然不同。直接将人类“金鸡独立”或“大跨步”的轨迹映射到机器人上，往往会导致机器人因质心 (CoM) 偏移而不可避免地摔倒。

## 技术方案

### 1. 物理可行性过滤 (Physics Feasibility Filtering)
AMS 并不信任所有的参考动作。它通过两阶段过滤确保进入训练集的数据是“能站稳”的：
- **第一阶段：运动学修正**。确保关节角度满足限位，姿态自然。
- **第二阶段：平衡约束惩罚**。强制要求参考运动在整个周期内的**质心投影 (CoM Projection)** 必须始终位于支撑足形成的支撑多边形内。
- **结果**：剔除所有在物理上注定会失稳的数据片段。

### 2. 混合奖励机制 (Hybrid Reward Mechanism)
AMS 提出了一个极具工程参考价值的原则：**不要在简单动作上过度约束**。

- **通用奖励**：涵盖关节误差、末端位姿、速度匹配等基础项。
- **平衡先验奖励**：**按需激活**。
    - 仅在“极端平衡动作”、“稳定性容差小”或“历史成功率低”的片段上，激活质心约束、脚底滑动惩罚等强力约束。
    - 作用：既保留了机器人在简单动作中的运动灵活性，又保证了在难动作下的稳定性。

## 主要技术路线

| 环节 | 实现方案 | 作用 |
|------|---------|------|
| **数据筛选** | 质心投影 (CoM) 多边形过滤 | 剔除物理上不可行的参考动作片段 |
| **奖励机制** | 混合/条件激活先验奖励 | 只在极端平衡需求下施加约束，保持灵活性 |
| **管线构建** | 运动学修正 + 物理一致性验证 | 自动生成大规模高质量合成平衡数据 |

## 工程模式：数据生成管线

AMS 实际上定义了一个自动化的合成数据生成流程：
1. **随机采样**：随机采样摆动足位姿、骨盆高度和初始姿态。
2. **轨迹插值**：构造连续的参考运动轨迹。
3. **优化过滤**：通过上述的平衡约束进行物理校验。
4. **高质量池**：形成一个物理可行的、多样化的专家动作数据库。

## 参考来源

- [sources/papers/motion_control_projects.md](../../sources/papers/motion_control_projects.md) — 飞书公开文档《开源运动控制项目》总结。
- [AMS 项目主页 (OpenDriveLab)](https://opendrivelab.com/AMS/)

## 关联页面

- [Reward Design](../concepts/reward-design.md) — AMS 是“条件奖励（Conditional Reward）”设计的典型案例。
- [Imitation Learning](./imitation-learning.md)
- [BeyondMimic](./beyondmimic.md) — BeyondMimic 强调自适应采样，AMS 强调物理过滤。
