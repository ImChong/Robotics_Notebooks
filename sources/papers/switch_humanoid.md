---
title: "Switch: Learning Agile Skills Switching for Humanoid Robots"
authors: ["Yuen-Fui Lau", "Qihan Zhao", "Yinhuai Wang", "Runyi Yu", "Hok Wai Tsui", "Qifeng Chen", "Ping Tan"]
date: 2026-04-30
url: "https://arxiv.org/abs/2604.14834"
arxiv: "2604.14834"
tags: ["skill-switching", "humanoid", "reinforcement-learning", "skill-graph", "unitree-g1", "agility"]
---

# Switch: 人形机器人敏捷技能切换框架

## 核心摘要
Switch 是由香港科技大学 (HKUST) 等团队提出的一种分层控制框架，旨在解决人形机器人在多样化敏捷技能（如从踢球切换到跳舞）之间的稳健、灵活切换问题。该方法通过构建增强的技能图（Skill Graph）并引入缓冲节点（Buffer Nodes），实现了 100% 的切换成功率。

## 解决的问题
1. **技能切换不稳**：现有方法在处理运动学差异较大的技能转换时，容易出现失稳或动作扭曲。
2. **固定路径依赖**：传统方法依赖预定义的切换轨迹，缺乏对实时扰动和跟踪误差的鲁棒性。
3. **足端交互质量**：许多模仿学习方法产生的动作在足地接触力学上不自然（如滑步）。

## 核心方法：Augmented Skill Graph & Buffer-aware RL

### 1. 增强技能图 (Augmented Skill Graph)
- **跨技能边 (Cross-skill Edges)**：在技能图的基础上，根据姿态和速度的运动学相似性，主动添加跨技能的连接边。
- **缓冲节点 (Buffer Nodes)**：在技能切换的空隙插入可探索的缓冲状态。这些节点不对应固定的参考帧，而是允许 RL 策略在其中寻找动态可行的过渡路径。

### 2. 缓冲感知模仿 (Buffer-aware Imitation)
- 策略在非缓冲节点跟踪参考轨迹，而在缓冲节点则最大化其到达下一个技能起始点的成功率，从而平衡了“模仿精度”与“切换可行性”。

### 3. 足地接触奖励 (Foot-Ground Contact Reward, FGR)
- 引入显式的接触一致性奖励，减少了人形机器人高动态动作中的足端滑移现象，提升了动作的真实感。

## 实验结果
- **成功率 (SSR)**：在 Unitree G1 平台上，Switch 在简单、中等和困难切换任务中均保持了 **100% 的成功率**，远超 baseline (GMT, Any2Track)。
- **鲁棒性**：能够承受高达 **500N** 的外部冲击，并通过在线搜索技能图自动规划“起身 (Get-up)”路径实现跌倒恢复。
- **真机验证**：在 **Unitree G1** 上实现了实时（50Hz）的技能切换展示。

## 在本项目中的角色
Switch 代表了运动模仿领域从“轨迹跟踪”向“基于图的可控性”演进的重要方向。它证明了将高层图搜索与底层 RL 策略结合，可以极大提升人形机器人的技能泛化能力。

---
## 参考资料
- [ArXiv (2604.14834)](https://arxiv.org/abs/2604.14834)
- [Project Page (HTML)](https://arxiv.org/html/2604.14834v1)
