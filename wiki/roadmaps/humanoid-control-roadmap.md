# Humanoid Control Roadmap

面向人形机器人运动控制算法工程师的学习研究路线。

## 适合谁

- 机械/控制/机器人背景，想转人形机器人运控
- 有一定编程基础（Python, C++）
- 了解基本线性代数和控制理论更好（但不是必须）

## 先修知识

### 核心必学
1. **Python 基础**：能读懂和改代码
2. **机器人学基础**：正逆运动学、正逆动力学概念（不需要特别深）
3. **强化学习基础**：理解 MDP、policy、value function 概念

### 推荐资源
- [斯坦福《机器人学导论》(B站)](https://www.bilibili.com/video/BV17T421k78T/)
- [Sutton & Barto RL Book](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [OpenAI Spinning Up](https://spinningup.openai.com)

## 阶段一：建模与控制（2-4 周）

### 目标
理解人形机器人的物理建模和基本控制方法。

### 内容
- 浮动基动力学模型
- 单刚体动力学模型
- 线性倒立摆（LIP）和 ZMP
- 全身控制框架（QPs / WBC）
- 质心动力学 + NMPC + WBC

### 推荐实践
- 用 Pinocchio 跑一个人形机器人正逆动力学
- 实现一个简单的 ZMP 行走控制器

### 推荐资源
- 《Robot Dynamics Lecture Notes》(Featherstones)
- [TSID](https://github.com/stack-of-tasks/tsid)
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio)

---

## 阶段二：仿真与训练（2-4 周）

### 目标
学会用仿真环境训练 RL/IL 策略。

### 内容
- IsaacGym / IsaacLab 使用
- MuJoCo / PyBullet 基本操作
- 强化学习基本训练流程（PPO 训练行走策略）
- 模仿学习基本流程（BC 训练）

### 推荐实践
- 用 IsaacGym + PPO 训练一个人形机器人行走策略
- 用 legged_gym 跑通四足行走实验

### 推荐资源
- [IsaacGym](https://docs.robotsfan.com/isaacgym/index.html)
- [Legged Gym](https://github.com/leggedrobotics/legged_gym)
- [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)

---

## 阶段三：Sim2Real（1-2 周）

### 目标
理解并实践仿真到真实的迁移。

### 内容
- Domain Randomization
- 动作空间/观测空间的处理
- 真实机器人部署流程
- 在线微调方法

### 推荐实践
- 把仿真训练好的策略零样本迁移到真实机器人
- 调一调 domain randomization 参数看效果

### 推荐资源
- [Deployment-Ready RL](https://thehumanoid.ai/deployment-ready-rl-pitfalls-lessons-and-best-practices/)
- [Sim2Real Gap Estimator (SAGE)](https://github.com/isaac-sim2real/sage)

---

## 阶段四：进阶专题（持续）

根据研究方向选择：

### A. 足式运动
- 跑、跳、楼梯、崎岖地形
- 关键词：CPI, NMPC, WBC, RL + sim2real

### B. 全身操作（loco-manipulation）
- 边走边操作、多手操作
- 关键词：WBC, VLA, end-effector control

### C. 视觉导航
- 感知+运动整合
- 关键词：perception, terrain mapping, learning-based navigation

### D. 模仿学习与技能库
- MoCap → Retarget → 技能嵌入
- 关键词：ASE, CALM, Motion Encoder, BFM

---

## 常见卡点

### 1. 仿真环境和真实机器人差太多
解决思路：
- 增加 domain randomization 范围
- 用 privileged information 训练
- 考虑在线自适应

### 2. RL 训练不稳定
解决思路：
- 从好的 initialization 开始（IL + RL）
- 看 reward curve 和 policy entropy
- 用 PPO 这种相对稳定的算法起步

### 3. 不知道自己模型对不对
解决思路：
- 先用 WBC 做 baseline
- 对比 RL 和 WBC 的结果

---

## 推荐论文路线（按时间）

1. PPO (Schulman 2017) — RL 基础
2. AMP (Peng 2021) — 对抗模仿学习
3. ASE (Peng 2022) — 对抗技能嵌入
4. CALM (Tessler 2023) — latent 方向控制
5. LessMimic / OmniXtreme / ULTRA (2024-2025) — 新进展

---

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Sim2Real](../concepts/sim2real.md)
- [Locomotion](../tasks/locomotion.md)
