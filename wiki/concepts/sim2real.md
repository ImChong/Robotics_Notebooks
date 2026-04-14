---
type: concept
tags: [sim2real, rl, domain-randomization, deployment]
status: complete
---

# Sim2Real

**Sim2Real**（仿真到现实迁移）：在仿真环境训练控制策略，然后部署到真实机器人上。

## 一句话定义

在仿真里学会，在现实中生效。

## 为什么重要

- 真实机器人训练成本高、速度慢、容易损坏
- 仿真可以并行加速、任意重置、无硬件损耗
- 但仿真和现实有 domain gap，必须解决迁移问题

## 核心问题：Domain Gap

仿真和现实的主要差异：

- **物理参数差异**：质量、摩擦力、延迟等参数不准
- **传感器差异**：相机噪声、IMU 漂移、触觉反馈
- **动作执行差异**：电机响应延迟、控制频率限制
- **视觉差异**：纹理、光照、背景

## 主要方法

### 1. Domain Randomization
在仿真中随机化物理参数，强制策略适应多样化环境。

代表工作：Tobio et al. 2018, "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"

### 2. System Identification
精确测量真实机器人参数，减少仿真-现实差距。

### 3. Domain Adaptation
用视觉/感知层面的 domain adaptation 减少感知差异。

### 4. Curriculum Learning
从简单环境逐步过渡到复杂/真实环境。

### 5. Privileged Information
训练时用额外信息（如 true state），推理时只用可观测信息。

## 常见误区

- **以为仿真越逼真越好**：太精确的仿真不一定更好，domain randomization 可能更 robust
- **忽略动作延迟**：仿真中动作瞬时执行，现实中有延迟
- **只看 reward 不看安全性**：sim2real 部署初期容易损坏硬件

## 在人形机器人中的应用

人形机器人 sim2real 的特殊挑战：

- 高维状态空间（30+ 自由度）
- 接触力难以精确建模
- 视觉感知差异大
- 足式接触的不确定性

典型 pipeline：

```
仿真训练 → 域随机化 → 零样本迁移 → 真实机器人部署 → 在线微调（可选）
```

## 参考来源

- Tobin et al. 2017, *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* — domain randomization 奠基论文
- [Sim2Real 论文导航](../../references/papers/sim2real.md) — 论文集合
- [Deployment-Ready RL: Pitfalls, Lessons, and Best Practices](https://thehumanoid.ai/deployment-ready-rl-pitfalls-lessons-and-best-practices/) — 工程实践

## 关联页面

- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Locomotion](../tasks/locomotion.md)
- [System Identification](./system-identification.md)（减少物理参数和执行器模型的 sim2real gap）

## 继续深挖入口

如果你想沿着 sim2real 继续往下挖，建议从这里进入：

### 论文入口
- [Sim2Real 论文导航](../../references/papers/sim2real.md)

### 仿真 / 平台入口
- [Simulation](../../references/repos/simulation.md)
- [RL Frameworks](../../references/repos/rl-frameworks.md)

## 推荐继续阅读

- [Deployment-Ready RL: Pitfalls, Lessons, and Best Practices](https://thehumanoid.ai/deployment-ready-rl-pitfalls-lessons-and-best-practices/)
- [Sim2Real Actuator Gap Estimator](https://github.com/isaac-sim2real/sage)
