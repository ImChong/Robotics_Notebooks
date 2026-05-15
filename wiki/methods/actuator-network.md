---
type: method
tags: [simulation, sim2real, hardware, control, deep-learning]
status: complete
updated: 2026-05-15
related:
  - ../concepts/sim2real.md
  - ../entities/anymal.md
  - ../concepts/system-identification.md
sources:
  - ../../sources/papers/system_identification.md
  - ../../sources/papers/locomotion_rl.md
summary: "执行器网络（Actuator Network）通过神经网络在仿真中拟合真实电机的非线性动力学、摩擦和延迟特性，是实现高精度足式机器人 Sim2Real 迁移的核心技术。"
---

# Actuator Network (执行器网络)

**执行器网络 (Actuator Network)** 是一种在机器人仿真中用于模拟物理驱动器（如电控伺服电机、SEA 驱动器）真实物理行为的深度学习模型。它是解决**足式机器人 Sim2Real 鸿沟**中“动力学不匹配”问题的利器。

## 为什么需要 Actuator Network？

在物理仿真器（如 MuJoCo 或 Isaac Gym）中，默认的关节驱动通常是理想化的 PD 控制：
$$ \tau = K_p (q_{target} - q) + K_d (\dot{q}_{target} - \dot{q}) $$
然而，真实的机器人电机存在以下复杂的非线性特性，而这些特性很难用简单的代数公式精确描述：
1. **反向电动势 (Back-EMF)**：高速转动时电机输出扭矩会下降。
2. **非线性摩擦**：谐波减速器带来的静摩擦与动摩擦。
3. **总线延迟**：从控制指令下发到电流产生作用的 2-10ms 不等时延。
4. **SEA 物理特性**：ANYmal 等机器人使用的串联弹性执行器中的弹簧动力学。

## 核心机制

执行器网络不再使用解析公式，而是训练一个小型神经网络来充当驱动器层：

- **输入**：当前位置误差 $(q_d - q)$、关节速度 $\dot{q}$、历史动作序列 $a_{t-k:t}$。
- **输出**：该时刻真实的驱动力矩 $\tau_{real}$。
- **训练数据**：通过真机实验采集，输入随机动作指令，测量真实产生的关节响应。

## 流程总览

```mermaid
flowchart LR
  subgraph offline["离线：辨识与训练"]
    R[真机悬空激励] --> D[(q, q_dot, tau, 指令)]
    D --> T[MLP / LSTM 拟合]
    T --> W[ActuatorNet 权重 / TorchScript]
  end
  subgraph sim["仿真步内闭环"]
    P[策略或 PD 目标] --> F[误差 / 速度 / 历史动作特征]
    W --> N[执行器网络前向]
    F --> N
    N --> Tau[tau_real]
    Tau --> Phy[物理仿真一步]
    Phy --> F
  end
```

## 主要技术路线

1. **数据采集**：让真机悬空，下发啁啾信号（Chirp Signal）或随机频率的 PD 目标，记录 $(q, \dot{q}, \tau)$。
2. **离线训练**：训练一个 MLP 或 LSTM 模型，使其能准确预测真机的力矩输出。
3. **仿真集成**：将训练好的网络（通常转为 TorchScript）嵌入到仿真器的控制循环中。
4. **策略训练**：在带有执行器网络的仿真环境中训练 RL 策略。

## 带来的价值

- **频率对齐**：网络能够自发学出系统的传输延迟特性。
- **功耗预测**：由于能够拟合真实力矩，RL 学出来的策略会更加节能。
- **零样本迁移**：极大提升了 Locomotion 策略在复杂地形上的稳健性，减少了在真机上二次调参的需求。

## 关联页面
- [Sim2Real (仿真到现实迁移)](../concepts/sim2real.md)
- [ANYmal 实体页](../entities/anymal.md) — 广泛使用执行器网络的代表
- [System Identification (系统辨识)](../concepts/system-identification.md)

## 参考来源

- Hwangbo et al. (2019). *Learning Agile and Dynamic Motor Skills for Legged Robots*（Science Robotics）— 提出 **ActuatorNet**：用神经网络从历史关节误差与力矩序列预测真实关节力矩，显著压缩 sim2real 中的执行器动力学误差；正式出版页：[Science Robotics (DOI)](https://www.science.org/doi/10.1126/scirobotics.aau5872)；开放获取预印本：[arXiv:1901.08652](https://arxiv.org/abs/1901.08652)。
- Hwangbo et al. (2018). *Sim-to-Real: Learning Agile Locomotion For Quadruped Robots*（RSS）— 同团队早期 **sim2real + 动力学随机化** 路线，为后续执行器级建模问题提供上下文；会议论文 PDF：[RSS 2018 p10](https://www.roboticsproceedings.org/rss14/p10.pdf)。
- [sources/papers/system_identification.md](../../sources/papers/system_identification.md) — ingest 档案（含 Hwangbo 2019 ActuatorNet 条目与摘要）。
- [sources/papers/locomotion_rl.md](../../sources/papers/locomotion_rl.md) — 足式 RL 论文索引（ANYmal 等应用背景）。
- Isaac Lab 执行器栈工程参考（理想/隐式 PD、数据驱动力矩路径等）：[`actuator_pd.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/actuators/actuator_pd.py)。
