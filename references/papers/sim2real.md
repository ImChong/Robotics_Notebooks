# Sim2Real

聚焦域随机化、系统辨识、鲁棒训练、部署经验与真实机器人迁移相关论文。

## 关注问题

- 如何弥合仿真和现实的物理参数差异？
- 如何通过随机化（DR）提升策略的鲁棒性？
- 如何在部署时实时感知并适应环境变化？
- 如何建立具有物理一致性的状态估计器？

## 代表性论文

### 域随机化 (Domain Randomization)

- **Tobin et al. (2017)** — *Domain Randomization for Transferring Deep Neural Networks*. 提出了视觉与物理参数随机化的基础范式。
- **Peng et al. (2018)** — *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*. 动力学随机化在腿式机器人控制中的应用。

### 在线自适应

- **RMA (Kumar et al., 2021)** — *Rapid Motor Adaptation for Legged Robots*. 提出了通过环境编码器实现快速在线自适应。

### 鲁棒状态估计

- **InEKF (Barrau & Bonnabel, 2017)** — *The Invariant Extended Kalman Filter as a Stable Observer*. 提供了基于李群的一致性状态估计理论。

## 关联页面

- [Sim2Real (Concept)](../../wiki/concepts/sim2real.md)
- [Domain Randomization (Concept)](../../wiki/concepts/domain-randomization.md)
- [System Identification (Concept)](../../wiki/concepts/system-identification.md)
- [Privileged Training (Concept)](../../wiki/concepts/privileged-training.md)
- [State Estimation (Concept)](../../wiki/concepts/state-estimation.md)
