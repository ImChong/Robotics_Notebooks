# sim2real

> 来源归档（ingest）

- **标题：** Sim2Real / Domain Randomization / Adaptation
- **类型：** paper
- **来源：** arXiv / conference / robotics venues
- **入库日期：** 2026-04-08
- **最后更新：** 2026-04-14
- **一句话说明：** 聚焦仿真到真实迁移的核心方法（随机化、辨识、在线适应），服务 sim2real 主线页面。

## 核心论文摘录（MVP）

### 1) Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World (Tobin et al., 2017)
- **链接：** <https://arxiv.org/abs/1703.06907>
- **核心贡献：** 提出系统化随机化策略，验证“广覆盖随机化”可提升真实部署鲁棒性。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Domain Randomization](../../wiki/concepts/domain-randomization.md)

### 2) Sim-to-Real Transfer of Robotic Control with Dynamics Randomization (Peng et al., 2018)
- **链接：** <https://arxiv.org/abs/1710.06537>
- **核心贡献：** 将动力学随机化用于策略迁移，成为 locomotion sim2real 常用基线。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 3) Rapid Motor Adaptation for Legged Robots (Kumar et al., 2021)
- **链接：** <https://arxiv.org/abs/2107.04034>
- **核心贡献：** 在线估计隐式环境变量并快速调整策略，是 sim2real 适应范式代表。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)

### 4) Invariant Extended Kalman Filter as a Stable Observer (Barrau & Bonnabel, 2017)
- **链接：** <https://arxiv.org/abs/1410.1465>
- **核心贡献：** 为 InEKF 在机器人状态估计中的稳定性与群结构一致性提供理论基础。
- **对 wiki 的映射：**
  - [EKF / InEKF](../../wiki/formalizations/ekf.md)
  - [State Estimation](../../wiki/concepts/state-estimation.md)

## 当前提炼状态

- [x] 已补域随机化 / RMA / InEKF 三条核心摘要
- [~] 后续补：按“参数随机化→辨识→在线适应”整理统一 pipeline 图
