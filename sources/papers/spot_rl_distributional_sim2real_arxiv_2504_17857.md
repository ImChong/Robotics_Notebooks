# Spot 高性能 RL 与分布距离 Sim2Real 标定（arXiv:2504.17857）

> 论文来源归档（ingest）

- **标题：** High-Performance Reinforcement Learning on Spot: Optimizing Simulation Parameters with Distributional Measures
- **类型：** paper / quadruped / reinforcement-learning / sim2real / isaac-lab
- **arXiv：** <https://arxiv.org/abs/2504.17857> · PDF：<https://arxiv.org/pdf/2504.17857>
- **机构：** NVIDIA + Boston Dynamics（作者含 AJ Miller, Fangzhou Yu, Michael Brauckmann, Farbod Farshidian 等）
- **入库日期：** 2026-07-05
- **一句话说明：** 基于 **Spot RL Researcher Development Kit** 低层电机 API，在 **Isaac Lab** 训练、真机部署 **端到端 RL 步态**；用 **Wasserstein 距离 / MMD** 度量仿真—真机数据分布差异，并以 **CMA-ES** 优化难测仿真参数，实现 **>5.2 m/s**、多步态含 **腾空相** 的高性能 locomotion。

## 核心摘录（面向 wiki 编译）

### 1) 公开端到端 RL on Spot 硬件

- **要点：** 声称 **首个公开演示** 的 Spot **低层 API 端到端 RL 部署**；训练代码经 **NVIDIA Isaac Lab**、部署经 **Boston Dynamics** 渠道发布——打通 **研究 SDK → 仿真训练 → 真机力矩环** 闭环。
- **对 wiki 的映射：**
  - [`wiki/entities/paper-spot-rl-distributional-sim2real.md`](../../wiki/entities/paper-spot-rl-distributional-sim2real.md)
  - [`wiki/entities/isaac-gym-isaac-lab.md`](../../wiki/entities/isaac-gym-isaac-lab.md)
  - [`wiki/entities/boston-dynamics.md`](../../wiki/entities/boston-dynamics.md)

### 2) 分布距离 + CMA-ES 的 Sim 参数辨识

- **要点：** 采集真机与仿真 **状态—动作分布**，以 **Wasserstein / MMD** 作为 sim2real gap 评分；用 **协方差矩阵自适应进化策略（CMA-ES）** 搜索 **未知或难测** 的仿真参数（摩擦、电机模型等），优于纯手动 DR 调参。
- **对 wiki 的映射：**
  - [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)
  - [`wiki/concepts/domain-randomization.md`](../../wiki/concepts/domain-randomization.md)

### 3) 性能与鲁棒性声称

- **要点：** 部署策略可达 **>5.2 m/s**（约为出厂控制器最大速度 **三倍** 量级）、**湿滑面鲁棒**、**扰动抑制** 与 **含 flight phase 的多步态**；展示 RL 在 **Spot 硬件极限** 上相对经典 WBC/MPC 栈的潜力。
- **对 wiki 的映射：**
  - [`wiki/entities/paper-spot-rl-distributional-sim2real.md`](../../wiki/entities/paper-spot-rl-distributional-sim2real.md)
  - [`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md)

## 当前提炼状态

- [x] 摘要级摘录与 wiki 映射
- [ ] 与 Isaac Lab Spot 官方示例、BD RL SDK 文档保持版本脚注同步
