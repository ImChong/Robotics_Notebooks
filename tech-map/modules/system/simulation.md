---
type: tech_map_node
status: complete
topic: [simulation, sim2real]
summary: "仿真（系统集成层）技术栈模块：定位仿真在策略训练、控制器验证、Sim2Sim 与系统辨识之间的中介桥梁角色，并区分于仿真项目清单。"
sources:
  - ../../../sources/papers/simulation_tools.md
  - ../../../sources/papers/sim2real.md
  - ../../../sources/repos/isaac_lab.md
updated: 2026-11-24
---

# 仿真（系统集成层）

仿真是控制、学习与部署之间的中介层：策略训练、控制器调试与场景复现多在仿真中完成，真机落地前通常还要经过 [Sim2Real](../../../wiki/concepts/sim2real.md)、[System Identification](../../../wiki/concepts/system-identification.md) 等桥接。

## 在全栈中的位置

```mermaid
flowchart LR
    MOD["动力学与执行器参数"] --> SIM["并行训练仿真（Isaac Lab / Isaac Gym）"]
    SIM --> S2S["高保真 Sim2Sim 验证（MuJoCo / Gazebo）"]
    S2S --> SYSID["系统辨识与域随机化对齐"]
    SYSID --> INT["ROS 2 / 真机部署闭环"]
```

## 和「仿真平台索引」的区别

| 页面 | 职责 |
|------|------|
| **本页（技术栈模块）** | 仿真在全栈中的位置、与上下游模块的关系 |
| **[仿真平台与工具链](../../../references/repos/simulation.md)** | 按用途整理的 MuJoCo / Isaac / Gazebo 等项目入口 |

## 上下游模块与站内入口

- **上游建模与训练**：[动力学](../robotics/dynamics.md) · [人形强化学习](../rl/humanoid-rl.md) · [全栈技术域总览](../../overview.md)
- **下游集成与部署**：[ROS 2（系统集成层）](ros2.md) · [部署（上真机）](deployment.md)
- **评测与选型参考**：[仿真评测基础设施](../../../wiki/concepts/simulation-evaluation-infrastructure.md) · [仿真器选型指南](../../../wiki/queries/simulator-selection-guide.md)

## 参考来源与收录边界

- [sources/papers/simulation_tools.md](../../../sources/papers/simulation_tools.md) — 主流物理仿真引擎对照归档
- [sources/papers/sim2real.md](../../../sources/papers/sim2real.md) — 仿真到现实迁移文献归档
- [sources/repos/isaac_lab.md](../../../sources/repos/isaac_lab.md) — Isaac Lab 并行仿真底座归档
- 具体各仿真平台的安装与源码入口见 [仿真平台与工具链](../../../references/repos/simulation.md)。

## 关联页面

- [仿真平台与工具链](../../../references/repos/simulation.md)
- [Sim2Real](../../../wiki/concepts/sim2real.md)
- [Domain Randomization](../../../wiki/concepts/domain-randomization.md)
- [仿真评测基础设施](../../../wiki/concepts/simulation-evaluation-infrastructure.md)
- [运动控制主路线](../../../roadmap/motion-control.md) — 控制主干的学习先后顺序
