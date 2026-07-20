# Realizing Robotic Swimming with Unified Fluid-Robot Multiphysics

> 来源归档

- **标题：** Realizing Robotic Swimming with Unified Fluid-Robot Multiphysics
- **类型：** paper
- **出处：** 2026 · RSS 2026 Finalist · arXiv preprint
- **arXiv：** <https://arxiv.org/abs/2506.05012>
- **论文 HTML：** <https://arxiv.org/html/2506.05012>
- **项目页：** <https://unified-fluid-robot-multiphysics.github.io/>
- **代码：** <https://github.com/RoboticExplorationLab/Aquarium.jl>（**已开源**，Julia）
- **作者：** Jeong Hun Lee et al.; CMU（Majidi Lab, Manchester Group）
- **入库日期：** 2026-07-20
- **一句话说明：** 从单一 Lagrangian 耦合多体刚/柔体动力学 + 不可压缩 NS 的可微多物理仿真器；优化波动游泳与 C-start；轨迹误差比粒子-流体基线低 ~75%；Aquarium.jl 开源（Julia）；真机迁移验证；RSS 2026 Finalist。

---

## 核心摘录（策展，非全文）

### 问题与动机

- 游泳机器人（鱼型等）性能高度依赖机体与流体的双向耦合，传统单独刚体或粒子流体近似丧失精确流体力反馈。
- 现有流固耦合仿真器要么不可微（无法直接优化），要么采用松散耦合（梯度不连续、迭代收敛慢）。
- 本文目标：从单一 Lagrangian 推导统一可微仿真器，使优化梯度连续穿透流体-机体边界。

### 关键贡献

1. **统一 Lagrangian 框架：** 单一广义坐标 Lagrangian 同时推导多体动力学 + 不可压缩 NS + 流固耦合边界条件。
2. **可微性：** 整个仿真系统可微，支持轨迹优化梯度直接回传。
3. **性能提升：** 波动游泳 + C-start 任务；轨迹误差比 SPH 粒子-流体基线低约 75%。
4. **真机迁移：** 优化控制序列直接部署到柔性鱼型硬件，无需额外真机微调。
5. **Aquarium.jl 开源：** Julia 实现，依赖 Zygote 自动微分。

### 方法要点

| 维度 | 统一多物理 |
|------|-----------|
| Lagrangian | 单一广义坐标，覆盖刚/柔体 + 流体 |
| 流体方程 | 不可压缩 Navier-Stokes（离散化） |
| 机体方程 | 铰接多体 / 柔性体动力学 |
| 耦合边界 | 从 Lagrangian 自然导出，无迭代 |
| 可微性 | 全系统可微；支持 Zygote 自动微分 |
| 优化任务 | 波动游泳、C-start 逃逸动作 |
| 基线 | SPH 粒子-流体仿真 |

### 实验摘要

- **波动游泳：** 游泳速度与轨迹精度；约 75% 更低轨迹误差 vs SPH 基线。
- **C-start：** 急速逃逸机动；轨迹误差改善相似量级。
- **真机：** 优化序列直接执行；真机游泳行为与仿真预测吻合。

### 代码状态

- Aquarium.jl：<https://github.com/RoboticExplorationLab/Aquarium.jl>，已公开，Julia，含环境定义、仿真步进与轨迹优化接口。

### 局限（论文自述）

- 高分辨率流体网格计算代价高；Julia 生态集成门槛；柔体材料参数需标定；当前仅水下不可压缩场景。

### 对 wiki 的映射

- [paper-unified-fluid-robot-multiphysics-swimming](../../wiki/entities/paper-unified-fluid-robot-multiphysics-swimming.md)
- [locomotion](../../wiki/tasks/locomotion.md)
- [sim2real](../../wiki/concepts/sim2real.md)
- [differentiable-simulation](../../wiki/concepts/differentiable-simulation.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2506.05012>
- 项目页：<https://unified-fluid-robot-multiphysics.github.io/>
- 代码：<https://github.com/RoboticExplorationLab/Aquarium.jl>
