---
type: entity
tags: [paper, humanoid, rl, motion-control, body-system-stack, sim2real, nvidia, cmu, lecar-lab]
status: complete
updated: 2026-09-23
venue: "RSS 2025"
code: https://github.com/LeCAR-Lab/ASAP
summary: "ASAP（RSS 2025）：真机 delta action 对齐 sim–real 动力学，回灌仿真微调敏捷全身 tracking；官方 MIT 代码 LeCAR-Lab/ASAP 基于 HumanoidVerse，含 G1 sim2real。"
related:
  - ./paper-notebook-asap-aligning-simulation-and-real-world-physics.md
  - ./humanoidverse.md
  - ../overview/humanoid-rl-motion-control-body-system-stack.md
  - ../overview/freedof-sim2real-44-papers-technology-map.md
  - ../methods/residual-policy-learning.md
sources:
  - ../../sources/papers/humanoid_rl_stack_25_asap_aligning_simulation_and_real_world_physics.md
  - ../../sources/sites/asap-agile-human2humanoid.md
  - ../../sources/repos/asap.md
  - ../../sources/papers/humanoid_rl_stack_42_catalog.md
---

# ASAP

**ASAP** 收录于 [具身智能研究室 · 42 篇 humanoid RL 运动控制长文](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA) **第 25/42** 篇，归类为 **03 感知式高动态运动**；完整方法、代码与评测见姊妹页 [paper-notebook-asap](./paper-notebook-asap-aligning-simulation-and-real-world-physics.md)（RSS 2025，[arXiv:2502.01143](https://arxiv.org/abs/2502.01143)）。

## 一句话定义

ASAP 用真机 rollout 训练 delta action 模型补偿 sim–real 动力学差，冻结回灌仿真微调 motion tracking 策略，部署时去掉 delta，使 Unitree G1 侧跳、前跳、球星动作等敏捷全身技能显著降低跟踪误差。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ASAP | Aligning Simulation And real-world Physics | 本文方法 |
| RL | Reinforcement Learning | motion tracking 与 delta 训练 |
| DR | Domain Randomization | 域随机化基线 |
| SysID | System Identification | 系统辨识基线 |
| Sim2Real | Simulation to Real | Gym→真机 G1 迁移 |

## 为什么重要

- 在 [人形 RL 身体系统栈](../overview/humanoid-rl-motion-control-body-system-stack.md) 中属于 **03 感知式高动态运动**（#25/42）。
- **Sim2Real 残差动力学路线：** 承认敏捷真机数据采集昂贵且危险（过热、损伤、规模受限），用 delta action 显式对齐而非纯调参 SysID 或保守 DR。
- **工程已开源：** [LeCAR-Lab/ASAP](https://github.com/LeCAR-Lab/ASAP)（MIT）基于 [HumanoidVerse](./humanoidverse.md)，含 motion 数据、重定向、sim2sim/sim2real。

## 核心信息

| 字段 | 内容 |
|------|------|
| 编号 | 25/42 |
| 系统栈层 | 03 感知式高动态运动 |
| 机构 | 卡内基梅隆大学（CMU）；英伟达（NVIDIA） |
| venue | RSS 2025 |
| 项目页 | <https://agile.human2humanoid.com/> |
| 代码 | <https://github.com/LeCAR-Lab/ASAP>（**已开源**） |

## 流程总览

```mermaid
flowchart LR
  sim["仿真 motion tracking 预训练"]
  real["真机 rollout"]
  delta["delta action 模型"]
  ft["回灌仿真微调"]
  dep["真机部署（无 delta）"]
  sim --> real --> delta --> ft --> dep
```

## 结论

**ASAP 把 sim-to-real 差距当成一个可学的量：先在仿真里用人类动作数据预训练 motion tracking 策略，再用真实数据训练 delta action 模型去修正仿真与真实之间的动力学偏差，且必须回灌仿真微调后才能在真机去掉 delta 部署。**

- 修正作用在 **动力学层面** 而非策略容量：敏捷全身动作的瓶颈是 sim–real 动力学偏差，补的是偏差项，不是继续堆更强的 tracking 策略。
- **回灌不可省略：** 仅学 delta 动力学而不用于仿真微调，效果不及完整管线；SysID 与纯 DR 亦落后（详见 [完整实体页](./paper-notebook-asap-aligning-simulation-and-real-world-physics.md)）。
- 它承认了多数 sim-to-real 工作回避的现实约束：真机采集敏捷动作会遇到电机过热、硬件损伤与数据规模受限。
- 定位与边界：本页属 42 篇栈 **03 感知式高动态运动**（#25/42）；完整 benchmark、源码时序与部署见姊妹页与 [asap.md](../../sources/repos/asap.md)。

## 常见误区

1. 感知 locomotion 的难点在 **闭环时延与几何误差**，不是单纯「加相机输入」。
2. **ASAP 需要真机闭环：** 不能期望纯仿真 DR 复现论文级敏捷 tracking；delta 训练依赖真机轨迹。
3. 与 [RobotDancing](./paper-notebook-robotdancing-residual-action-rl-enables-robust-l.md) Table V 的 ASAP-style 基线为 **同协议重实现**，不可与原论文表格直接横比。

## 实验与评测

- 三类迁移：**IsaacGym→IsaacSim**、**IsaacGym→Genesis**、**IsaacGym→真机 G1**；相对 SysID、DR、不回灌 delta 基线降低跟踪误差。
- 技能：侧跳、前跳、踢球、球星庆祝等全身敏捷动作（项目页 demo）。
- 量化细节以 [论文 PDF](https://arxiv.org/pdf/2502.01143) 与 [项目页](https://agile.human2humanoid.com/) 为准。

## 与其他页面的关系

- **完整归纳：** [paper-notebook-asap-aligning-simulation-and-real-world-physics.md](./paper-notebook-asap-aligning-simulation-and-real-world-physics.md)
- **框架底座：** [HumanoidVerse](./humanoidverse.md)
- 总框架：[humanoid-rl-motion-control-body-system-stack.md](../overview/humanoid-rl-motion-control-body-system-stack.md)
- Sim2Real 地图：[freedof-sim2real-44-papers-technology-map.md](../overview/freedof-sim2real-44-papers-technology-map.md)
- 残差谱系：[residual-policy-learning.md](../methods/residual-policy-learning.md)

## 参考来源

- [humanoid_rl_stack_25_asap_aligning_simulation_and_real_world_physics.md](../../sources/papers/humanoid_rl_stack_25_asap_aligning_simulation_and_real_world_physics.md) — 42 篇栈策展摘录
- [asap-agile-human2humanoid.md](../../sources/sites/asap-agile-human2humanoid.md) — 项目页核查
- [asap.md](../../sources/repos/asap.md) — 官方代码归档
- [humanoid_rl_stack_42_catalog.md](../../sources/papers/humanoid_rl_stack_42_catalog.md) — 总表

## 推荐继续阅读

- 官方代码：<https://github.com/LeCAR-Lab/ASAP>
- [42 篇 RL 运动控制（微信公众号）](https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA)
