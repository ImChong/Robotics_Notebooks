---
type: entity
tags: [entity, simulator, embodied-ai, manipulation, pybullet, stanford]
status: complete
updated: 2026-06-22
related:
  - ./pybullet.md
  - ./ai2-thor.md
  - ./habitat-sim.md
  - ./sapien.md
  - ../tasks/loco-manipulation.md
  - ../overview/sim-platforms-decade-technology-map.md
sources:
  - ../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md
summary: "斯坦福等 2020 年推出的 iGibson：在 PyBullet 上融合大规模真实感室内场景与可交互日常物体，支持长视距操作任务，为 VLA 提供更贴近现实的测试环境。"
---

# iGibson

**iGibson** 是斯坦福大学等机构 2020 年发布的 **交互式室内仿真环境**，强调 **真实感视觉场景** 与 **高保真物理交互** 的融合。

## 一句话定义

> iGibson 让智能体不仅能「看和走」，还能在 **接近真实物理规律** 的室内场景中开关柜门、拧水龙头——把大规模真实感网格与 PyBullet 动力学接到同一工作台。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| iGibson | Interactive Gibson | 可交互 Gibson 场景扩展 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略 |
| URDF | Unified Robot Description Format | 机器人模型描述格式 |
| RL | Reinforcement Learning | 强化学习 |

## 为什么重要

早期导航平台（[Habitat](./habitat-sim.md)、[Matterport3D](./matterport3d-simulator.md)）强在 **移动**；[AI2-THOR](./ai2-thor.md) 强在 **状态交互** 但物理简化。iGibson 填补：

1. **真实感 + 物理**：基于 [PyBullet](./pybullet.md)，物体带关节与物理属性（门、抽屉、水龙头）。
2. **大规模场景**：延续 Gibson 真实扫描室内布局传统。
3. **长视距操作**：支持需要移动 + 操作的复合任务，贴近 VLA 落地前的 **仿真评测** 需求。

## 核心结构/机制

- **PyBullet 后端**：刚体动力学与接触。
- **交互物体库**：日常物体带 articulation 与语义标签。
- **多模态传感**：RGB、深度、语义分割等（随版本）。

## 常见误区或局限

- **误区：iGibson = Gibson 数据集** — Gibson 是 **场景资产**；iGibson 是 **可交互仿真环境**。
- **局限：GPU 并行吞吐** — 万环境并行 RL 见 [Isaac Gym](./isaac-gym.md)；部件级操作基准见 [ManiSkill2](./maniskill2.md)。

## 关联页面

- [PyBullet](./pybullet.md)
- [SAPIEN](./sapien.md)
- [Habitat-Sim](./habitat-sim.md)
- [十年仿真平台技术地图](../overview/sim-platforms-decade-technology-map.md)

## 参考来源

- [sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md](../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md)
- Shen et al., *iGibson 1.0: A Simulation Environment for Interactive Tasks in Large Realistic Scenes* — [arXiv](https://arxiv.org/abs/2012.02924)

## 推荐继续阅读

- [iGibson 项目页](http://svl.stanford.edu/igibson/)
- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
