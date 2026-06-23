---

type: entity
tags: [entity, simulator, manipulation, benchmark, sapien, generalization, ucsd]
status: complete
updated: 2026-06-22
related:
  - ./sapien.md
  - ./paper-notebook-maniskill3-gpu-parallelized-robotics-simulation.md
  - ../methods/imitation-learning.md
  - ../overview/sim-platforms-decade-technology-map.md
sources:
  - ../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md
summary: "2023 年基于 SAPIEN 的统一机械臂操作基准：跨类别/几何物体、多观测模态与一致评估协议，系统性考察操作技能泛化与模仿学习数据质量。"
---

# ManiSkill2

**ManiSkill2** 是 2023 年发布的 **通用可泛化操作技能（Generalizable Manipulation Skills）统一基准**，构建于 [SAPIEN](./sapien.md) 仿真器之上。

## 一句话定义

> ManiSkill2 把机械臂操作评测从「单一任务过拟合」推进到 **跨物体类别与几何的泛化考试**：大量多样化物体模型 + 统一观测/控制接口 + 高质量演示数据，成为 IL/RL 操作路线的标准对照台。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IL | Imitation Learning | 模仿学习 |
| RL | Reinforcement Learning | 强化学习 |
| RGB-D | RGB + Depth | 彩色与深度联合观测 |
| GPU | Graphics Processing Unit | 并行仿真算力 |

## 为什么重要

操作研究亟需 **可比的泛化指标**。ManiSkill2 的贡献：

1. **跨类别物体**：覆盖多几何、多语义类别，避免单一 mesh 过拟合。
2. **多观测模式**：点云、RGB-D 等，服务不同策略架构。
3. **演示 + 评测一体**：提供模仿学习数据，并在刚体/软体等子任务上维持 **一致 leaderboard**。
4. **后继演进**：社区已有 ManiSkill3 等 GPU 并行扩展（见 [paper-notebook-maniskill3](./paper-notebook-maniskill3-gpu-parallelized-robotics-simulation.md)），本节点锚定 **ManiSkill2 论文基准**。

## 核心结构/机制

- **SAPIEN 后端**：关节体交互与渲染。
- **任务套件**：抓取、放置、工具使用等多类操作 primitive。
- **数据接口**：标准化 trajectory 与评估脚本。

## 常见误区或局限

- **误区：ManiSkill2 = SAPIEN** — SAPIEN 是 **引擎**；ManiSkill2 是 **基准与任务层**。
- **局限：人形全身 loco-manip** — 见 [BEHAVIOR-1K](./behavior-1k.md)、[Isaac Lab](./isaac-lab.md) 等更厚栈。

## 关联页面

- [SAPIEN](./sapien.md)
- [十年仿真平台技术地图](../overview/sim-platforms-decade-technology-map.md)
- [模仿学习](../methods/imitation-learning.md)

## 参考来源

- [sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md](../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md)
- Gu et al., *ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills* — [arXiv](https://arxiv.org/abs/2302.04659)

## 推荐继续阅读

- [ManiSkill 官方站](https://maniskill.ai/)
- [SAPIEN](./sapien.md) — 底层仿真引擎
