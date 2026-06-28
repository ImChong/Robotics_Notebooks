---
type: overview
tags: [simulation, embodied-ai, survey, mujoco, habitat, isaac-gym, benchmark]
status: complete
updated: 2026-06-22
related:
  - ../entities/mujoco.md
  - ../entities/ai2-thor.md
  - ../entities/matterport3d-simulator.md
  - ../entities/habitat-sim.md
  - ../entities/igibson.md
  - ../entities/isaac-gym.md
  - ../entities/maniskill2.md
  - ../entities/behavior-1k.md
  - ../entities/carla.md
  - ../entities/pybullet.md
  - ../entities/robogen.md
  - ../entities/genesis-sim.md
  - ../entities/sapien.md
  - ../overview/robot-training-stack-layers-technology-map.md
  - ../queries/simulator-selection-guide.md
  - ../tasks/vision-language-navigation.md
sources:
  - ../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md
  - ../../sources/raw/wechat_sim_platforms_top8_2026-06-22.md
summary: "依据深蓝具身智能 2026-06 盘点，把 2010 年后改变机器人学习的 TOP 8 仿真平台整理为十年演进技术地图；核心判断：平台轨迹反映重心从物理精度→视觉交互→渲染吞吐→GPU 并行→泛化基准→人类需求对齐的迁移。"
---

# 十年仿真平台技术地图（TOP 8）

> **本页定位**：为 [深蓝具身智能 · 10 年间哪些仿真平台真正改变了机器人学习？](https://mp.weixin.qq.com/s/iaw_lWAR--AwppyMeIK4lw) 提供 **按历史脉络组织的阅读坐标**；不复述各平台安装细节，只保留 **演进判断、TOP 8 节点索引、与训练栈分层地图的分工**。姊妹篇 [训练栈分层技术地图](./robot-training-stack-layers-technology-map.md) 偏 **2026 工具链六层互补**；本页偏 **2010–2023 代表性平台史**。

## 一句话观点

仿真平台的发展轨迹清晰反映具身智能研究重心的演进：**没有单一「赢家通吃」**，而是从 [MuJoCo](../entities/mujoco.md) 的物理奠基，经 [AI2-THOR](../entities/ai2-thor.md) / [Matterport3D](../entities/matterport3d-simulator.md) 的视觉交互与 VLN 基础设施，到 [Habitat](../entities/habitat-sim.md) 的渲染吞吐、[Isaac Gym](../entities/isaac-gym.md) 的 GPU 并行时代，再到 [ManiSkill2](../entities/maniskill2.md) 的操作泛化与 [BEHAVIOR-1K](../entities/behavior-1k.md) 的人类需求对齐——**每一代补的是当时最贵的瓶颈**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-and-Language Navigation | 视觉–语言导航 |
| VLA | Vision-Language-Action | 视觉–语言–动作多模态策略 |
| RL | Reinforcement Learning | 强化学习 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| GPU | Graphics Processing Unit | 并行仿真与训练算力 |

## 流程总览：十年演进主线

```mermaid
flowchart LR
  MJ["① MuJoCo<br/>物理控制奠基"]
  TH["② AI2-THOR<br/>室内状态交互"]
  MP["③ Matterport3D Sim<br/>VLN 真实感 + R2R"]
  HB["④ Habitat<br/>极致渲染吞吐"]
  IG["⑤ iGibson<br/>真实感 + PyBullet 物理"]
  IGym["⑥ Isaac Gym<br/>GPU 端到端并行"]
  MS["⑦ ManiSkill2<br/>操作泛化基准"]
  B1K["⑧ BEHAVIOR-1K<br/>1000 日常活动"]
  MJ --> TH --> MP --> HB
  HB --> IG --> IGym --> MS --> B1K
```

## TOP 8 平台节点（图谱 hub）

| # | 平台 | Wiki 节点 | 文内强调 | 被引量（文内） |
|---|------|-----------|----------|----------------|
| 01 | MuJoCo | [mujoco](../entities/mujoco.md) | 连续时间动力学、[Gymnasium](../entities/gymnasium.md) MuJoCo 域底层 | 9530 |
| 02 | AI2-THOR | [ai2-thor](../entities/ai2-thor.md) | 室内 3D + 细粒度状态交互 | 1510 |
| 03 | Matterport3D Simulator | [matterport3d-simulator](../entities/matterport3d-simulator.md) | 真实扫描 RGB-D + R2R 基准 | 2270 |
| 04 | Habitat | [habitat-sim](../entities/habitat-sim.md) | 单 GPU 数千–上万 FPS | 2426 |
| 05 | iGibson | [igibson](../entities/igibson.md) | PyBullet + 可交互日常物体 | 268 |
| 06 | Isaac Gym | [isaac-gym](../entities/isaac-gym.md) | GPU 原生仿真管线 | 1820 |
| 07 | ManiSkill2 | [maniskill2](../entities/maniskill2.md) | SAPIEN 上操作泛化基准 | 260 |
| 08 | BEHAVIOR-1K | [behavior-1k](../entities/behavior-1k.md) | 社会调查驱动 1000 活动 | 514 |

## 文尾补充平台（非 TOP 8 正文，但文内点名）

| 平台 | Wiki 节点 | 方向 |
|------|-----------|------|
| CARLA | [carla](../entities/carla.md) | 自动驾驶城市场景 |
| PyBullet / Flex | [pybullet](../entities/pybullet.md) | 软体与形变、通用刚体 |
| RoboGen | [robogen](../entities/robogen.md) | LLM 驱动任务/数据生成 |
| Genesis | [genesis-sim](../entities/genesis-sim.md) | 新兴多物理并行仿真 |
| SAPIEN（ManiSkill2 后端） | [sapien](../entities/sapien.md) | 关节体操作引擎 |

## 文内收束判断（策展）

| 判断 | 含义 |
|------|------|
| 基石叙事 | VLA/VLN 数据饥渴使仿真从「辅助工具」变为「每次跃迁的基石」 |
| 瓶颈迁移 | 从物理精度 → 视觉交互 → 真实感导航 → 渲染 FPS → GPU 并行 → 操作泛化 → 人类需求 |
| 非洗牌 | 老平台未被简单替代；新平台补 **当时最贵环节**（与 [训练栈分层](./robot-training-stack-layers-technology-map.md) 的「分层共存」一致） |
| Sim2Real | 平台同时为前沿任务供数据引擎，并加速仿真到现实转化 |

## 按目标选入口

| 你的目标 | 从哪开始 |
|----------|----------|
| 理解 RL 物理引擎为何选 MuJoCo | [MuJoCo](../entities/mujoco.md) |
| 室内指令跟随与状态交互 | [AI2-THOR](../entities/ai2-thor.md) |
| VLN 起源与 R2R | [Matterport3D Simulator](../entities/matterport3d-simulator.md) + [R2R 论文](../entities/paper-vln-01-r2r.md) |
| 亿级步数导航 RL 训练 | [Habitat-Sim](../entities/habitat-sim.md) |
| 移动 + 物理操作室内任务 | [iGibson](../entities/igibson.md) |
| 万环境 GPU 并行 loco | [Isaac Gym](../entities/isaac-gym.md) → 新实验 [Isaac Lab](../entities/isaac-lab.md) |
| 机械臂操作泛化 leaderboard | [ManiSkill2](../entities/maniskill2.md) |
| 开放世界日常活动上限考试 | [BEHAVIOR-1K](../entities/behavior-1k.md) |
| 2026 六层训练栈怎么选 | [训练栈分层地图](./robot-training-stack-layers-technology-map.md) |
| Locomotion 三选一速查 | [仿真器选型指南](../queries/simulator-selection-guide.md) |

## 关联页面

- [训练栈分层技术地图](./robot-training-stack-layers-technology-map.md)
- [VLN 10 篇技术地图](./vln-10-papers-technology-map.md)
- [视觉–语言导航](../tasks/vision-language-navigation.md)
- [仿真器选型指南](../queries/simulator-selection-guide.md)

## 参考来源

- [sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md](../../sources/blogs/wechat_shenlan_sim_platforms_top8_decade.md)
- [sources/raw/wechat_sim_platforms_top8_2026-06-22.md](../../sources/raw/wechat_sim_platforms_top8_2026-06-22.md)

## 推荐继续阅读

- [深蓝具身智能 原文](https://mp.weixin.qq.com/s/iaw_lWAR--AwppyMeIK4lw)
- [具身智能研究室 · 训练栈分层姊妹篇](https://mp.weixin.qq.com/s/Z9pgVa48wQKLYVRD3psnhw)
