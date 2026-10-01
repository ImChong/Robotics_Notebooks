---
type: entity
tags: [navigation, social-navigation, ros2, simulation, benchmark, mobile-robot, reinforcement-learning, nus, tum, open-source]
status: complete
updated: 2026-10-01
related:
  - ./navigation2.md
  - ./paper-legnav-calf.md
  - ./paper-commnav.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ../tasks/autonomous-exploration.md
  - ../concepts/ros2-basics.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/sites/5-arena-rosnav.md
  - ../../sources/repos/arena-rosnav.md
  - ../../sources/papers/arena_rosnav_lineage.md
summary: "Arena-Rosnav 5.0（NUS/TUB/TUM，RSS 2025）：ROS 2 社交导航开发与评测平台——多仿真后端、行人/机器人 spawn、MBF 与 DRL 训练、arena-evaluation 社交指标；GitHub 组织 Arena-Rosnav 已开源。"
---

# Arena-Rosnav（社交导航仿真与 Benchmark）

**Arena-Rosnav**（[5.0 项目页](https://5.arena-rosnav.org/)，[文档](https://docs.arena-rosnav.org/)，[GitHub 组织](https://github.com/Arena-Rosnav)）是面向 **人机共存环境** 的 **ROS 2 导航算法开发、训练与对比评测** 开源平台。5.0 版（RSS 2025）在 4.0 的生成式世界与 ROS 2 工具链上，强调 **NVIDIA Isaac 系照片级仿真接入**、**SOTA 社交导航 benchmark** 与 **可定制紧急/救援类场景**。

## 一句话定义

**把「造世界 → 放行人 → 放机器人 → 跑规划器 → 用社交指标打分」收成一条 ROS 2 流水线**，让研究者专注算法而不是重复搭仿真与评测脚本。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ROS 2 | Robot Operating System 2 | Arena 的规划器、传感器与评测接口层 |
| MBF | Move Base Flex | 灵活导航动作服务器；Arena 生态内完整集成 |
| DRL | Deep Reinforcement Learning | `rosnav-rl` 等与 stable-baselines3 训练管线 |
| RL | Reinforcement Learning | 局部规划器学习基线之一 |
| SFM | Social Force Model | 行人社会力等 crowd 模型（组织内 pedsim/crowdsim 系） |

## 为什么重要

- **社交导航专用栈：** 区别于只测几何避障的 Nav2 教程场景，Arena 默认带 **行人动力学 + 社交度量**（个人空间、注视等），与 [LegNav/CALF](./paper-legnav-calf.md)、[CommNav](./paper-commnav.md) 等「人感知/人交互导航」研究同一问题带。
- **多后端、一条 ROS 接口：** **Gazebo / Isaac Sim / Unity / Flatland** 等可切换，降低「换仿真就要重写栈」的迁移成本。
- **Benchmark 可复现：** [arena-evaluation](https://github.com/Arena-Rosnav/arena-evaluation) 与项目页列出的指标，支持在同一套世界上对比经典规划、社会力跟随与 DRL 规划器。
- **工程入口清晰：** 文档 **automatic installation** + 组织内 40+ 子仓，MIT 许可为主（以各仓为准）。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 新加坡国立大学（NUS）、柏林工业大学（TUB）、慕尼黑工业大学（TUM） |
| **5.0 论文** | [arena5.pdf](https://5.arena-rosnav.org/arena5.pdf)（RSS 2025） |
| **前代** | Arena 4.0 — [arXiv:2409.12471](https://arxiv.org/abs/2409.12471) |
| **代码** | [Arena-Rosnav/arena-rosnav](https://github.com/Arena-Rosnav/arena-rosnav) |
| **开源** | **已开源**（步骤 2.5，2026-09-30 项目页 Code → GitHub 组织） |

## 流程总览

```mermaid
flowchart LR
  W["Generate Worlds\n办公室/医院/仓库等"]
  H["Spawn Humans\n社会力 / crowd"]
  R["Spawn Robots\n多平台模型"]
  P["Planners\n经典 / MBF / DRL"]
  E["arena-evaluation\n社交 + 效率指标"]
  W --> H
  H --> R
  R --> P
  P --> E
```

### 任务模式（项目页）

| Task Mode | 简述 | 机器人 | 障碍/行人 |
|-----------|------|--------|-----------|
| `scenario` | 加载场景文件 | ✓ | ✓ |
| `random` | 随机位姿 | ✓ | ✓ |
| `parametrized` | 细粒度随机 | | ✓ |
| `guided` | 路点序列 | ✓ | |
| `explore` | 探索地图 | ✓ | |

### Benchmark 指标（策展）

- 碰撞次数、到达目标时间、路径长度
- 停留在个人空间的时间、注视行人时间、被行人看见时间

## 工程实践

| 检查项 | 建议 |
|--------|------|
| **安装** | 优先 [docs 自动安装](https://docs.arena-rosnav.org/en/latest/tutorials/installation/)，再选仿真后端 |
| **与 Nav2 关系** | 底盘仍常经 **MBF / Nav2 插件**；Arena 提供 **世界 + 行人 + 评测**，不是替代 [Navigation2](./navigation2.md) 的全栈定位 SLAM |
| **Isaac 路径** | 照片级与 GPU 训练走 [arena-isaac](https://github.com/Arena-Rosnav/arena-isaac)；需 NVIDIA Isaac 环境与许可 |
| **DRL** | [rosnav-rl](https://github.com/Arena-Rosnav/rosnav-rl) + 组织 fork 的 [stable-baselines3](https://github.com/Arena-Rosnav/stable-baselines3) |
| **真机** | [real-world-bench](https://github.com/Arena-Rosnav/real-world-bench) 指向真机评测扩展；Sim2Real 仍须自校传感器与行人检测栈 |

## 局限与风险

- **依赖栈重：** 全功能路径涉及 ROS 2、可选 Isaac/Unity 与大量子仓，首次装环境成本高于单一 Gazebo 教程。
- **5.0 论文分发：** 5.0 正文以 **项目页 PDF** 为主；系列方法细节部分仍引用 **4.0 arXiv** 叙事，读论文时需对照版本号。
- **社交指标 ≠ 真机社交合规：** 仿真行人模型与真实人群行为有 gap；上线仍需真机试验与感知延迟评估。
- **TUB 机构 tag：** 柏林工业大学尚未写入 `institutions.json`；正文机构表已列全称，后续可补注册表 alias。

## 与其他页面的关系

- [Navigation2](./navigation2.md) — ROS 2 通用导航框架；Arena 在其上叠加 **行人场景与社交评测**。
- [navigation-slam-autonomy-stack](../overview/navigation-slam-autonomy-stack.md) — 移动机器人 SLAM + Nav2 总览；学习型社交导航实验可接 Arena benchmark。
- [LegNav/CALF](./paper-legnav-calf.md) — 踝高 LiDAR **腿部感知** 社交导航（不同传感与仿真假设，可互补对比）。
- [具身大模型评测基准选型闭环知识链](../queries/embodied-eval-benchmark-selection-loop.md) — Arena 属其 ③ 策略任务成功率评测层的社交导航仿真基准；仿真指标外推真机需 ④ sim↔real 校准
- [CommNav](./paper-commnav.md) — **主动通信** 找人导航（Habitat）；Arena 侧重 **避障与社交距离** 连续导航。

## 参考来源

- [Arena 5.0 项目页](../../sources/sites/5-arena-rosnav.md)
- [Arena-Rosnav 仓库归档](../../sources/repos/arena-rosnav.md)
- [Arena 论文谱系（4.0 arXiv / 5.0 RSS）](../../sources/papers/arena_rosnav_lineage.md)

## 推荐继续阅读

- [Arena 文档 — Installation](https://docs.arena-rosnav.org/en/latest/tutorials/installation/)
- [RSS 2025 论文页](https://roboticsconference.org/program/papers/92/)
- [Arena 5.0 演示视频（YouTube）](https://www.youtube.com/watch?v=MrKrnHhk8IA)
- [Nav2 官方文档](https://docs.nav2.org/)
