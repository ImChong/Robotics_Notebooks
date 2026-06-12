# A Hierarchical, Model-Based System for High-Performance Humanoid Soccer（ARTEMIS · 群控摘录）

> 来源归档（ingest · 人形机器人群控 / RoboCup 2024 Adult-Size 冠军系统）

- **标题：** A Hierarchical, Model-Based System for High-Performance Humanoid Soccer
- **缩写：** **ARTEMIS**（Advanced Robotic Technology for Enhanced Mobility and Improved Stability）
- **类型：** paper / system
- **arXiv：** <https://arxiv.org/abs/2512.09431>
- **机构：** UCLA RoMeLa（Dennis W. Hong 组）等
- **成果：** **RoboCup 2024 Adult-Sized Humanoid Soccer 冠军**
- **开源：** [artemis_open_source](https://github.com/)（论文引用平台仓库）
- **入库日期：** 2026-06-12
- **一句话说明：** 与去中心化 swarm 路线对照的 **集中式行为管理器** 范例：立体视觉+CLAP 定位球/队友/对手 → DAVG+cf-MPC 碰撞规避导航 → **centralized behavior planner** 维护比赛记忆、短视界运动预测、**角色与射门决策**，经 kick/neck manager 下发至 1 kHz 全身控制。

## 核心论文摘录（MVP · 侧重多机协调）

### 1) 软件栈与多体感知（§II, Figure 1–2）

- **链接：** <https://arxiv.org/abs/2512.09431>
- **摘录要点：**
  - **感知：** ZED 2 立体 ~60 Hz 检测场地标志、球、**队友、对手**；深度近距障碍作碰撞规避备份。
  - **定位：** CLAP 几何定位融合标志与惯性，输出世界系下 **自机与对象状态**。
  - **局限对照（§I）：** 近年 RL 技能（如 Booster 2025、Haarnoja 2024）多为 **单机** 追球射门，需嵌入更大 **手工协调栈** 才能处理队友/对手/避障——ARTEMIS 强调 **全场集成** 而非孤立技能。
- **对 wiki 的映射：**
  - [paper-notebook-a-hierarchical-model-based-system-for-high-perfo](../../wiki/entities/paper-notebook-a-hierarchical-model-based-system-for-high-perfo.md)

### 2) 集中式行为管理器（§II 末段）

- **摘录要点：**
  - **Behavior planner（ROS 2 节点）：** 维护球/对手 **内部记忆**；**短视界运动预测**；根据赛况选 **期望位姿、角色、踢球动作**。
  - **Kick manager / Neck manager：** 将战术决策转为踢球类型/时机与凝视策略（平衡盯球与看标志）。
  - **Mid-level：** DAVG 全局路径 + cf-MPC 跟踪，考虑场地边界与 **移动障碍（其他机器人）**。
  - **与低层：** 通过 shared memory 与 1 kHz locomotion / in-gait kicking 同步，避免高层慢速更新破坏平衡。
- **对 wiki 的映射：**
  - [人形多机协调](../../wiki/concepts/humanoid-multi-robot-coordination.md) — **集中式战术层** 代表

### 3) 与 swarm / SPL 路线对比（归纳）

| 维度 | ARTEMIS（本文） | Swarm ACO+flocking（Sensors 2025） | SPL DWM+DTA（arXiv:2401.15026） |
|------|-----------------|-------------------------------------|----------------------------------|
| 决策拓扑 | 单机 **集中 behavior planner**（每机自主但战术逻辑集中式设计） | **全去中心化** ACO+flocking | **分布式** 市场拍卖 + Voronoi |
| 通信 | 队友/对手主要靠 **视觉检测**，非论文重点 | **周期性 UDP 状态** | **极低带宽事件** |
| 验证 | **真机 Adult-Size 国际赛** | Webots 仿真为主 | NAO **真机 RoboCup** |
| 强项 | 感知–导航–踢球 **一体化**、成人尺寸动态对抗 | 角色重分配快、故障/丢包容忍 | 规则驱动极限低通信 |

- **对 wiki 的映射：**
  - [humanoid-multi-robot-coordination.md](../../wiki/concepts/humanoid-multi-robot-coordination.md)

## 对 wiki 的映射（汇总）

- [paper-notebook-a-hierarchical-model-based-system-for-high-perfo.md](../../wiki/entities/paper-notebook-a-hierarchical-model-based-system-for-high-perfo.md) — 升格 team-coordination 归纳
- [humanoid-multi-robot-coordination.md](../../wiki/concepts/humanoid-multi-robot-coordination.md)
- [humanoid-soccer.md](../../wiki/tasks/humanoid-soccer.md)

## 引用

```bibtex
@article{wang2025artemis,
  title   = {A Hierarchical, Model-Based System for High-Performance Humanoid Soccer},
  author  = {Wang, Quanyou and Zhu, Mingzhang and Hou, Ruochen and others},
  journal = {arXiv preprint arXiv:2512.09431},
  year    = {2025}
}
```
