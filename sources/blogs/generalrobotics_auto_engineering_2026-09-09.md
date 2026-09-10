# Introducing Auto Engineering for Robotics（General Robotics 官方博客）

> 来源归档

- **标题：** Introducing Auto Engineering for Robotics
- **类型：** blog / product-announcement / platform
- **作者：** General Robotics
- **原始链接：** <https://www.generalrobotics.company/post/introducing-auto-engineering-for-robotics>
- **发布日期：** 2026-09-09
- **入库日期：** 2026-09-10
- **一句话说明：** General Robotics 提出 **Auto-Engineering**：在 **GRID** 单体仓库上，用四类 **robotics harness**（机体摄取、世界经验、技能创建、部署评测）让 agent 闭环完成「理解机器人 → 造仿真/数据 → 建技能 → 真机评测与修复」，并把每次部署沉淀为可复用工程知识。

## 核心摘录

### 问题与类比

- 前沿能力（基础模型、算力、传感器、仿真器、新本体）日增，但把任意能力变成**可靠真机任务**仍依赖大量手工工程：硬件集成、标定、仿真环境、数据采集清洗、感知/控制环适配、评测设计、真机失败诊断与迭代。
- 类比软件工程：**coding harness**（仓库、工具、执行反馈、可验证信号）+ **可扩展可验证反馈**（测试、编译器、基准）使 agent 能客观评估并自改进；机器人需要自己的 harness。

### GRID 与 Auto-Engineering

- **GRID**：两年建设、覆盖 **50+ OEM 机器人**、数百模型与现代工作流（感知、控制、规划、RL/IL、分布式推理、仿真、部署），跨形态单体仓库。
- **Auto-Engineering**：让 GRID monorepo **compound**——新模型/算法/仿真器被吸收为 agent 可用能力；缺能力时 agent 自建并回灌平台。

### 四类 Robotics Harness

| Harness | 作用 | 实验室用例要点 |
|---------|------|----------------|
| **Robot Ingestion** | 关节、夹爪、相机、工作空间、控制接口、标定 → 可跨仿真/真机复用的工件；支持形态相近机器人技能迁移 | 双 Flexiv Rizon 取试管→倒烧杯；再迁移到 UR5e |
| **World Experience** | 仿真作为工程变量：选型/组合/扩展物理求解器、渲染器、传感器与后端 | DFSPH（NVIDIA Warp）+ MuJoCo 刚体耦合建模液体；合成 demo、随机化、失败定向实验 |
| **Skill Creation** | 模块化组合、策略训练（BC+DAgger）、人类示教（GELLO）、单视频 video-to-sim | 静态倒液→移动烧杯（状态策略）；搅棒搅拌（示教）；烧瓶旋液（仅手机视频） |
| **Deployment & Evaluation** | 仿真前检、系统辨识、真机评测、根据证据决定下一步改什么 | 运动学误差 143 mm/6.4°→5.7 mm/0.68°；分割 prompt 调优；实时 beaker 跟踪；控制频率 500 Hz→30 Hz 修复；深度模型选型 tabletop 误差 28 mm→1.2 mm |

### 时间与复利

- Flexiv 上**首个完整技能**约 **4 h**（机体摄取 ~20 min、初始仿真 ~10 min、其余为技能/评测/部署迭代）；同 setup 后续任务可至 **10–15 min**。
- 每次闭环沉淀：**机体知识**、**可迁移技能**、**模型与能力**、**失败与修复知识**；失败可回灌 ingestion / 仿真 / 感知 / 技能构造。

### 对 wiki 的映射

- [GRID（General Robotics）](../../wiki/entities/grid-general-robotics.md) — 平台实体与 harness 流程总览
- [真机策略 autoresearch 闭环搭建指南](../../wiki/queries/real-robot-policy-autoresearch-harness.md) — coding/agent harness 与可验证反馈对照
- [Sim2Real](../../wiki/concepts/sim2real.md) — 部署 harness 中的 sim-to-real gap 修复案例
- [Data Flywheel](../../wiki/concepts/data-flywheel.md) — 「部署产生知识 → 下次更快」的复利叙事
