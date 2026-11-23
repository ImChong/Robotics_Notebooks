---
type: entity
project_id: sparc-2611-15443
tags: [paper, robot-platform, locomotion, navigation]
status: complete
updated: 2026-11-21
arxiv: "2611.15443"
project: https://rrl.robocup.org/sparc/
code: https://github.com/STAR-Explorer-Robotics/Robocup_system
original:
  - ../../sources/papers/sparc-2611-15443.md
sources:
  - ../../sources/papers/sparc-2611-15443.md
  - ../../sources/sites/sparc-2611-15443.md
  - ../../sources/repos/sparc-2611-15443.md
related:
  - ../overview/embodied-frontier-29-papers-2026-11-16.md
  - ./paper-maniladder-manipulation-benchmark.md
summary: "以 Unitree A2-W 轮足机器人统一救援比赛的硬件基础，提供初始地形、导航软件与仿真材料。"
---

# SPARC: RoboCup Standard Platform for Autonomous Rescue Competition with Initial Datasets, Software, and Simulation Environments｜SPARC：RoboCup 自主救援轮足统一研究平台

## 一句话定义

**SPARC：RoboCup 自主救援轮足统一研究平台**：以 Unitree A2-W 轮足机器人统一救援比赛的硬件基础，提供初始地形、导航软件与仿真材料。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| SPARC | Standard Platform for Autonomous Rescue Competition | 自主救援竞赛标准平台 |
| LiDAR | Light Detection and Ranging | 激光测距 |
| ROS2 | Robot Operating System 2 | ROS 2 |

## 为什么重要

以 Unitree A2-W 轮足机器人统一救援比赛的硬件基础，提供初始地形、导航软件与仿真材料。 需要结合任务边界、数据/算力条件及作者真实实现来判断，而不是只看展示视频。

## 核心信息

| 证据项 | 官方信息 |
|---|---|
| 论文 | [arXiv:2611.15443v1](https://arxiv.org/abs/2611.15443) |
| 项目页 | [原始页面](https://rrl.robocup.org/sparc/) |
| 源码 | [作者仓库](https://github.com/STAR-Explorer-Robotics/Robocup_system)；仅限已发布模块 |

## 核心原理

标准平台含 LiDAR/IMU、移动执行器接口及可扩展机械臂；搭配激光惯导定位、地形高程与风险图、路径规划、跟踪器及 ROS 2 机器人接口；仓库整理不同技术模块和场景资产。

### 流程总览

```mermaid
flowchart LR
  V0["Unitree A2-W 传感器"]
  V1["激光惯性里程计与地形图"]
  V2["风险地形路径规划"]
  V3["路径跟踪与执行接口"]
  V4["救援场景仿真评估"]
  V0 --> V1
  V1 --> V2
  V2 --> V3
  V3 --> V4
```

该图为研究问题与方法/控制流，实际软件调用须看源码时序图及开放边界。

## 源码运行时序图

**不适用：** 官方源码尚无经过核实的可运行完整入口，或当前公开的代码不具备端到端可执行部署能力，故不虚构 sequenceDiagram。

## 工程实践

1. 检查输入观察、动作/轨迹表示、数据来源和实验任务的部署条件是否一致。
2. 从[源码核查记录](../../sources/repos/sparc-2611-15443.md)确认公开模块的真实入口、依赖与输出。
3. 将仿真指标、真机成功率、推理延迟与安全/碰撞检查分开汇报。

## 实验与评测

论文提供 RoboCup/LRC 地形样例与开源软件结构，评估面向导航与竞赛工具链，尚非全行业统一实机基准。

## 结论

**以 Unitree A2-W 轮足机器人统一救援比赛的硬件基础，提供初始地形、导航软件与仿真材料。**

1. 先复核指标口径与论文约束，再比较其他策略或系统。
2. 将作者验证的模块和尚未发布/未复现的模块分开记录。
3. 把误差、失败模式与传感器/真实场景分布偏移作为后续验收依据。

## 与其他工作对比

- **对比传统 RoboCup Rescue League（RRL）异构平台模式（§I、§II-D）：** 过去二十余年 RRL 虽然标准化了灾后测试通道与客观评分规程，但各参赛队自研硬件差异巨大，导致工程重复投入且算法难以跨队公平复现与对比；SPARC 借鉴 RoboCup Soccer 标准平台组（SPL）经验，以商用轮足机器人 Unitree A2-W 统一移动底座与感知接口，并规定赛后强制开源软件栈，将研究竞争聚焦于定位、建图、导航规划与具身决策。
- **与已有单点开源基准数据对比定位（§IV）：** 不同于纯粹室内桌面或通用街景数据集，SPARC 同步开放来自 IEEE ICRA 2026 LRC（Unitree A2 Pro 倾斜赛道 ROS 2 bags + 手持高精度彩色点云真值）与 RoboCup Rescue 2026（Incheon 实赛 Unitree A2-W ROS 2 bags）的两组实测救援数据集；原文定位是竞赛平台与生态发布，未给出不同外部导航算法在该平台的横向跑分表。

## 局限与风险

官网规则仍在发展，机械臂配置未统一；官方仓 README 明确是源码整理发布，未验证整包可直接部署运行。

## 关联页面

- [29 篇具身论文研究地图](../overview/embodied-frontier-29-papers-2026-11-16.md)
- [ManiLadder 操作任务难度分层](./paper-maniladder-manipulation-benchmark.md)

## 参考来源

- [论文原文归档](../../sources/papers/sparc-2611-15443.md)
- [官方项目及开源状态](../../sources/sites/sparc-2611-15443.md)
- [官方源码核对](../../sources/repos/sparc-2611-15443.md)

## 推荐继续阅读

- [原始论文 HTML](https://arxiv.org/html/2611.15443v1)
- [代码仓库](https://github.com/STAR-Explorer-Robotics/Robocup_system)
