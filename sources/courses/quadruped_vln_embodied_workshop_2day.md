# 四足 × VLN × 具身智能：2 天线下实战 + 1 月线上答疑（课程大纲）

> 来源归档（ingest）

- **标题：** 2 Days Offline Practical Combat + 1 Month Online Q&A Guidance（四足机器人 / VLN / 具身智能实战营）
- **类型：** course
- **来源：** 具身智能研究室课程日程截图整理（硬件：四足 + LiDAR + 相机 + NVIDIA Orin NX + RTX 4070 工作站）
- **收录日期：** 2026-08-04
- **一句话说明：** 以 TravExplorer 导航框架与 OpenClaw 语音控制为项目主线，串联四足建图定位、Habitat 语义感知、SAM3/BLIP-2 零样本目标导航与仿真→真机部署。

## 为什么值得保留

- 把 **传统导航 → VLN → 零样本 ObjectNav** 的任务演进压进两天可练链路，并点名 **TravExplorer / OpenClaw / Habitat / SAM3 / BLIP-2** 等可复现节点。
- 硬件栈（Orin NX + LiDAR + 四足 + 4070）与本库四足导航、语义建图、边缘部署文档直接对齐。
- 可作为「截图技术点 → 独立 wiki 详情节点」的覆盖验收清单。

## 日程与技术点

### Day 1 上午（10:00–12:00）— 四足与 VLN 导航概述

| 类型 | 内容 |
|------|------|
| 技术点 | 四足导航任务与应用场景；传统导航 → VLN；TravExplorer 框架下的导航任务 |
| 项目 | **OpenClaw**：部署 OpenClaw，实现四足语音指令控制 |

### Day 1 下午（14:00–18:00）— 四足运动控制进阶

| 类型 | 内容 |
|------|------|
| 技术点 | 建图与定位；局部规划与避障；高层导航接口 vs 底层控制接口 |
| 实践 | 真机点到点导航与动态避障 |

### Day 2 上午（10:00–12:00）— 构建具身语义认知地图

| 类型 | 内容 |
|------|------|
| 技术点 | 视觉语义理解（像素→实体）；视觉–语言特征融合与语义空间对齐；Habitat 仿真器 |
| 项目 | **室内环境语义感知**（Habitat） |

### Day 2 下午（14:00–18:00）— 零样本目标导航实战

| 类型 | 内容 |
|------|------|
| 技术点 | 语言指令 → 机器人动作；SAM3 + BLIP-2 零样本泛化；仿真 → 真机 → 项目设计工作流 |
| 项目 | **零样本目标导航**：仿真与四足真机部署 |

## 对 wiki 的映射

| 课程节点 | wiki 详情页 |
|----------|-------------|
| 课程总览 | [quadruped-vln-embodied-workshop](../../wiki/overview/quadruped-vln-embodied-workshop.md) |
| VLN | [vision-language-navigation](../../wiki/tasks/vision-language-navigation.md) |
| TravExplorer | [paper-travexplorer](../../wiki/entities/paper-travexplorer.md) |
| OpenClaw | [openclaw](../../wiki/entities/openclaw.md) |
| 四足分层栈 / 高低层接口 | [hierarchical-quadruped-navigation-stack](../../wiki/concepts/hierarchical-quadruped-navigation-stack.md) |
| 建图定位 / LiDAR | [state-estimation](../../wiki/concepts/state-estimation.md)、[lidar-sensing](../../wiki/concepts/lidar-sensing.md)、[lidar-odometry-fusion](../../wiki/methods/lidar-odometry-fusion.md) |
| 局部规划与动态避障 | [dwa](../../wiki/methods/dwa.md)、[dynamic-obstacle-filtering](../../wiki/concepts/dynamic-obstacle-filtering.md) |
| Orin NX | [jetson-orin-nx](../../wiki/entities/jetson-orin-nx.md) |
| 语义认知地图 | [embodied-semantic-cognitive-map](../../wiki/concepts/embodied-semantic-cognitive-map.md) |
| 视觉–语言特征融合 | [vision-language-feature-fusion](../../wiki/concepts/vision-language-feature-fusion.md) |
| Habitat | [habitat-sim](../../wiki/entities/habitat-sim.md) |
| SAM3 / BLIP-2 | [paper-sam3](../../wiki/entities/paper-sam3.md)、[paper-blip2](../../wiki/entities/paper-blip2.md) |
| 零样本目标导航 | [zero-shot-object-navigation](../../wiki/tasks/zero-shot-object-navigation.md) |
| Sim2Real | [sim2real](../../wiki/concepts/sim2real.md) |
| 四足本体 | [quadruped-robot](../../wiki/entities/quadruped-robot.md) |
