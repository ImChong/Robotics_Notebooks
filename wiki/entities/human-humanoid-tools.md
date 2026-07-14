---
type: entity
tags: [humanoid, motion-retargeting, open-source, roboparty, dataset, newton-physics, teleoperation]
status: complete
updated: 2026-07-14
related:
  - ./party-os.md
  - ../overview/roboparty-lab-party-os-technology-map.md
  - ../concepts/motion-retargeting.md
  - ./newton-physics.md
  - ./mimiclite.md
  - ../queries/humanoid-training-data-pipeline.md
  - ../tasks/teleoperation.md
sources:
  - ../../sources/repos/human_humanoid_tools.md
  - ../../sources/blogs/wechat_roboparty_lab_party_os_3_tools.md
summary: "human-humanoid-tools（hhtools）是 RoboParty 开源的 Human-to-Humanoid 动作重定向工作台：Newton IK 与 Interaction-Mesh 双后端、约 30 秒级复杂全身 retarget、Any Motion/Any URDF/R2R 与一体化数据分析和 3D 可视化。"
---

# human-humanoid-tools（hhtools）

**human-humanoid-tools**（简称 **hhtools**）是 [Party OS](./party-os.md) 首批开源的 **Human-to-Humanoid 动作重定向与数据工作台**，面向 Web 前端全程操作，目标在约 **30 秒** 内完成单段复杂全身动作（跑酷、舞蹈、物体交互等）的重定向，并支持任意标准 URDF、主流开源数据集格式与 **机器人到机器人（R2R）** 动作互转。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IK | Inverse Kinematics | 逆运动学，满足姿态/末端约束的关节解算 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| R2R | Robot to Robot | 机器人到机器人动作迁移通道 |
| BVH | BioVision Hierarchy | 常用动捕骨骼动画格式 |
| SMPL | Skinned Multi-Person Linear Model | 参数化人体网格模型 |
| MPC | Model Predictive Control | 模型预测控制，Interaction-Mesh 后端求解 |

## 为什么重要

- **上游摩擦最大环节：** [Motion Retargeting](../concepts/motion-retargeting.md) 常是模仿学习、跟踪训练与遥操的瓶颈；hhtools 把「每数据集写脚本、每 URDF 写适配」推向 **拖放式工作台**。
- **速度：** 文称单段复杂全身动作约 **30 秒** 完成完整重定向，支持 **批量并行**，大幅压缩动作调试周期。
- **R2R 差异化：** 不仅 Human→Robot，还支持 **Robot→Robot**，把 A 机型成熟动作库迁移到结构差异较大的 B 机型，缓解「目标机型无数据集」问题。

## 核心原理

### 1. Fast Retarget：双后端架构

| 后端 | 技术 | 特点 |
|------|------|------|
| **Newton IK** | [Newton](./newton-physics.md) / Warp 可并行 | 轻量化高速 IK |
| **Interaction-Mesh** | MPC solver | 交互网格约束，适合接触丰富动作 |

- **目标：** 时序平滑、无关节突变、减少滑脚失真。
- **场景：** 单段地形跑酷、舞蹈、物体交互等全身复杂动作；支持批量并行 retarget。

### 2. Any Motion：数据集格式自动识别

兼容并可视化市面上绝大部分开源动作格式，**自动识别**包括但不限于：

| 类型 | 示例 |
|------|------|
| 文件格式 | BVH、GLB、SMPL |
| 数据集 | bvh Mocap、AMASS、GVHMR、LAFAN1、OMOMO、PHUMA、Intermimic、Meshmimic |

降低「换数据集就换一套预处理脚本」的维护成本。

### 3. Any URDF：零定制机型适配

- 开发者拖入机器人 **URDF + Mesh 文件夹** 即可。
- **自动识别** 标准 URDF 人形模型，无需为不同机器人编写定制适配代码。
- 打破传统 retarget 工具与 **单一机型绑定** 的限制。

### 4. Robot → Robot（R2R）

- 将一款人形机器人的成熟动作库 **直接迁移** 至另一款结构差异较大的机器人。
- 依托 **统一骨骼对齐管线**，解决「目标机型找不到合适数据集」的常见问题。
- 与 Human→Robot 形成 **双向动作资产流动**。

### 5. Dataset Analysis & Visualize

一体化运动数据分析与 3D 可视化模块：

| 功能 | 用途 |
|------|------|
| 关节轨迹曲线 | 质检异常帧与关节限位 |
| 重心变化 | 动态稳定性筛选 |
| 接触点热力图 | 接触丰富动作分析 |
| 条件筛选 | 例：从 1000 条数据中筛水平速度 3 m/s 片段 |
| 分类导出 | 一站式预览、质检、导出训练集 |

## 工程实践

| 步骤 | 建议 |
|------|------|
| 导入 | Web 前端上传动作文件（或选内置数据集）+ 目标 URDF |
| 后端选择 | 接触简单动作用 Newton IK；交互/接触丰富场景试 Interaction-Mesh |
| 质检 | 用内置可视化与运动学指标筛选；导出前检查滑脚与穿透 |
| 下游 | 导出统一格式供 [MimicLite](./mimiclite.md) 跟踪训练或外部 RL 管线 |
| R2R | 源机器人动作库 → 目标 URDF，无需重新采集人类演示 |

**仓库：** [github.com/Roboparty/human-humanoid-tools](https://github.com/Roboparty/human-humanoid-tools)

## 局限与风险

- **物理可行性：** 30s 级运动学 retarget 不保证动力学可跟踪；高动态动作仍需下游仿真/RL 修正（见 [Motion Retargeting](../concepts/motion-retargeting.md) 常见误区）。
- **Web 前端依赖：** 全程 Web 操作的体验与离线批处理需求须以仓库实际能力为准。
- **自动识别边界：** 「绝大部分格式」以维护列表为准；冷门私有格式可能仍需扩展。
- **与学术 SOTA 对比：** TopoRetarget、OmniRetarget 等交互保留方法在特定 HOI 任务上可能有不同 trade-off。

## 关联页面

- [Party OS](./party-os.md)
- [MimicLite](./mimiclite.md) — 下游跟踪训练
- [Motion Retargeting](../concepts/motion-retargeting.md)
- [Newton Physics](./newton-physics.md)
- [人形训练数据管线](../queries/humanoid-training-data-pipeline.md)
- [Teleoperation](../tasks/teleoperation.md)

## 参考来源

- [human_humanoid_tools.md](../../sources/repos/human_humanoid_tools.md)
- [wechat_roboparty_lab_party_os_3_tools.md](../../sources/blogs/wechat_roboparty_lab_party_os_3_tools.md)

## 推荐继续阅读

- [human-humanoid-tools GitHub](https://github.com/Roboparty/human-humanoid-tools)
- [Motion Retargeting 概念页](../concepts/motion-retargeting.md)
- [RoboParty Lab / Party OS 技术地图](../overview/roboparty-lab-party-os-technology-map.md)
