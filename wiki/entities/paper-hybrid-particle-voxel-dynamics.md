---
type: entity
project_id: hybrid-particle-voxel-dynamics
tags:
  - paper
  - world-model
  - perception
status: complete
updated: 2026-11-05
arxiv: "2611.04492"
project: https://jwang771.github.io/3dworldsimulator/
code: https://github.com/jwang771/3d-world-simulator
related:
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-11-04-57-papers-technology-map.md
  - ../queries/robot-perception-stack-selection-loop.md
  - ../overview/embodied-station-2026-11-05-43-papers-technology-map.md
sources:
  - ../../sources/papers/hybrid_particle_voxel_dynamics_arxiv_2611_04492.md
  - ../../sources/repos/hybrid_particle_voxel_dynamics.md
  - ../../sources/sites/hybrid-particle-voxel-dynamics.md
  - ../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md
  - ../../sources/blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md
summary: "Hybrid Particle-Voxel Dynamics（arXiv:2611.04492）：以粒子与体素混合表示建立动作条件世界模型，联合描述物体形变、触觉与多视角观测；面向操作任务，重点关注长时预测的物理一致性。"
---

# Hybrid Particle-Voxel Dynamics（arXiv:2611.04492）

**Hybrid Particle-Voxel Dynamics**（*Hybrid Particle-Voxel Dynamics for Multimodal 3D World Simulation*，[arXiv:2611.04492](https://arxiv.org/abs/2611.04492)，[项目页](https://jwang771.github.io/3dworldsimulator/)，[代码](https://github.com/jwang771/3d-world-simulator)）来自 [具身智能小站 2026-11-04 盘点](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)，属「世界模型与感知」组。

## 一句话定义

**以粒子与体素混合表示建立动作条件世界模型，联合描述物体形变、触觉与多视角观测；面向操作任务，重点关注长时预测的物理一致性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Hybrid Particle-Voxel Dynamics | Hybrid Particle-Voxel Dynamics for Multimodal 3D World Simulation | 本文方法 / 基准的简称 |
| 3D | 3D | 三维 |

## 为什么重要

- 让世界模型同时预测几何与触觉。
- 开源结论：**已开源**（步骤 2.5，2026-11-04）。仓库可达（git ls-remote）。项目页 HTTP 200。
- 同期同组工作见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2611.04492](https://arxiv.org/abs/2611.04492) |
| **开源** | **已开源** |
| **要点** | 以粒子与体素混合表示建立动作条件世界模型，联合描述物体形变、触觉与多视角观测；面向操作任务，重点关注长时预测的物理一致性。 |
| **文内指标** | 公众号未给具体数字，本页不引用。 |

## 源码运行时序图

以下为按仓库入口（`jwang771/3d-world-simulator`）推断的通用运行链路；入库日仅核实仓库可达，未逐文件阅读，具体脚本名以 README 为准。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as jwang771/3d-world-simulator
  participant E as 环境 / 数据
  participant M as 模型 / 策略
  U->>R: clone + 安装依赖（README）
  U->>E: 准备基准 / 数据集或真机接口
  U->>R: 训练或加载预训练权重
  R->>M: 构建 Hybrid Particle-Voxel Dynamics 模型
  U->>R: 运行评测脚本
  R->>E: 重置环境、取观测
  E-->>M: 观测
  M-->>E: 动作 / 预测
  R-->>U: 指标与日志
```

## 实验与评测

- 公众号未给具体数字，本页不引用。
- **读法：** 本页为索引级摘要，依据公众号转述的官方摘要与仓库信息整理；逐项数字与 baseline 以原文 PDF 为准。

## 与其他工作对比

- 横向索引见 [VLM 奖励评估、动作动力学与开放权重机器人智能体：具身智能小站 11-04 期 57 篇阅读坐标](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)；同 arXiv 不重复造页。

## 结论

**让世界模型同时预测几何与触觉：以粒子与体素混合表示建立动作条件世界模型，联合描述物体形变、触觉与多视角观测；面向操作任务，重点关注长时预测的物理一致性。**

1. 开源边界：**已开源** — 仓库可达（git ls-remote）。项目页 HTTP 200。
2. 核心机制：以粒子与体素混合表示建立动作条件世界模型，联合描述物体形变、触觉与多视角观测；面向操作任务，重点关注长时预测的物理一致性。
3. 复现与选型前核对任务协议、硬件与基准口径，勿直接横比公众号摘录数字。

## 关联页面

- [world-action-models](../concepts/world-action-models.md)
- [generative-world-models](../methods/generative-world-models.md)
- [vla](../methods/vla.md)
- [embodied-station-2026-11-04-57-papers-technology-map](../overview/embodied-station-2026-11-04-57-papers-technology-map.md)
- [机器人视觉感知栈选型闭环（知识链）](../queries/robot-perception-stack-selection-loop.md)

## 参考来源

- [hybrid_particle_voxel_dynamics_arxiv_2611_04492.md](../../sources/papers/hybrid_particle_voxel_dynamics_arxiv_2611_04492.md)
- [wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md](../../sources/blogs/wechat_embodied_station_57_papers_da_reward_model_failure_2026-11-04.md)
- [arXiv:2611.04492](https://arxiv.org/abs/2611.04492)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.04492)
- [项目页](https://jwang771.github.io/3dworldsimulator/)
- [https://github.com/jwang771/3d-world-simulator](https://github.com/jwang771/3d-world-simulator)
