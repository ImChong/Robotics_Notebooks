---
type: entity
title: URDF-Studio
tags: [utility, design, workstation, hardware, ai]
summary: "URDF-Studio 是专业级 Web 机器人设计与组装工作站，支持模块化设计流、硬件 BOM 管理及 AI 辅助建模。"
updated: 2026-05-16
---

# URDF-Studio

**URDF-Studio** 是由 OpenLegged 社区开发的一款专业级** Web 机器人设计与组装工作站**。它不仅是一个查看器，更是一个涵盖了从拓扑设计到硬件物料管理（BOM）的全流程工具。

## 核心功能

- **模块化设计流**：
    - **Skeleton 模式**：可视化编辑机器人的运动树和关节层级。
    - **Detail 模式**：调整碰撞体、视觉模型、惯性张量及网格路径。
    - **Hardware 模式**：关联具体的执行器（电机）元数据，支持生成物料清单。
- **多机器人组装**：支持在同一个工作空间内通过“桥接关节”将多个独立的机器人或附件（如灵巧手、传感器）组合在一起。
- **AI 辅助建模**：集成 AI 能力，支持通过自然语言生成初步机器人结构、检查模型一致性并自动生成分析报告。
- **全生态导出**：支持导出为 URDF, MJCF, USD, SDF, Xacro 等格式，满足不同仿真引擎的需求。

## 工程应用场景

- **硬件原型开发**：通过其 Hardware 模式管理电机选型，并一键生成 CSV/PDF 格式的 BOM 清单供采购使用。
- **快速仿真适配**：将复杂的机器人设计快速转换为适合 [isaac-gym-isaac-lab](isaac-gym-isaac-lab.md) 或 [mujoco](mujoco.md) 的描述文件。
- **团队协作**：提供项目级的工作区管理，方便分享和复用机器人资产。

## 技术架构

- **前端**：React 19, TypeScript, Vite。
- **3D 渲染**：**React Three Fiber** (R3F)。
- **状态管理**：Zustand, Tailwind CSS。

## 关联页面

- [robot-viewer](robot-viewer.md) (模型快速预览)
- [robot-explorer](robot-explorer.md) (运动学分析)
- [open-source-humanoid-hardware](open-source-humanoid-hardware.md) (开源硬件参考)
- [Articraft](./articraft.md)（agent + SDK + harness：可关节 3D 仿真资产生成，与 URDF/网格下游对照）
- [文字生成 CAD（Text-to-CAD）](../concepts/text-to-cad.md)（自然语言 → STEP 等上游几何与机器人描述衔接）

## 参考来源
- [URDF-Studio 原始资料](../../sources/repos/urdf-studio.md)
- [URDF-Studio GitHub](https://github.com/OpenLegged/URDF-Studio)
