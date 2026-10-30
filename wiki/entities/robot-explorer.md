---
type: entity
title: Robot Explorer
tags: [utility, kinematics, visualization, education]
summary: "Robot Explorer 是基于 Web 的交互式 3D 机器人探索工具，支持运动学可视化、可操纵性分析及力多胞体展示。"
updated: 2026-10-29
---

# Robot Explorer

**Robot Explorer** 是一个基于 Web 的交互式 3D 机器人探索工具，专注于机器人运动学可视化、可操纵性分析与教育演示。它由开发者 `ferrolho` 维护，支持在浏览器中直接操控 81+ 个 URDF 机器人模型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| IK | Inverse Kinematics | 满足末端/姿态约束求解关节角的运动学逆解 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |

## 核心功能

- **交互式运动学**：支持实时正向 (FK) 和逆向运动学 (IK)。其 IK 支持转动与移动关节、多末端全身 IK，并通过零空间关节限位规避与关节限位锁定保证边界处的稳定表现。
- **可操纵性分析**：可视化机器人在当前位姿下的**可操纵性椭球 (Manipulability Ellipsoids)**，包括速度、加速度和力三类椭球，由雅可比矩阵与关节空间惯性矩阵计算。
- **工作空间评估**：通过随机采样关节构型生成机器人的**可达性点云**，帮助工程师评估任务可行性。
- **力多胞体可视化**：基于 URDF 中定义的关节力矩限制（effort limits），计算并展示末端执行器可达的输出力集合。
- **质心可视化**：根据 URDF 惯性参数显示机器人质心。

## 教育与工程价值

- **数学背景说明**：工具内置了详细的数学面板，解释了 IK 求解、雅可比矩阵以及椭球体计算的底层逻辑。
- **广泛的模型支持**：从独立的模型仓库 [robot-explorer-models](https://github.com/ferrolho/robot-explorer-models) 加载来自 35+ 品牌的 81+ 机器人 URDF 模型（README 截图示例含 Franka Panda、ABB IRB 1200、NASA Valkyrie、Unitree Go2）。
- **运动录制**：支持记录与回放运动关键点，并将其凸包导出为 STL，适合快速原型设计和教学演示。
- **多语言界面**：支持英文、日文与简体中文。

## 技术架构

- **前端**：TypeScript, Vite, **Three.js**, `@tweenjs/tween.js`。
- **URDF 解析**：使用 `urdf-loader` 加载模型。
- **公式渲染**：集成 KaTeX 提供高质量数学符号展示。

## 关联页面

- [urdf-studio](urdf-studio.md) (专业设计工作站)
- [robot-viewer](robot-viewer.md) (多格式仿真预览)
- [pinocchio](pinocchio.md) (后端动力学计算库)

## 参考来源
- [Robot Explorer 原始资料](../../sources/repos/robot-explorer.md)
- [Robot Explorer GitHub](https://github.com/ferrolho/robot-explorer)
