---
type: entity
title: Robot Explorer
tags: [utility, kinematics, visualization, education]
summary: "Robot Explorer 是基于 Web 的交互式 3D 机器人探索工具，支持运动学可视化、可操纵性分析及力多胞体展示。"
updated: 2026-05-01
---

# Robot Explorer

**Robot Explorer** 是一个基于 Web 的交互式 3D 机器人探索工具，专注于机器人动力学分析、运动学可视化与教育演示。它由开发者 `ferrolho` 维护，支持在浏览器中直接操控数十种主流机器人模型。

## 核心功能

- **交互式运动学**：支持实时正向 (FK) 和逆向运动学 (IK)。其 IK 引擎支持多末端全身控制，并能自动避开关节限位及零空间约束。
- **可操纵性分析**：可视化机器人在当前位姿下的**可操纵性椭球 (Manipulability Ellipsoids)**，直观展示机器人在速度、加速度和力维度上的性能潜力。
- **工作空间评估**：通过蒙特卡洛采样生成机器人的**可达性点云**，帮助工程师评估任务可行性。
- **力多胞体可视化**：基于 URDF 中定义的关节力矩限制，计算并展示末端执行器在不同方向上的输出力极限。

## 教育与工程价值

- **数学背景说明**：工具内置了详细的数学面板，解释了 IK 求解、雅可比矩阵以及椭球体计算的底层逻辑。
- **广泛的模型支持**：内置了来自 KUKA, ABB, Universal Robots, Franka Emika 等 35+ 品牌的 81+ 机器人 URDF 模型。
- **运动录制**：支持记录关键帧并导出，适合快速原型设计和教学课件制作。

## 技术架构

- **前端**：TypeScript, Vite, **Three.js**。
- **URDF 解析**：使用 `urdf-loader` 加载模型。
- **公式渲染**：集成 KaTeX 提供高质量数学符号展示。

## 关联页面

- [[urdf-studio]] (专业设计工作站)
- [[robot-viewer]] (多格式仿真预览)
- [[pinocchio]] (后端动力学计算库)

## 参考来源
- [Robot Explorer 原始资料](../../sources/repos/robot-explorer.md)
- [Robot Explorer GitHub](https://github.com/ferrolho/robot-explorer)
