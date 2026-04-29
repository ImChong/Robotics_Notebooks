---
type: entity
title: Robot Viewer
tags: [utility, simulation, mujoco, xacro]
---

# Robot Viewer

**Robot Viewer** 是由开发者 `fan-ziqi` 开发的一个全功能 Web 机器人模型查看与仿真平台。它最大的特点是支持多种主流机器人描述格式，并能直接在浏览器中运行物理仿真。

## 核心功能

- **多格式支持**：
    - **URDF**: 标准机器人描述。
    - **Xacro**: 支持原生 Xacro 宏展开，无需 ROS 环境即可解析复杂的参数化模型。
    - **MJCF**: 支持 MuJoCo 模型文件。
    - **USD**: 支持 NVIDIA Omniverse 的标准格式。
- **集成 MuJoCo 仿真**：利用 MuJoCo WASM 技术，支持一键将 MJCF 模型转换为可交互的动力学仿真环境。
- **实时代码编辑**：集成 CodeMirror 编辑器，允许用户实时修改模型源码并立即查看 3D 渲染效果。
- **测量工具**：提供精确的测距仪，用于测量关节中心距离、链接长度以及相对于地面的高度。

## 独特优势

- **轻量级 Xacro 处理**：它是目前少数几个能够在 Web 端独立完成 Xacro 逻辑（宏、条件判断、数学运算）解析的工具。
- **调试利器**：非常适合算法工程师快速检查模型结构、碰撞体定义以及惯性参数是否正确。

## 技术架构

- **核心引擎**：Three.js (渲染), MuJoCo WASM (物理)。
- **工具链**：Vite, JavaScript, `xacro-parser`。

## 关联页面

- [[robot-explorer]] (动力学分析工具)
- [[urdf-studio]] (机器人组装工作站)
- [[mujoco]] (底层仿真引擎)

## 参考来源
- [Robot Viewer 原始资料](../../sources/repos/robot-viewer.md)
- [Robot Viewer GitHub](https://github.com/fan-ziqi/robot_viewer)
