---
type: method
tags: [robotics, simulation, data]
status: draft
updated: 2026-11-26
summary: "为机器人训练或评测构造可运行的场景、物体和任务，并检查几何与物理属性。"
sources:
  - ../../sources/repos/awesome-agentic-robots-survey-index.md
related:
  - ../concepts/robot-simulation-three-layers.md
---

# 机器人仿真场景生成

## 一句话定义

为机器人训练或评测构造可运行的场景、物体和任务，并检查几何与物理属性。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 组件交互接口 |

## 关联概念

- [相关概念](../concepts/robot-simulation-three-layers.md)：了解本方法涉及的背景与边界。

## 为什么重要

明确任务边界、数据来源和验证方式，有助于比较不同系统并避免把背景任务误认为已经完成的实验。

## 主要技术路线

1. **场景与资产构造**：从程序化规则、重建或生成模型得到场景和物体；按任务需要赋予材质、碰撞形状、关节及语义标签。
2. **仿真与任务配置**：把资产导入仿真器，配置机器人、传感器、初始状态及成功判据；生成场景不自动生成可行任务。
3. **检查与迭代**：验证物体布局、接触稳定性、可达性、数据划分与运行时开销，再用于训练或评测。

[生成式数据增强](generative-data-augmentation.md) 侧重样本扩增；场景生成还需要可执行的几何和物理配置。

## 局限与后续核查

本页是跨来源的导航与比较框架，具体实现、指标和适用条件应回到相应论文或项目核对；不能据此推断任一条目已实现全部环节。

## 参考来源

- [来源归档](../../sources/repos/awesome-agentic-robots-survey-index.md)
