---
type: method
tags: [robotics, motion, retargeting]
status: draft
updated: 2026-11-26
summary: "将人体或其他机器人参考动作映射到目标机器人，并检查运动可行性。"
sources:
  - ../../sources/blogs/wechat_realxiaoze_robot_simulation_stack_2026-09-20.md
related:
  - ../concepts/motion-retargeting-pipeline.md
---

# 机器人动作重定向流程

## 一句话定义

将人体或其他机器人参考动作映射到目标机器人，并检查运动可行性。

## 英文缩写速查

| 缩写 | 英文全称 | 说明 |
|---|---|---|
| API | Application Programming Interface | 组件交互接口 |

## 关联概念

[相关概念](../concepts/motion-retargeting-pipeline.md) 介绍该方法的背景。

## 主要技术路线

1. **对齐本体**：明确输入动作、骨架、坐标系、帧率、关节和目标机器人约束。
2. **求解映射**：可使用运动学优化、几何匹配或学习模型；不同本体不能直接复制关节角。
3. **检查执行**：评估末端误差、接触、速度和动力学可行性，再选择是否用于策略训练或部署。

[GMR](motion-retargeting-gmr.md) 是一种具体方法，不能假定所有重定向项目都采用它。

## 局限与后续核查

本页用于跨项目导航；具体接口、功能与许可应以原始项目资料和实际环境为准。

## 参考来源

- [资料归档](../../sources/blogs/wechat_realxiaoze_robot_simulation_stack_2026-09-20.md)
