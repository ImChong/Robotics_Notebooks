---
type: concept
tags: [humanoid, soccer, simulation, robocup, perception, education]
status: complete
updated: 2026-07-23
related:
  - ../tasks/humanoid-soccer.md
  - ../methods/soccer-field-line-detection.md
  - ../entities/intel-realsense.md
  - ../methods/htwk-gym.md
  - ../entities/booster-robocup-demo.md
  - ../entities/humanoid-system-curriculum.md
  - ../../roadmap/depth-humanoid-soccer.md
sources:
  - ../../sources/courses/shenlan_humanoid_system_theory_practice.md
summary: "足球场仿真环境：为 RoboCup/教学构建含场地线、球门、球体与相机噪声的仿真世界，支撑检测、线定位与踢球闭环实验。"
---

# 足球场仿真环境

## 一句话定义

**足球场仿真环境**是按比赛/教学规格搭建的 **场地几何 + 球/门实体 + 传感器模型** 仿真世界，使检测与定位算法可在无真机场地时迭代——课程第 6.3 节。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RoboCup | Robot Soccer World Cup | 机器人足球赛事与规则源 |
| Sim2Real | Simulation to Real | 视觉域差是足球感知主风险 |
| FOV | Field of View | 相机能否同时看到线与球 |
| Gym | RL Environment API | 部分足球 RL 环境接口 |
| URDF / SDF | Robot/World description | 常用场景描述格式 |

## 为什么重要

- 真机场地预约成本高；检测数据与 EKF 调试应先在可控仿真完成。
- 与 [htwk-gym](../methods/htwk-gym.md) 等 RL 足球环境不同：本概念强调 **感知几何真实感**（线宽、门柱、光照），不限于策略训练。

## 核心原理

仿真最少要素：

1. 场地平面与 **标准线图案**（中圈、禁区、球门线）。
2. 球门网/门柱、足球碰撞体。
3. 机载相机（对齐 [RealSense](../entities/intel-realsense.md) 内参更佳）。
4. 可选：人群/对手、随机光照。

## 工程实践

- 课程线：Gazebo/Webots/自研场 → YOLO 数据采集 → 线交点检测。
- 策略线：可再接到 [Humanoid Soccer](../tasks/humanoid-soccer.md) 与 [纵深路线](../../roadmap/depth-humanoid-soccer.md)。

## 局限与风险

- 纹理过干净会导致检测在真机崩；需域随机化或真机微调。
- 物理引擎球弹跳与真草差距大，踢球策略迁移另论。

## 关联页面

- [场地线检测](../methods/soccer-field-line-detection.md)
- [人形系统课程策展](../entities/humanoid-system-curriculum.md)

## 参考来源

- [深蓝学院人形系统课程大纲](../../sources/courses/shenlan_humanoid_system_theory_practice.md)

## 推荐继续阅读

- RoboCup Humanoid League 规则文档（场地尺寸与线规格）
