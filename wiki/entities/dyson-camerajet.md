---
type: entity
tags: [consumer-electronics, camera, data-collection, dyson]
status: complete
updated: 2026-09-07
summary: "Dyson CameraJet（2026-09）：带 100k 像素口腔相机与 Gap Optical Targeting AI 的电动牙刷；官方称图像不存云；机器人圈解读为潜在臂部运动数据采集形态。"
related:
  - ../queries/humanoid-robot-data-collection-landscape.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/blogs/leoinai_humanoid_robot_datacollection_2026-09-06.md
---

# Dyson CameraJet

**Dyson CameraJet™**（2026-09 发布，约 **$499**）是带 **100k 像素宏距口腔相机** 与 **Gap Optical Targeting™** 机器学习算法的电动牙刷：以约 **28 fps** 识别齿缝并 **100 ms 内** 喷射漱口水精冲。开发期采集约 **47 万** 张牙科图像训练模型。

## 一句话定义

**把相机和 AI 塞进牙刷做齿缝检测——消费品赛道里极高频的「手–臂–口」运动，被机器人社区当作潜在数据采集形态讨论。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AI | Artificial Intelligence | Gap Optical Targeting 齿缝检测 |
| IoT | Internet of Things | Wi-Fi 2.4G + 蓝牙连接 MyDyson App |
| POV | Point of View | 口腔内窥镜式 live view |
| IL | Imitation Learning | 若数据用于机器人，属推测性下游 |
| SDK | Software Development Kit | 无公开机器人数据 SDK |

## 为什么重要（机器人视角）

- **Substack 读法：** 高频刷牙动作 ≈ 丰富 **腕部轨迹 + 接触** 统计；与 [Shift](./shift-app-nyc.md)「服务换数据」同属 **消费品侧数据面**。
- **官方隐私立场：** Dyson 称 live view / auto-jet 时 **不录制、不上传** 图像；与「数据收割」叙事需区分——**以官方政策为准**。

## 核心信息

| 项 | 内容 |
|----|------|
| **发布** | 2026-09-01；巴黎发布 |
| **传感** | 100k pixel macro camera；16M 行代码（官方宣传） |
| **连接** | MyDyson App live view；固件/使用统计可上云 |
| **机器人数据** | **无官方机器人训练计划**；产业评论为推测 |

## 关联页面

- [人形数据采集地图](../queries/humanoid-robot-data-collection-landscape.md)

## 参考来源

- [LeoInAI Substack 摘录](../../sources/blogs/leoinai_humanoid_robot_datacollection_2026-09-06.md)
- [Dyson 官方发布](https://www.dyson.com/discover/news/latest/introducing-camerajet)
