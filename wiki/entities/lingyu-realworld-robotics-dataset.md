---
type: entity
project_id: lingyu-realworld-robotics-dataset
project: https://huggingface.co/datasets/LingYu-Robotics/lingyu-realworld-robotics-dataset
tags:
  - dataset
  - manipulation
  - teleoperation
  - imitation-learning
status: complete
updated: 2026-11-05
related:
  - ./lerobot.md
  - ../tasks/teleoperation.md
  - ../methods/imitation-learning.md
sources:
  - ../../sources/datasets/lingyu-realworld-robotics-dataset.md
summary: "灵御真实世界机器人数据集首发 139.4 小时双臂 VR 遥操作数据，采用 LeRobot v2.1。"
---

# LingYu Real-World Robotics Dataset（灵御真实世界机器人数据集）

面向模仿学习的双臂真机遥操作数据，首发涵盖桌面操作；零售环境数据不包含在本次公开集中。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VR | Virtual Reality | 操作者通过头显遥操作 |
| RGB | Red Green Blue | 彩色图像 |
| EEF | End Effector | 机器人末端执行器 |
| IL | Imitation Learning | 从演示学习策略 |

## 数据规模与采集

| 项目 | 数据卡描述 |
|------|------------|
| 规模 | 139.4 小时，13,263 条轨迹，15 类任务、22 子集，约 9.16 TB |
| 本体 | TA2，双 7 自由度机械臂及平行夹爪 |
| 任务 | 叠毛巾、分类、形状插入、积木堆叠等 |
| 观测 | 头部立体相机 3840×1920；左右腕立体相机各 2560×800；状态和动作各 62 维 |
| 时间 | 多路数据同步到 45 Hz |
| 格式 | 每个子目录是独立的 LeRobot v2.1 数据集，Parquet 与 MP4/HEVC |
| 许可 | CC BY-NC-SA 4.0 |

## 数据流程

```mermaid
flowchart LR
  vr["VR 第一视角遥操作"] --> robot["TA2 双臂操作"]
  robot --> streams["相机、状态与动作"]
  streams --> sync["同步至 45 Hz"]
  sync --> qc["人工与自动质量筛查"]
  qc --> data["LeRobot v2.1 子集"]
```

发布方表示会剔除指令跳变、跟踪误差、录制中断和传感流不完整的轨迹；轨迹完整发布，不裁剪。

## 工程使用与局限

- 每个任务目录是独立数据集根目录；加载时应将 root 指向子集，而不是仓库根目录。新版 LeRobot 可能需要使用官方 v2.1 到 v3.0 转换工具。
- 双臂采用速度控制；关节位置目标和速度指令均有记录。夹爪指令在 action 的 39、47 维，夹爪实测位置在 state 的 7、15 维，单位为弧度，不是归一化开度。
- 末端位姿分别相对于对应肩部坐标系，四元数顺序为 x、y、z、w。
- 数据卡报告零售场景轨迹拟经人工审核申请提供，不属于 139.4 小时桌面首发。数据卡没有提供策略评测结果，不能据此推断成功率。
- 数据卡提供下载和读取示例，但未列出该数据集专用的可运行采集或训练实现；源码运行时序图不适用。

## 关联页面

- [LeRobot](./lerobot.md)
- [遥操作](../tasks/teleoperation.md)
- [模仿学习](../methods/imitation-learning.md)

## 参考来源

- [官方数据卡归档](../../sources/datasets/lingyu-realworld-robotics-dataset.md)

## 推荐继续阅读

- [官方数据卡](https://huggingface.co/datasets/LingYu-Robotics/lingyu-realworld-robotics-dataset)
