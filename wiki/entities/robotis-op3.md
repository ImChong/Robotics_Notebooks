---
type: entity
tags: [humanoid, hardware, open-source, robotis, ros, education]
status: complete
updated: 2026-09-05
related:
  - ./robotis.md
  - ./robotis-ai-sapiens.md
  - ./robotis-mujoco-menagerie.md
  - ./humanoid-robot.md
  - ../overview/robot-open-source-wechat-issue01-curator.md
  - ./robotis-thormang3.md
  - ../concepts/ros2-basics.md
sources:
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md
  - ../../sources/repos/robotis-git.md
summary: "ROBOTIS OP3：小型人形开源平台，文档与三维模型在 eManual；手册源码树位于 ROBOTIS-GIT/emanual 的 OP3 目录。"
---

# ROBOTIS OP3

## 一句话定义

**ROBOTIS OP3** 是 **ROBOTIS** 的 **小型人形** 开源平台：硬件结构、电气与传感器接口说明在 **[OP3 eManual](https://emanual.robotis.com/docs/en/platform/op3/introduction/)**；官方手册 Markdown 源码在 GitHub **[emanual …/op3](https://github.com/ROBOTIS-GIT/emanual/tree/master/docs/en/platform/op3)** 树下。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ROS 2 | Robot Operating System 2 | 机器人系统集成与通信的常用中间件 |

## 为什么重要

- **教育与研究人形基线**：在 **ROS** 课程与小型比赛里长期作为 **可编程全身关节** 的参考机。
- **与 THORMANG3 对照**：同厂商 **大尺寸** 与 **桌面级** 人形的软件栈与维护节奏不同，升级前需分别核对。

## 开源入口（策展摘录）

| 类型 | 链接 |
|------|------|
| eManual | [OP3 introduction](https://emanual.robotis.com/docs/en/platform/op3/introduction/) |
| 硬件 / 三维模型 | [OP3 hardware](https://emanual.robotis.com/docs/en/platform/op3/hardware/) |
| 手册源码树 | [ROBOTIS-GIT/emanual/docs/en/platform/op3](https://github.com/ROBOTIS-GIT/emanual/tree/master/docs/en/platform/op3) |

## 机制与架构图

```mermaid
flowchart LR
    D["OP3 eManual 与手册源码 (硬件 / 3D 模型 / 电气接口)"] --> P["ROBOTIS OP3 小型人形平台"]
    P --> R["ROS / ROS 2 集成与全身关节编程"]
    R --> E["教育课程与小型机器人比赛"]
```

## 关联页面

- [ROBOTIS 组织 hub](./robotis.md)
- [AI Sapiens](./robotis-ai-sapiens.md)（新一代人形对照）· [MuJoCo Menagerie](./robotis-mujoco-menagerie.md)
- [人形机器人](./humanoid-robot.md)
- [ROBOTIS THORMANG3](./robotis-thormang3.md)
- [ROS 2 基础](../concepts/ros2-basics.md)
- [机器人开源宝库（微信策展第01期）索引](../overview/robot-open-source-wechat-issue01-curator.md)
- [Humanoid Locomotion (人形机器人移动)](../tasks/humanoid-locomotion.md)：ROBOTIS OP3 作为可编程全身关节的小型开源人形平台，常用于教育课程与小型比赛中的双足运动研究。
- [Humanoid Soccer](../tasks/humanoid-soccer.md)：该小型人形平台具备完整的开源硬件资料与 ROS 接口，是人形机器人足球竞赛与教学场景的典型载体。

## 推荐继续阅读

- [ROBOTIS-GIT 组织](https://github.com/ROBOTIS-GIT)（检索 `op3` 相关仓库）

## 参考来源

- [wechat_jixie_robot_open_source_treasury_issue01_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md)
- [sources/repos/robotis-git.md](../../sources/repos/robotis-git.md)
