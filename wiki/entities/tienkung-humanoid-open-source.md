---
type: entity
tags: [humanoid, hardware, open-source, tienkung, x-humanoid]
status: complete
updated: 2026-07-21
related:
  - ./x-humanoid.md
  - ./humanoid-robot.md
  - ./open-source-humanoid-hardware.md
  - ./openloong.md
  - ../overview/robot-open-source-wechat-issue01-curator.md
sources:
  - ../../sources/sites/x-humanoid.md
  - ../../sources/sites/x-humanoid-opensource-cloud.md
  - ../../sources/repos/open-x-humanoid.md
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md
summary: "北京人形机器人创新中心「天工」Lite/Pro：URDF、STEP、ROS、SDK 与二次开发说明分散在门户站与 TienKung_Docs 仓库，需分清主入口。"
---

# 天工 Lite / Pro（开源人形）

## 一句话定义

**天工（TienKung）** Lite / Pro 是**北京人形机器人创新中心（[X-Humanoid](./x-humanoid.md)）**推动的开源人形母平台：云端文档总览、[官网开源页](https://x-humanoid.com/opensource.html) 与 GitHub **[Open-X-Humanoid/TienKung_Docs](https://github.com/Open-X-Humanoid/TienKung_Docs)** 并行分发 URDF、STEP、手册与 SDK。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| URDF | Unified Robot Description Format | Lite/Pro 机器人描述与 mesh |
| SDK | Software Development Kit | 电机 / IMU / 力觉 / 相机接口 |
| ROS | Robot Operating System | 底层 `body_control` 等软件系统 |
| DoF | Degrees of Freedom | 文档称 Lite≈20、Pro 最多≈42 |
| STL | Stereolithography | URDF 配套网格文件格式 |

## 为什么重要

- **「中心制」分发**：结构、电气、ROS、SDK 与上层算法资料并行更新，读者需固定一个「跟踪主分支」的习惯；机构总览见 [X-Humanoid](./x-humanoid.md)。
- **与 ROS / IsaacLab 生态衔接**：底层 ROS 软件系统 + [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) 运控栈；实际部署请核对发行版与支持矩阵。

## 开源入口

| 类型 | 链接 |
|------|------|
| 机构 hub | [X-Humanoid（北京人形）](./x-humanoid.md) |
| 开源门户（文档中心） | [opensource.x-humanoid-cloud.com 文档](https://opensource.x-humanoid-cloud.com/plugin.php?id=zhanmishu_doc:index) |
| 开源资料页 | [x-humanoid.com/opensource.html](https://x-humanoid.com/opensource.html) |
| 文档仓库 | [Open-X-Humanoid/TienKung_Docs](https://github.com/Open-X-Humanoid/TienKung_Docs) |
| URDF / ROS | [TienKung_URDF](https://github.com/Open-X-Humanoid/TienKung_URDF)、[TienKung_ROS](https://github.com/Open-X-Humanoid/TienKung_ROS) |
| 运控训练 | [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) |
| 真机部署 | [Deploy_Tienkung](https://github.com/Open-X-Humanoid/Deploy_Tienkung) |

## 核心原理（文档侧规格摘录）

社区文档中心对两档本体的公开描述（以页面当日为准）：

| 版本 | 文档要点 |
|------|----------|
| **Lite** | 约 **20 DoF**（单臂 4 + 单腿 6 等）；约 **6 km/h** 奔跑；避障 / 上下坡 / 抗冲击；SDK 侧重电机与 IMU |
| **Pro** | 头 3 DoF、单臂 7 DoF；整机最多约 **42 DoF**；SDK 含六维力、IMU、相机接口 |

官网开源页并列发布：Lite/Pro **URDF**、**STEP 结构图纸**、用户手册、SDK 与基于 ROS 的软件系统（`body_control`、`robot_description`、`usb_sbus` 等）。

## 工程实践

1. 先读 [文档中心](https://opensource.x-humanoid-cloud.com/plugin.php?id=zhanmishu_doc:index) 或 `TienKung_Docs` 对应 Lite/Pro PDF，再下载 URDF/STEP。
2. 仿真规划：用 `TienKung_URDF` 进 ROS/Gazebo；RL 运控走 `TienKung-Lab`（Isaac Sim 4.5 / Lab 2.1）。
3. 真机：按手册接 SDK；策略部署参考 `Deploy_Tienkung`；**天工 3.0** 另见 [xhumanoid_sdk](https://github.com/Open-X-Humanoid/xhumanoid_sdk)（ROS 2 Jazzy）。

## 局限与风险

- 入口分散（官网 ZIP、社区文档、多 GitHub 仓）；版本号（2.0 / 3.0 / Lite / Pro）易混。
- 微信策展网盘链接可能失效——以官方 GitHub / 文档中心为准。
- 制造与真机采购门槛远高于桌面 DIY 人形；「开源资料」不等于可低成本整机复刻。

## 关联页面

- [X-Humanoid（北京人形机器人创新中心）](./x-humanoid.md)
- [人形机器人](./humanoid-robot.md)
- [开源人形硬件方案对比](./open-source-humanoid-hardware.md)
- [OpenLoong（青龙·公版机）](./openloong.md)
- [机器人开源宝库（微信策展第01期）索引](../overview/robot-open-source-wechat-issue01-curator.md)

## 推荐继续阅读

- [TienKung_Docs 仓库](https://github.com/Open-X-Humanoid/TienKung_Docs)
- [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)
- [Open-X-Humanoid 组织](https://github.com/Open-X-Humanoid)

## 参考来源

- [sources/sites/x-humanoid.md](../../sources/sites/x-humanoid.md)
- [sources/sites/x-humanoid-opensource-cloud.md](../../sources/sites/x-humanoid-opensource-cloud.md)
- [sources/repos/open-x-humanoid.md](../../sources/repos/open-x-humanoid.md)
- [wechat_jixie_robot_open_source_treasury_issue01_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md)
