---
type: entity
title: Eidon Tracker + POV 数据集
tags: [dataset, egocentric, imu, data-collection, household, manipulation]
status: complete
updated: 2026-10-02
summary: "Eidon 发布的第一视角家庭任务数据集，以 7-IMU 上肢追踪配对 POV 视频；主配对集 1,273.8 小时，另有纯视频集。"
related:
  - ../tasks/teleoperation.md
  - ../queries/humanoid-robot-data-collection-landscape.md
  - ./hiw-500-dataset.md
sources:
  - ../../sources/sites/eidon-ai.md
  - ../../sources/datasets/eidon-tracker-pov.md
  - ../../sources/repos/eidon-ai.md
---

# Eidon Tracker + POV 数据集

**Eidon Tracker + POV** 是一套以可穿戴 **7-IMU 上肢运动追踪**配对第一视角视频的人类日常操作数据；它提供人类动作与视觉观测，不是机器人本体的关节动作数据。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| POV | Point of View | 第一人称视角视频 |
| IMU | Inertial Measurement Unit | 测量方向、角速度与加速度的惯性传感器 |
| CC-BY | Creative Commons Attribution | 需按许可要求署名的数据许可系列 |
| BLE | Bluetooth Low Energy | 可穿戴 Tracker 与主机通信的低功耗蓝牙 |

## 为什么重要

机器人示范数据往往受限于机器人本体、遥操作设备和部署场景。Eidon 的数据路线把传感器戴在人身上，在家庭和工作场所直接记录第一视角视频与上肢运动，降低了“先把机器人搬到每个采集现场”的依赖。它适合研究从人类日常活动中提取视觉、动作和任务结构，再探索如何迁移到机器人。

## 数据集速查

| 子集 | 规模 | 信号与入口 |
|------|------|------|
| Tracker + POV 配对集 | 1,273.8 小时；13,451 段；27 位贡献者 | 7-IMU 上肢运动与第一视角视频配对；[视频/元数据](https://huggingface.co/datasets/eidon-ai/tracker-pov) · [IMU 表](https://huggingface.co/datasets/eidon-ai/tracker-pov-imu) |
| POV 纯视频集 | 305.7 小时；1,370 段；37 位贡献者 | 不带 Tracker 的头戴第一视角视频；[Hugging Face](https://huggingface.co/buckets/eidon-ai/egocentric-pov) |
| 官方汇总 | 1,579 小时；14,821 段；10.6 TB | 官方归档页当前汇总；[Eidon AI](https://www.eidon.ai/) |

主配对集包括 779,010,314 行 IMU；视频约 9.03 TB，IMU 表约 9.54 GB。官方数据卡以 `recording_id` 作为视频与 IMU 关联键。分集贡献者统计分别为 27 和 37，不能假设是互斥群体并相加；归档页另感谢 50 位贡献者。

## 采集系统

Eidon Tracker 在胸部、上臂、前臂和手部布置 7 个 BNO085 9-DOF IMU。项目页标注 24 Hz 采样率、约 1–2° RMS、8–12 小时续航。配套 Eidon Glove 使用 Hall-effect 传感器与嵌入磁体追踪 16 个手指自由度，约 100 Hz BLE。官方公开了 [Tracker](https://github.com/Eidon-AI/eidon-tracker)、[Glove](https://github.com/Eidon-AI/eidon-glove) 和 [Eidon Sim](https://github.com/Eidon-AI/eidon-sim) 仓库，以及介绍完整系统的[白皮书](https://www.eidon.ai/api/whitepaper)。

## 工程实践

1. **先按需选子集：** 数 TB 级视频不适合默认全量下载；先读数据卡和文件索引，可流式读取 IMU 表。
2. **对齐异步信号：** 依据 `recording_id` 连接同一段视频与 IMU，再检查时间戳覆盖、缺帧和传感器槽位定义。
3. **明确目标变量：** Tracker 记录人体上肢运动，不包含机器人关节角、末端力或机器人动作标签；迁移到机器人前仍需人体姿态重建、动作分段和运动重定向。
4. **核对许可与隐私要求：** 数据卡标注 CC-BY-4.0；遵循署名与数据卡中的移除请求说明。代码仓库许可分别以各仓库 LICENSE 为准。

## 局限与风险

- 数据体量达到多 TB，下载与预处理成本高；先检查各分片和字段，再规划存储。
- 人类 IMU 轨迹不是机器人关节命令，需经过姿态估计与重定向，不能直接训练本体控制策略。
- 第一视角家庭数据含真实个人空间影像，复用时应尊重 CC-BY-4.0 条款及数据卡的隐私/移除说明。
- 采集公司及其应用后端已经停止运营；社区可依赖的是归档的硬件、代码与已发布数据。

## 关联页面

- [Teleoperation](../tasks/teleoperation.md) — 对照有人遥操作与可穿戴无机器人示范。
- [HIW-500](./hiw-500-dataset.md) — 对照机器人本体遥操作数据。
- [人形机器人数据采集产业地图](../queries/humanoid-robot-data-collection-landscape.md) — 按采集范式定位 Eidon。
- [人体动作重定向](../concepts/motion-retargeting.md) — 人体运动映射到机器人动作的下游步骤。

## 参考来源

- [Eidon 官方归档项目页](../../sources/sites/eidon-ai.md)
- [Eidon Tracker + POV 数据集归档](../../sources/datasets/eidon-tracker-pov.md)
- [Eidon 开源仓库归档](../../sources/repos/eidon-ai.md)

## 推荐继续阅读

- [Eidon Tracker + POV 数据集卡](https://huggingface.co/datasets/eidon-ai/tracker-pov) — 数据字段、分片和流式加载方式。
- [Eidon Tracker 白皮书](https://www.eidon.ai/api/whitepaper) — 硬件配置与完整追踪系统描述。
- [HIW-500](./hiw-500-dataset.md) — 真机示范数据对照。
