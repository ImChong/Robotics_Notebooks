# Eidon AI（归档项目页）

- **标题：** Eidon AI Robotics Data
- **类型：** 项目页 / dataset / wearable hardware
- **项目页：** <https://www.eidon.ai/>
- **GitHub 组织：** <https://github.com/Eidon-AI>
- **Hugging Face 组织：** <https://huggingface.co/eidon-ai>
- **白皮书：** <https://www.eidon.ai/api/whitepaper>
- **状态：** 公司已停止运营；官方归档页仍保留项目与开放资源
- **核查日期：** 2026-10-02
- **数据许可：** 数据集卡标注 CC-BY-4.0；代码/硬件仓库许可按各仓库 LICENSE 为准
- **沉淀到 Wiki：** [Eidon Tracker + POV 数据集](../../wiki/entities/eidon-tracker-pov-dataset.md)

## 一句话摘要

Eidon 将可穿戴上肢 IMU、第一人称视频与家务场景采集流程组合成机器人示范数据管线；关闭后公开 Tracker、Glove、仿真/可视化代码以及相关数据集。

## 官方项目页核查

官方归档页列出 7-IMU Tracker、16-DOF Hall-effect Glove、Eidon Sim、配对 Tracker + POV 数据及纯视频数据，并给出 GitHub 与 Hugging Face 入口。公司声明已经停止运营，数据集以 CC-BY-4.0 发布。

配对数据集当前归档统计为 **1,273.8 小时、13,451 段录制、27 位贡献者**，包含 **779,010,314 行 IMU**；视频约 9.03 TB，IMU 表约 9.54 GB。另有 **305.7 小时、1,370 段**纯 POV 视频。官方首页汇总口径为 **1,579 小时、14,821 段、10.6 TB**。配对集贡献者数与纯视频集贡献者数是各自统计；首页感谢的 50 位贡献者不能据此与分集数字直接相加。

## 系统与开源边界

| 部分 | 公开内容 | 入口 |
|------|----------|------|
| Tracker | 7 × BNO085 9-DOF IMU、胸部/上臂/前臂/手部布置；官方页称 24 Hz、约 1–2° RMS | [eidon-tracker](https://github.com/Eidon-AI/eidon-tracker)；[白皮书](https://www.eidon.ai/api/whitepaper) |
| Glove | Hall-effect 传感器与嵌入磁体，16-DOF 手指关节追踪；约 100 Hz BLE | [eidon-glove](https://github.com/Eidon-AI/eidon-glove) |
| Sim | 轨迹可视化、七自由度手臂运动学与视频同步回放 | [eidon-sim](https://github.com/Eidon-AI/eidon-sim) |
| 配对数据 | 第一人称视频 + 上肢 IMU，使用 recording_id 对齐 | [视频与元数据](https://huggingface.co/datasets/eidon-ai/tracker-pov)、[IMU](https://huggingface.co/datasets/eidon-ai/tracker-pov-imu) |
| 纯视频数据 | 无 Tracker 的头戴 POV 任务视频 | [Hugging Face bucket](https://huggingface.co/buckets/eidon-ai/egocentric-pov) |

数据采集端的 iOS / Android 应用依赖的后端已关闭；现有公开代码、硬件设计和数据不等于托管采集服务仍可用。复用前应查看各仓库最新 LICENSE、数据卡和个人隐私/移除说明。

## 对 Wiki 的映射

- [Eidon Tracker + POV 数据集实体页](../../wiki/entities/eidon-tracker-pov-dataset.md)
- [人形机器人数据采集产业地图](../../wiki/queries/humanoid-robot-data-collection-landscape.md)
