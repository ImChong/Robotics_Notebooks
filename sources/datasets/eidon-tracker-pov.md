# Eidon Tracker + POV 家庭任务数据集

- **标题：** Eidon Tracker POV Dataset
- **类型：** egocentric video + wearable IMU dataset
- **项目页：** [Eidon AI Robotics Data](../sites/eidon-ai.md)
- **视频与元数据：** <https://huggingface.co/datasets/eidon-ai/tracker-pov>
- **IMU 表：** <https://huggingface.co/datasets/eidon-ai/tracker-pov-imu>
- **纯 POV 视频：** <https://huggingface.co/buckets/eidon-ai/egocentric-pov>
- **许可证：** CC-BY-4.0（数据卡）
- **核查日期：** 2026-10-02
- **实体页：** [Eidon Tracker + POV 数据集](../../wiki/entities/eidon-tracker-pov-dataset.md)

## 数据范围

主配对集包含 **1,273.8 小时、13,451 段录制、27 位贡献者**，配有 **779,010,314 行 IMU**。视频数据约 **9.03 TB**，IMU 表约 **9.54 GB**。视频与 IMU 通过 `recording_id` 关联。数据来自日常家庭/工作场所操作任务，项目示例包括洗衣、折叠衣物、清洁、做饭和洗碗等；具体任务分布应以数据卡元数据为准。

Tracker 在胸部、上臂、前臂和手部布置 7 个 IMU，官方页列出的采样率为 24 Hz。数据集并非机器人关节动作轨迹，也不是直接可用于任意机器人控制器的遥操作命令；它提供人类第一视角视觉和上肢运动观测，可用于 egocentric 动作理解、人体运动估计、数据对齐及机器人示范先验研究。

## 独立的纯视频集

官方还发布不带 Tracker 的 **305.7 小时、1,370 段、37 位贡献者、约 1.55 TB** 第一视角视频。归档首页汇总主配对集与纯视频集为 **1,579 小时、14,821 段录制、10.6 TB**。分集贡献者统计口径不同，官方首页感谢的 50 位贡献者不应与 27 和 37 简单相加推断去重人数。

## 许可与使用边界

数据卡标注 **CC-BY-4.0**。引用或再分发时按许可要求署名，并保留数据卡中关于数据来源、移除请求及隐私的说明。数据体量很大；先读数据卡文件布局，再按需流式读取 IMU 或选择视频子集，避免直接下载全部数 TB 视频。

## 来源

- [Hugging Face 视频/元数据数据卡](https://huggingface.co/datasets/eidon-ai/tracker-pov)
- [Hugging Face IMU 数据卡](https://huggingface.co/datasets/eidon-ai/tracker-pov-imu)
- [官方项目页归档](https://www.eidon.ai/)

## 对 Wiki 的映射

- [Eidon Tracker + POV 数据集实体页](../../wiki/entities/eidon-tracker-pov-dataset.md)
- [人形机器人数据采集产业地图](../../wiki/queries/humanoid-robot-data-collection-landscape.md)
