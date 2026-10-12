# Eidon Tracker + POV 家庭任务数据集

- **标题：** Eidon Tracker POV Dataset
- **类型：** egocentric video + wearable IMU dataset
- **项目页：** [Eidon AI Robotics Data](../sites/eidon-ai.md)
- **配对视频与元数据：** <https://huggingface.co/datasets/eidon-ai/tracker-pov>
- **配对 IMU 表：** <https://huggingface.co/datasets/eidon-ai/tracker-pov-imu>
- **纯 POV 视频：** <https://huggingface.co/buckets/eidon-ai/egocentric-pov>
- **许可证：** 三份数据资源的卡片/说明均标注 CC-BY-4.0
- **核查日期：** 2026-10-11
- **实体页：** [Eidon Tracker + POV 数据集](../../wiki/entities/eidon-tracker-pov-dataset.md)

## 数据范围与拆分

配对集包含 **1,273.8 小时、13,451 段录制、27 位贡献者**；视频为 MP4，1080p 至 4K、以 30 fps 为主，并提供逐录制 metadata.parquet。IMU 长表约 **779M 行**，通过 recording_id 与视频/元数据关联。场景来自参与者在住所等真实环境完成日常操作，示例包括洗衣、清洁、洗碗和烹饪。

纯视频集不带 Tracker：**约 306 小时、1,370 段、37 位贡献者、1.55 TB**。它以 Storage Bucket 发布，不能直接使用标准 datasets.load_dataset；需用 Hugging Face 的 hf buckets 工具或兼容对象存储方式浏览/下载。数据卡说明纯视频与配对集有 14 位相同贡献者，contributor_id 保持一致；跨子集拆分训练与测试时应按贡献者分组，避免身份泄漏。

## 文件结构与元数据

- tracker-pov 提供 recordings/metadata.parquet（每段录制一行）和 recordings/data/000 至 013 下的 MP4。
- tracker-pov-imu 提供相同录制的 24 Hz 长格式 Parquet IMU 流。
- egocentric-pov bucket 提供纯视频、metadata.parquet 和 users.parquet。
- 数据卡列出的元数据包括文件名、recording_id、contributor_id、时长、任务类型、录制时间及 qc_status 等。

建议先按任务和质量状态筛选 metadata，再读取目标录制的视频或 IMU 分片。配对视频与 IMU 用 recording_id 关联；按需下载，避免一次性同步数 TB 原始视频。具体脚本和 hf CLI 参数以各数据卡当前版本为准。

## 质量控制

官网称历史 Eidon Sym 管线曾运行 VLM 驱动质量控制、自动过滤和标记录制，并输出仿真兼容格式。纯视频 bucket 数据卡列出手部可见性、光照、模糊和相机稳定性检查，统计 qc_status 为 1,059 valid、241 flagged、70 invalid。这些是自动化质量标记，不等于人工逐帧审核。

## 信号含义、许可与限制

Tracker 在胸部、上臂、前臂和手部放置 7 个 9 轴 IMU，官方标注采样率 24 Hz。记录的是人体姿态观测和第一视角画面，不是机器人关节角、末端力或控制指令。迁移到机器人任务仍需姿态重建、任务分段、运动重定向和机器人侧验证。

三份数据资源标注 CC-BY-4.0。发布或再分发衍生数据时保留署名和来源说明，并遵守数据卡关于参与者隐私、同意和移除请求的说明。官网列 IMU 约 9.54 GB；Hugging Face 卡片概览写约 9.5 GB，结构表另写约 6.6 GB，故实际体量以当前文件索引为准。

## 来源

- [Tracker POV 数据卡](https://huggingface.co/datasets/eidon-ai/tracker-pov)
- [Tracker POV IMU 数据卡](https://huggingface.co/datasets/eidon-ai/tracker-pov-imu)
- [纯 POV Storage Bucket 说明](https://huggingface.co/buckets/eidon-ai/egocentric-pov)
- [官方项目页归档](https://www.eidon.ai/)

## 对 Wiki 的映射

- [Eidon Tracker + POV 数据集实体页](../../wiki/entities/eidon-tracker-pov-dataset.md)
- [人形机器人数据采集产业地图](../../wiki/queries/humanoid-robot-data-collection-landscape.md)
