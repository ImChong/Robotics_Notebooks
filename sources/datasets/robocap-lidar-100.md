# RoboCap LiDAR-100：BitRobot 头戴 LiDAR 真实工作数据集

> 来源归档（ingest · Hugging Face 数据集官方卡片）

- **标题：** RoboCap LiDAR-100: head-mounted LiDAR, camera and IMU recordings of restaurant, auto repair and car detailing work
- **类型：** dataset
- **发布机构：** BitRobot
- **原文：** <https://huggingface.co/datasets/BitRobot/RoboCap-LiDAR-100>
- **许可证：** CC BY-NC 4.0；下载还须接受数据集访问条件
- **核查日期：** 2026-11-13
- **核对范围：** 已逐节通读官方 Hugging Face 数据卡（Overview、first 50 hours selection、Capture device、Diversity、Contents / Channel contract、Episode table、Download and loading、Citation、Contact）；未因登录门禁下载 MCAP 或验证真实载荷。
- **通读：** 已完成
- **沉淀到 wiki：** [RoboCap LiDAR-100](../../wiki/entities/dataset-robocap-lidar-100.md)
- **一句话说明：** 真实工作现场的头戴 LiDAR、鱼眼 RGB、IMU 第一视角同步数据；目标 100 小时，已发布首批 50 小时。

## 官方发布状态（仅以 2026-11-13 页面为准）

| 项目 | 首批已发布的统计 |
|---|---|
| 已发布时长 | 50 / 100 小时；另 50 小时待完成清理后再发布 |
| 片段及录制场次 | 1,964 个 episodes、132 场 sessions |
| 地点与任务 | 5 个地点：3 个餐厅厨房、1 个汽修店、1 个汽车精洗店；37 类任务、196 个动作标签 |
| 数据体积 | 约 3.4 TB 的 MCAP 载荷（不是 100 小时总量） |
| 已验证的候选池 | 88.5 小时 / 3,657 片段，按任务、动作与场次分层选出 50 小时 |
| 分段时长 | 中位数 61 秒、平均 92 秒、最长 300 秒 |

## 原始传感器与文件

- 固态 LiDAR：每扫描 320×240 距离图（76,800 点），约 125°×89° 视场；cap-01 / cap-02 采样 30 Hz，cap-03 采样 10 Hz。
- 头戴鱼眼 RGB：1600×1200 JPEG，30 fps；IMU：陀螺与加速度 870 Hz；数据卡称多模态使用同一时间轴。
- 单 episode 为一个 zstd / protobuf 编码的 .mcap，含 LiDAR、相机及 IMU 消息；另有 calibration.yaml、metadata/episodes.parquet、sessions.csv、cut_list_50h.csv 和 taxonomy.md。
- 对应通道：LiDAR 的 /rs_lidar/points，相机的 /rs_camera/color/image_raw，IMU 的 /rs_imu。
- **解析差异：** 需要兼容直接 xyz 平面、distance_f32 配合 lidar_angle_vectors.bin 光线表、以及极少数 foxglove.PointCloud 打包帧三种编码；相机在第三种编码中也变为 foxglove.CompressedImage。
- LiDAR 测距无效像素需过滤；数据卡称距离图需旋转 180° 才正向。相机内参、畸变、相机与 LiDAR 外参在每场录制的校准文件中。

## 取数限制与研究边界

HF 仓库可见，但文件访问要求登录同意条件：非商业研究（商业使用须单独许可）、不得识别视频中的人/商业实体/地点、不得再分发原始数据、收到 BitRobot 要求时应删除。官方数据卡示例通过 snapshot_download 的 allow_patterns 先取 metadata，再按选中的 episodes 下载，不推荐整库拉取。

官方卡片将视频到机器人策略预训练列作**可能的应用方向**，但没有提供机器人动作真值、训练完毕的策略或基于该数据集的性能结果。不得将人类演示的 100 小时宣称为已发布的 100 小时可训练机器人动作数据。

## 与其他数据的区别

[RoboMoCap](../../wiki/entities/paper-robomocap.md) 也是 BitRobot 参与的平台，但其录制采用 6 路头戴 RGB、双腕视觉、NOKOV 光学动捕和人体 SMPL-X 标注，目标是全身运动重建与重定向；RoboCap LiDAR-100 是自然工作现场的第一视角度量深度、视觉与惯性流，卡片没有声明提供同类光学动捕真值。

## 一手入口

- [Hugging Face 官方数据卡与下载条件](https://huggingface.co/datasets/BitRobot/RoboCap-LiDAR-100)
- [RoboCap LiDAR-100 wiki 实体](../../wiki/entities/dataset-robocap-lidar-100.md)
