---
type: entity
project_id: robocap-lidar-100
project: https://huggingface.co/datasets/BitRobot/RoboCap-LiDAR-100
tags: [dataset, egocentric, lidar, point-cloud, rgb, imu, manipulation, bitrobot]
status: complete
updated: 2026-11-13
original:
  - ../../sources/datasets/robocap-lidar-100.md
sources:
  - ../../sources/datasets/robocap-lidar-100.md
related:
  - ./paper-robomocap.md
  - ./dataset-ego-exo4d.md
  - ./awesome-egocentric-vision.md
  - ../tasks/manipulation.md
summary: "BitRobot RoboCap LiDAR-100 是头戴固态 LiDAR、鱼眼 RGB 与 IMU 的真实工作现场第一视角数据集；目标 100 小时，目前开放首批 50 小时（1,964 片段、132 场录制、约 3.4 TB），访问须同意非商业研究条款。"
---

# RoboCap LiDAR-100：100 小时头戴 LiDAR 真实工作数据集

## 一句话定义

**RoboCap LiDAR-100** 是 BitRobot 发布的真实工作场景第一视角多传感器数据集，以头戴固态 LiDAR、鱼眼 RGB 相机和 IMU 记录厨房、汽修和汽车精洗操作；**100 小时是计划总规模，2026-11-13 已开放的是前 50 小时**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| LiDAR | Light Detection and Ranging | 头戴固态测距传感器，每帧形成有物理尺度的距离图与点云 |
| RGB | Red, Green, Blue | 1600×1200 鱼眼彩色图像 |
| IMU | Inertial Measurement Unit | 870 Hz 加速度和角速度采样 |
| MCAP | Message Capture | 按 episode 保存同步多传感器消息的容器格式 |
| QA | Quality Assurance | 数据选取前的载荷摘要与质量核验 |
| VLA | Vision-Language-Action | 潜在下游机器人策略训练方向，并非此数据集中已有机器人动作标签 |

## 为什么重要

只有第一视角视频时，工作台面、工具和手部的深度信息常需由视觉模型估计。此数据集以真实工作场景的同步度量深度和惯性流提供补充，可用于第一视角场景理解、动作识别、视频到机器人策略的预训练研究。但数据发布不等于已有机器人动作真值或经验证的机器人策略收益。

## 数据集速查

| 维度 | 截至 2026-11-13 官方数据卡披露 |
|---|---|
| 机构 | BitRobot |
| 发布目标 / 当前开放 | 100 小时 / **50 小时**；剩余 50 小时待发布 |
| 规模 | 1,964 个任务片段、132 场录制、5 个工作地点、约 3.4 TB MCAP |
| 任务覆盖 | 3 个餐厅厨房、1 个汽修店、1 个汽车精洗店；37 类任务、196 个动作 |
| LiDAR | 320×240 = 76,800 点/扫描，约 125°×89°，10 或 30 Hz |
| 摄像头 | 鱼眼 RGB，1600×1200 JPEG，30 fps |
| IMU | 陀螺仪与加速度计，870 Hz |
| 数据组织 | MCAP + Parquet/CSV 元数据 + 每场标定 YAML |
| 许可证 | CC BY-NC 4.0；另需同意数据访问协议 |
| 数据性质 | 人类真实作业第一视角观测，不是光学动捕真值或机器人关节动作数据 |

## 核心原理与流程总览

该图依据官方数据卡的 Capture device、Selection 和 Contents：**记录与选择是实际的数据组织过程**，不代表已训练出机器人控制策略。

\`\`\`mermaid
flowchart TD
  Work["厨房、汽修与汽车精洗真实作业"] --> Capture["头戴同步采集"]
  Capture --> Lidar["固态 LiDAR 10 或 30 Hz"]
  Capture --> Camera["鱼眼 RGB 30 fps"]
  Capture --> IMU["IMU 870 Hz"]
  Lidar --> QC["原始录制验证与 QA"]
  Camera --> QC
  IMU --> QC
  QC --> Pool["88.5 小时已核验候选池"]
  Pool --> Select["按任务类别、动作标签和录制场次分层选取"]
  Select --> Release["首批 50 小时 · 1964 片段"]
  Release --> MCAP["每片段一个 MCAP"]
  Release --> Metadata["episodes.parquet 与 sessions.csv"]
  Release --> Calib["每场 calibration.yaml"]
  MCAP --> Research["第一视角深度、操作理解和策略预训练研究"]
  Metadata --> Research
  Calib --> Research
\`\`\`

- 固态 LiDAR 每帧输出 320×240 距离图，数据卡给出三种消息编码：原始 xyz 平面、distance_f32 加光线方向附件、少数 foxglove.PointCloud 打包数据。读取时必须分支处理，不能假设全部 MCAP 使用同一 protobuf schema。
- 标定文件给出鱼眼相机内参与畸变、LiDAR 视场和相机到 LiDAR 的外参。头戴传感器向前下方取景，手与工作台居多；头后方和远角落覆盖有限。
- 官方以 88.5 小时已核验的池子为基础，保留任务与动作标签多样性后选出首批 50 小时；余下 50 小时尚未列为已开放。

## 工程实践

1. 在 [HF 数据集页](https://huggingface.co/datasets/BitRobot/RoboCap-LiDAR-100) 登录并同意数据使用条款，再按官方 README 用 \`snapshot_download(repo_type="dataset", allow_patterns=...)\` 获取 \`metadata/*\` 与 \`calibration/*\`。
2. 按 \`metadata/episodes.parquet\` 选定任务、LiDAR 采样率及片段，只下载对应 \`episodes/<session_id>/<episode_id>.mcap\`，避免直接拉取约 3.4 TB。
3. 使用 \`mcap\` 与 \`mcap-protobuf-support\` 读取 \`/rs_lidar/points\`、\`/rs_camera/color/image_raw\`、\`/rs_imu\`；按原生 xyz、distance + \`lidar_angle_vectors.bin\` 和 foxglove 打包格式分别解码。
4. 处理无效深度和图像旋转，使用 \`calibration.yaml\` 做跨传感器几何对齐；不能直接将相机像素与 LiDAR 点视为已配准。
5. 这里仅有官方数据卡中的 **Python 加载片段**，未发布独立训练/推理系统及运行入口，因此**源码运行时序图不适用**。本次未下载原始 MCAP，也未验证数值解码。

## 局限与风险

- **许可与访问：** CC BY-NC 4.0；文件需要同意非商业研究条款，不得识别个人、商家或地点，不得再分发原始数据；商业用途需另行授权。
- **发布阶段：** 首批 50 小时 / 3.4 TB，剩余 50 小时仍待开放；不能写为 100 小时已可下载。
- **标注边界：** 有环境、场景、动作与文本描述，未给出机器人关节轨迹、受力真值或光学全身动捕；不能直接用来监督关节动作。
- **读取兼容：** 录制间 LiDAR 消息格式及频率不同；无效距离像素、相机鱼眼畸变、坐标变换都需要处理。
- **验证范围：** 本页通读官方数据卡；未申请实际文件访问、下载 3.4 TB 载荷或独立验证原始数据质量。

## 与其他数据集对比

| 数据集 | 数据获取方式 | 关键差异 |
|---|---|---|
| [RoboMoCap](./paper-robomocap.md) | 光学动捕棚 + 六路头戴视觉 + 双腕相机 | 有全身光学动捕与 SMPL-X，对人形运动重定向更直接；不是同一个数据集 |
| [Ego-Exo4D](./dataset-ego-exo4d.md) | 大规模第一/第三视角视频 | 覆盖丰富人类活动；本数据集突出工作现场头戴 LiDAR 度量深度 |

## 关联页面

- [RoboMoCap：带全身光学真值的第一视角数据](./paper-robomocap.md)
- [Ego-Exo4D](./dataset-ego-exo4d.md)
- [第一视角视觉资料](./awesome-egocentric-vision.md)
- [机器人操作任务](../tasks/manipulation.md)

## 参考来源

- [BitRobot RoboCap LiDAR-100 原始数据卡归档与阅读记录](../../sources/datasets/robocap-lidar-100.md)
- [Hugging Face 官方数据集卡](https://huggingface.co/datasets/BitRobot/RoboCap-LiDAR-100)

## 推荐继续阅读

- [官方数据集卡：Contents / Channel contract / Download and loading](https://huggingface.co/datasets/BitRobot/RoboCap-LiDAR-100)
- [RoboMoCap：不同传感器与标注路线](./paper-robomocap.md)
