# 图解 IMU：原理、算法与摄像头驱动协同

> 来源归档（blog / 微信公众号）

- **标题：** 图解 IMU：原理、算法与摄像头驱动协同
- **类型：** blog（工程教程 / 原理图解）
- **作者：** 飞一样的成长（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/gbA1Crm94uG_5hqWati4Kw
- **发表日期：** 2026-09-05
- **入库日期：** 2026-09-19
- **抓取方式：** wechat-article-for-ai（Camoufox）
- **原始抓取落盘：** [`sources/raw/wechat_feiyiyangcheng_imu_principles_algorithms_camera_2026-09-05.md`](../raw/wechat_feiyiyangcheng_imu_principles_algorithms_camera_2026-09-05.md)
- **一句话说明：** 从 MEMS 六轴原理、误差标定、Mahony/Madgwick/EKF 姿态融合，到 Linux V4L2/IIO 驱动分工、曝光/元数据/时钟同步、EIS 与滚动快门补偿的完整 IMU–相机工程链教程。

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 总览概念页 | [imu-principles-algorithms-camera-sync](../../wiki/concepts/imu-principles-algorithms-camera-sync.md) |
| 交叉：传感器融合 | [sensor-fusion](../../wiki/concepts/sensor-fusion.md) |
| 交叉：EKF | [ekf](../../wiki/formalizations/ekf.md) |
| 交叉：VIO 选型 | [lidar-slam-lio-vio-selection](../../wiki/comparisons/lidar-slam-lio-vio-selection.md) |

## 核心摘录（MVP）

### 1) 六轴 ≠ 六自由度位姿

- 加速度计测 **比力**（静止桌面模长 ≈ 1 g），陀螺仪测 **角速度**；姿态/位置需算法估计。
- IMU / AHRS / INS 分层：IMU 只负责测量；AHRS 融合得姿态航向；INS 再传播速度与位置。

### 2) 误差会被积分放大

- 陀螺零偏 0.05°/s 静止积分 60 s ≈ 3° 漂移；加速度恒定误差 0.01 m/s² 在 60 s 对应约 18 m 位置误差（教学量级）。
- 标定（六面静态、温度模型）与 Allan 偏差分析是工程前置，不是可选优化。

### 3) 融合算法选型

- 单轴互补滤波 → 教学/受约束倾角；三维姿态用 Mahony（叉积反馈）或 Madgwick（梯度下降）。
- 需协方差/多传感器时用 EKF/ESKF；六轴 alone 不可观测绝对航向。

### 4) 相机–IMU 协同（嵌入式 Linux）

- **V4L2** 管图像流与帧元数据；**IIO** 管 IMU 采样/FIFO/时间戳；同步适配层建立时钟映射与帧间 IMU 窗口。
- 曝光寄存器「行数」须按 pixel rate / HBLANK / VBLANK 换算；批量 FIFO 读取 ≠ 同时采样。
- EIS 用陀螺轨迹做数字重映射；滚动快门需按行/分块使用不同时刻旋转。

## 当前提炼状态

- [x] 公众号正文抓取
- [x] 升格 wiki 概念页（非论文实体；本文无 arXiv 论文列表）
