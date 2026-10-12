# Quadruped-Provided Localization for Micro-UAVs 论文来源归档

## 元数据

- **题目：** Towards Quadruped-Provided Localization and Active Tracking for Micro-UAVs
- **作者：** Alejandro Lorite Mora, Andrés Faíña
- **单位：** IT University of Copenhagen；Helix Lab（Kalundborg）；Novo Nordisk A/S
- **预印本：** https://arxiv.org/abs/2610.06215
- **HTML：** https://arxiv.org/html/2610.06215v1
- **版本：** v1，12 页 5 图；评论栏称已被 CLAWAR 2026（Cranfield，2026-09-14 至 16）接收，出版版标题改为 “...for Nano-UAVs”（Springer LNNS）
- **代码与数据：** 页面与正文均未列项目页或仓库（仅链接依赖 spot_ros2、Crazyswarm2）；截至 2026-10-12 未发布。

## 论文要解决的问题

微型无人机载重与算力有限，无法携带完整定位传感器。论文把定位负担转移到地面四足机器人：Spot 臂端相机检测机上 AprilTag，与 Spot 自身的地图定位复合，得到无人机在共享地图坐标系中的位姿，无人机只带 IMU，通过外部位姿融合飞行。

## 方法与训练信息

1. **平台：** Boston Dynamics Spot + Spot Arm（含夹爪 7 自由度）+ Jetson AGX Orin 64 GB；自定位使用五路立体 RGBD、地面灰度相机与 IMU。臂端相机闭环用 640×480，型号原文未说明。
2. **无人机：** Crazyflie 2.1+（裸机 27 g），定制机架上四个 36h11 AprilTag（58.5×58.5 mm），飞行质量 42 g，无定位传感器。
3. **位姿链：** T_map→drone 由 map→vision、vision→body、body→cam、cam→tag、tag→drone 五个变换复合；apriltag_ros 在 CPU 运行，多标签可见时按决策裕度、再按 PnP 协方差迹选最强检测。
4. **无人机端：** 机载 EKF 融合外部位姿与 IMU，检测间隙靠 IMU 外推；超时无位姿则看门狗命令降落；使用 Crazyswarm2 控制栈、2.4 GHz Crazyradio PA 通信。
5. **主动跟踪：** 以近期检测拟合速度外推无人机位置，使相机光轴指向预测点并保持约 0.9 m 间距；调用 Spot Arm IK，2 s 内到达 6D 位姿，设最小安全距离与最大可达球，2 s 无检测则停止伺服；可选同时移动机身。
6. **软件：** ROS 2 Humble，Ubuntu 22.04 Docker，spot_ros2 驱动。

## 论文报告的结果

- **实验 1（静止 Spot 的 AprilTag 精度）：** 29 次飞行、134.8k 样本。原始位置误差约 50 mm（其中固定安装偏移约 46 mm）；刚性对齐后中位位置误差 1280×720 为 12 mm、640×480 为 16 mm；姿态误差 3.4° / 8.8°；p95 为 30 / 45 mm；位姿速率约 17 Hz / 40 Hz，超过 100 ms 的间断占比 15.7% / 1.1%，故闭环选 640×480。
- **实验 2（动捕 vs AprilTag 位姿飞行）：** 18 次飞行、84.8k 样本，指标为相对指令航点的轨迹误差。AprilTag 位姿中位 7.8 cm（1.5–2.0 m 距离），2.5–3.0 m 升至 11.0 cm；动捕约 5.5 cm 且基本不随距离变化。位姿延迟：AprilTag 中位 72 ms（p95 约 105 ms），动捕 2 ms。
- **实验 3（主动跟踪）：** 8 次飞行，640×480，轨迹误差中位数：Spot 静止 11.0 cm、仅臂跟踪 7.1 cm、臂+机身跟踪 6.9 cm。
- 动捕：8 台 OptiTrack PrimeX 13W，4×4×2.5 m 区域，120 Hz。

## 复现与解释边界

- 仅在室内实验室、动捕真值与已标定标签下验证；未做户外或穿越真实狭窄开口。
- 只测了定位与跟踪，没有建图/规划模块；快速偏航、长距离或斜视角会使标签退化，穿越窄缝时相机会丢失无人机。
- 实验 3 中 “still” 条件的距离约 2.7 m，而跟踪保持约 0.9 m，比较混入了自运动与距离两个因素（原文自述）。
- 无人机电池约 7 分钟；光照与 2.4 GHz 通信约束在实验室外未测。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-quadruped-uav-localization-active-tracking.md
- 周报汇总：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.06215>
- arXiv HTML 全文：<https://arxiv.org/html/2610.06215v1>
