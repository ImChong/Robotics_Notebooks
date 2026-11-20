# lidar_localization_ros2 — ROS 2 地图式三维 LiDAR 定位（一手源码归档）

> 项目身份：开源软件仓库（非独立论文）；不等同于 SLAM 建图，也不等同于 GN10 ESDF/CUDA 定位器。

- **标题：** lidar_localization_ros2 / 基于预建三维点云地图的 ROS 2 全局初始化与激光定位
- **类型：** repo
- **维护者：** Ryohei Sasaki（GitHub：rsasaki0109）
- **链接：** https://github.com/rsasaki0109/lidar_localization_ros2
- **原文：** https://github.com/rsasaki0109/lidar_localization_ros2/blob/main/README.md
- **核对版本：** `285070fc1702209286e538f4355ba1e4f0fe926a`（上游 main，提交于 2026-11-19）
- **许可证：** BSD-2-Clause（上游 README / LICENSE）
- **代码：** https://github.com/rsasaki0109/lidar_localization_ros2
- **开源状态：** 已开源，包含 ROS 2 C++ 定位组件、NDT 相关依赖接入、Python 启动与自动初始化、BBS 候选搜索、恢复监督及 benchmark 工具；不等于完成真机闭环验证。
- **入库日期：** 2026-11-20
- **核对范围：** 官方 README 全文（安装/quickstart/运行机制/Topics与TF/Benchmark/限制）；docs/global_localization.md（G1/G2/G3、安全门控和退化案例）、docs/benchmark.md（指标定义、Go2 室内/大地图/跨采集时段表格及复现数据）、docs/go2_mid360_quickstart.md（安装流程和低安装高度参数）；结合 scripts/quickstart.py 的地图栅格生成及 launch_parts，scripts/startup_initialization_node.py 的状态、初始参数和话题，scripts/global_localization_node.py 的 G2 服务接口与算法导入，以及 src/component_alignment.cpp 组件逻辑入口检查。未逐行审计所有代码、未在 ROS 2 设备编译/实机复现实验。
- **通读：** 已完成
- **一句话说明：** 消费 .pcd/.ply 预建点云地图与 PointCloud2 激光数据，启动时在地图范围内搜索初始位姿，随后按预测位姿做 NDT/GICP scan-to-map 跟踪，必要时请求受门控的重定位；可接入 Nav2。
- **沉淀到 wiki：** [lidar_localization_ros2 独立实体](../../wiki/entities/lidar-localization-ros2.md)
- **路线：** [导航纵深 Stage 1](../../roadmap/depth-navigation.md)

## 从官方原文确认的系统边界

1. **预建地图是必要输入。** .pcd/.ply 点云用于局部匹配；首次初始化还需从它生成（或提供）2D 占据栅格用于 Branch-and-Bound 搜索。不能理解成没有地图也能定位。
2. **初始位姿可选。** `quickstart.py --map` 发现 ROS 点云、IMU、TF 与时间源；优先显式位姿，其次按同一地图校验已保存位姿，否则全局搜索；不确定时要求 RViz 人工指定，而非无条件保证成功。
3. **全局搜索与跟踪不是同一个优化器。** G1/G2 用 BBS_2D 栅格候选、NDT 候选排序/可选细化；C++ 定位器将当前扫描与预建点云图做 NDT（默认）或 GICP 配准。
4. **里程计不是 GPS。** 可选外部 RKO-LIO、腿式/轮式 odom 和 IMU 为帧间预测及短时桥接服务；实际性能取决于运动、外参、时钟、点云重叠与地图质量。
5. **受门控的丢失恢复。** G3 recovery supervisor 是单独的 bringup/显式选项，不可理解成默认所有运行方式都持续自动全图重置；候选评分、去抖、冷却、次数上限和验证门槛用于防止错误重置。

## 数据与指标：不能把「2 cm」当通用指标

官方的 `docs/benchmark.md` 说明所有主要 Go2 测试从空初始位姿开始，在地图坐标系直接对运动捕捉真值评分，**未做轨迹后对齐**。首次出位姿耗时与跟踪误差是两个不同量。

| 测试条件 | 结果（官方历史回放，不是本文复现） |
|---|---|
| JEPLO Go2，EIL_Mix 地图、4 个室内序列 | 12/12 次自动初始化；首次位姿约 9.8–17.1 秒；xy 误差中位数 0.03 m（2026-10-06） |
| JEPLO Go2，大地图及 EIL 重建输入，低位安装传感器参数 | 18/18 次初始化、无错误初始位姿；中位 0.02–0.03 m（2026-10-11） |
| JEPLO 跨采集时段，全部 24 次回放 | 23 次有位姿、其中 5 次错误初始位姿；正确初始化子集的各序列中位数 0.02–0.22 m（2026-11-18） |
| 不变场景跨会话 Long_Stairs ↔ Outdoor2 | 4/4 次正确初始化；首次位姿约 12.3–13.3 秒；起始误差 0.02 m，定位中位 0.02 m（2026-11-18） |
| 变化/近对称 EIL_Box 地图跨会话 | EIL_Box→EIL_Mix 与 EIL_Box→EIL_Stairs 共 4/4 次初始位置错误，首次偏移 1.05–5.50 m |

注意：跨会话实验含「同一路线 1.5 h 后重新采集」与「室内道具移动」两类条件，不能混称任意变化场景也有 2 cm。作者明确列出无多小时运行、无真实硬件在线验证；请优先检查首位姿、对称/重复结构以及状态告警。

## 上游运行入口与代码对照

```bash
# ROS 2 Humble 或 Jazzy；官方推荐脚本安装
mkdir -p ~/lidarloc_ws/src && cd ~/lidarloc_ws/src
git clone https://github.com/rsasaki0109/lidar_localization_ros2.git
cd lidar_localization_ros2
scripts/bootstrap_colcon_workspace.sh --build
source ~/lidarloc_ws/install/setup.bash

# 需要事先启动 LiDAR driver；可选先启动 RKO-LIO 里程计
ros2 run lidar_localization_ros2 quickstart.py --map /absolute/path/to/map.pcd
ros2 run lidar_localization_ros2 check_lidar_localization_bringup.py --profile standalone
```

Unitree Go2 + Livox MID-360 路线：先用 lidar_slam_ros2 的 `lidarslam-map start my_area.bag` 建图，再 `ros2 launch lidarslam rko_lio_odometry.launch.py`，最后用 `quickstart.py --map` 定位。注意「建图」由外部系统进行，并非本项目定位节点暗中构建。

低位 MID-360 在大地图的官方参数示例：

```bash
ros2 run lidar_localization_ros2 quickstart.py --map output/my_area/map.pcd \
  --occupancy-inflate-radius-m 0.2 --global-z-min 0.0 \
  --global-max-scan-points 1024 --global-max-candidates 16 \
  --refine-global-candidates
```

官方明确指出这组参数与安装高度/视野相关，不应直接照搬到手持 LiDAR 或车载场景。

## 一手资料（固定快照）

- [README](https://github.com/rsasaki0109/lidar_localization_ros2/blob/285070fc1702209286e538f4355ba1e4f0fe926a/README.md)
- [Global Localization：G1/G2/G3 与安全门控](https://github.com/rsasaki0109/lidar_localization_ros2/blob/285070fc1702209286e538f4355ba1e4f0fe926a/docs/global_localization.md)
- [Benchmark：误差指标、逐序列与跨会话失败案例](https://github.com/rsasaki0109/lidar_localization_ros2/blob/285070fc1702209286e538f4355ba1e4f0fe926a/docs/benchmark.md)
- [Go2 MID-360 复现教程](https://github.com/rsasaki0109/lidar_localization_ros2/blob/285070fc1702209286e538f4355ba1e4f0fe926a/docs/go2_mid360_quickstart.md)
- [quickstart.py 实际运行入口](https://github.com/rsasaki0109/lidar_localization_ros2/blob/285070fc1702209286e538f4355ba1e4f0fe926a/scripts/quickstart.py)
- [startup_initialization_node.py](https://github.com/rsasaki0109/lidar_localization_ros2/blob/285070fc1702209286e538f4355ba1e4f0fe926a/scripts/startup_initialization_node.py)
- [global_localization_node.py](https://github.com/rsasaki0109/lidar_localization_ros2/blob/285070fc1702209286e538f4355ba1e4f0fe926a/scripts/global_localization_node.py)
- [JEPLO 数据集](https://huggingface.co/datasets/ASIG-Lab/JEPLO-Dataset)
