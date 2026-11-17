# GN10 PointCloud Localization — ESDF + CUDA 3D Texture 实时点云定位

> 一手源码归档；按项目实体而非论文身份收录。

- **标题：** GN10 PointCloud Localization / 基于 ESDF 与 CUDA 3D 纹理的实时激光点云定位
- **类型：** repo
- **来源：** 东京都立产业技术高等专门学校荒川机器人竞赛 2026A 团队（TMCIT Ararobo 2026A）；源码文件版权标注 Gento Aiba and contributors
- **链接：** https://github.com/tmcit-ararobo-2026a/gn10-pointcloud-localization
- **原文：** https://github.com/tmcit-ararobo-2026a/gn10-pointcloud-localization/blob/main/README.md
- **核对版本：** `3691e308d7e66d6c277c67f76a4d1cfcbc6e32a4`（main，2026-11-12）
- **许可证：** Apache-2.0（仓库 LICENSE / README）
- **代码：** https://github.com/tmcit-ararobo-2026a/gn10-pointcloud-localization
- **开源状态：** 已开源；提供 C++/CUDA ROS 2 节点、launch、参数、ESDF 转换程序、算法源码
- **入库日期：** 2026-11-17
- **核对范围：** 官方 README.md 全文（简介、系统架构、依赖与构建、地图转换与启动、参数、图表和许可）；结合 `launch/localization.launch.py`、`src/main.cpp`、`src/localization_node.cpp` 的初始化/回调/丢失状态与输出段、`src/pose_solver.cpp`、`src/global_searcher.cpp`、`src/cuda/esdf_matcher.cu`、`src/map_loader.cpp`、`src/esdf_map.cpp`、`src/pcd_to_esdf_converter.cpp`、`config/common_params.yaml` 核对主要运行链与关键边界。未逐行审计全部源码，未编译或做真机基准测试。
- **通读：** 已完成
- **一句话说明：** 用预建静态地图生成 ESDF，并借助 GPU 3D 纹理快速评估 LiDAR 扫描候选位姿；支持无手动初始位姿启动、丢定位全局重搜索和非地图障碍提取。
- **沉淀到 wiki：** [GN10 PointCloud Localization](../../wiki/entities/gn10-pointcloud-localization.md)
- **路线：** [导航纵深 · Stage 1](../../roadmap/depth-navigation.md)

## 已核实的系统边界

1. **定位而不是从零建图：** 依赖预置场地/静态环境地图，读取 `.pcd`、`.esdf`、`.json` 或 ROS 参数中的几何物体；PCD 可预转换并缓存为 ESDF。
2. **三维点云、平面位姿：** ROS 节点发布的位姿为 `x, y, yaw`（`z=0`），不是完整 6-DoF 位姿估计；3D ESDF 用于空间几何匹配。
3. **不用轮式里程计，也无需手输初始位姿：** 使用 LiDAR 点云与静态 ESDF 匹配；实现仍用 IMU yaw 历史、扫描去畸变和匹配结果平移速度外推。代码提供可选外部 fused prior。
4. **双雷达配置可选：** 仓库默认配置启用 2D Lakibeam 点云融合与退化回退（`lidar_2d.enable: true`）；若只接 Livox MID-360，需核对/关闭该配置，并确保 3D LiDAR 与 IMU、TF 输入齐全。
5. **全局重定位：** 默认连续 25 帧匹配失败进入 LOST，进入对场地范围内 `x/y/yaw` 的 GPU 批量粗搜索，取 Top-K（默认 4）候选并细化、验证接受条件。首次无已知位姿时也可触发全局搜索。
6. **CUDA 加速：** `src/cuda/esdf_matcher.cu` 使用 `cudaTextureObject_t` 和 `tex3D<float>` 三线性插值查询距离；README 的单次点查询 O(1) 不代表整帧或全局搜索 O(1)。
7. **动态点提取：** 与静态 ESDF 距离超过 `matching_params.dynamic_dist_thresh`（默认 0.15 m）的非地图点可用于 `/dynamic_cloud`，不能代替通用目标语义识别。
8. **掉线安全行为：** 连续失败/LOST 停止更新 TF；但 `cloudCallback` 对至多两帧、距上次成功匹配不超过 0.10 s 的短暂失败，可能发布预测 TF。不能概括为“一帧失败即绝不发布 TF”。

## 关键参数（来自 config/common_params.yaml）

| 配置 | 仓库默认值 | 用途 |
|---|---|---|
| `esdf.resolution` | `0.05` m | 三维 ESDF 体素分辨率 |
| `esdf.max_dist` | `0.50` m | ESDF 截断距离 |
| `scan_accumulation.window_s` | `0.10` s | Livox 非重复扫描累计窗口 |
| `matching_params.dynamic_dist_thresh` | `0.15` m | 动态点/非地图点判别 |
| `global_search.lost_count_thresh` | `25` 帧 | 转入全局搜索的匹配失败阈值 |
| `global_search.top_k_candidates` | `4` | 全局粗搜后细化候选数量 |
| `lidar_2d.enable` | `true` | 可选 2D LiDAR 融合（默认打开） |

## 官方运行入口（未执行）

```bash
# 配置 map/ 中的静态地图、输入话题与 TF 后
ros2 launch gn10_pointcloud_localization localization.launch.py

# rosbag / 仿真时钟
ros2 launch gn10_pointcloud_localization localization.launch.py use_sim_time:=true

# 预转换 PCD 地图
ros2 run gn10_pointcloud_localization pcd_to_esdf_converter \
  map/my_field.pcd map/my_field.esdf 0.05 0.50
```

构建依赖：Ubuntu 22.04、ROS 2 Humble、CUDA 12.0+、受支持的 NVIDIA GPU、Ceres、Eigen3、PCL 等。官方列举 Jetson Orin Nano；文档未给出可独立验证的跨地图精度、延迟/频率或重定位成功率统计，入库时不写未经验证的数值。

## 与其他技术的关系

- [FAST-LIO](../../wiki/entities/fast-lio.md) / [LIO-SAM](../../wiki/entities/lio-sam.md) 偏激光/惯性里程计与建图；GN10 主线是**已知静态地图中的全局定位和失锁恢复**，概念上互补，但不能直接宣称即插即用。
- [Isaac ROS nvblox](../../wiki/entities/isaac-ros-nvblox.md) 强调 GPU 体素地图生成与障碍感知；GN10 用 ESDF 作为匹配残差的查询结构。
- [导航纵深 Stage 1](../../roadmap/depth-navigation.md) 将其放在地图匹配 / 全局重定位的位置，而非建图算法一栏。

## 一手资料

- 官方 README（完整）：https://github.com/tmcit-ararobo-2026a/gn10-pointcloud-localization/blob/3691e308d7e66d6c277c67f76a4d1cfcbc6e32a4/README.md
- ROS 入口和实时回调：https://github.com/tmcit-ararobo-2026a/gn10-pointcloud-localization/blob/3691e308d7e66d6c277c67f76a4d1cfcbc6e32a4/src/localization_node.cpp
- 全局重定位：https://github.com/tmcit-ararobo-2026a/gn10-pointcloud-localization/blob/3691e308d7e66d6c277c67f76a4d1cfcbc6e32a4/src/global_searcher.cpp
- CUDA ESDF 匹配：https://github.com/tmcit-ararobo-2026a/gn10-pointcloud-localization/blob/3691e308d7e66d6c277c67f76a4d1cfcbc6e32a4/src/cuda/esdf_matcher.cu
