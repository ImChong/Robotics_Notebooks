# JixinGao/KILVO

> 来源归档

- **标题：** KILVO（官方开源实现与数据集）
- **类型：** repo
- **组织 / 作者：** Jixin Gao 等（哈尔滨工业大学机器人技术与系统全国重点实验室）
- **代码：** <https://github.com/JixinGao/KILVO>
- **论文：** <https://arxiv.org/abs/2608.05647>
- **DOI：** <https://doi.org/10.1109/TMECH.2026.3721778>
- **入库日期：** 2026-08-08
- **复核日期：** 2026-11-21
- **一句话说明：** KILVO 官方 ROS Noetic C++ 开源实现（GPL-2.0，46 files，`pushed_at: 2026-11-02`），包含 `LIVMapper`（KIO/VIO/LIO 异步顺序融合与模态自适应）及 OneDrive 15 段 Unitree G1 rosbag 序列。

## 开源核查（2026-11-21 复核）

| 项 | 状态 |
|----|------|
| 仓库可见 | 是（公开仓，46 files） |
| 可运行代码 | **是** — ROS Noetic catkin C++ 包：`src/main.cpp`、`src/LIVMapper.cpp`、`kio.cpp`、`vio.cpp`、`voxel_map.cpp`、`IMU_Processing.cpp` |
| Datasets | **已开放** — OneDrive 提供 15 段 Unitree G1 rosbag 序列（Mid360 LiDAR/IMU + RealSense D455 / Hikvision MVS + `/g1/joint_state` 500 Hz） |
| License | GPL-2.0 |
| 结论 | **已开源（源码 + 数据集均已发布）** |

## 入口速查

| 路径 / 命令 | 作用 |
|-------------|------|
| `roslaunch kilvo mapping_mid360_d455i.launch` | Mid360 + D455i 在线/rosbag 启动入口（加载 `mid360_d455.yaml` 与 `camera_d455_pinhole.yaml`） |
| `roslaunch kilvo mapping_mid360_mvs.launch` / `mapping_liko.launch` | 海康 MVS 相机与 LIKO 数据集启动入口 |
| `src/main.cpp` → `LIVMapper::run()` | ROS 订阅回调与 `slam_mode_adjust()` → `stateEstimationAndMapping()` 主循环 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [KILVO 论文实体](../../wiki/entities/paper-kilvo.md) | 方法归纳与评测读法 |
| [里程计–激光融合定位](../../wiki/methods/lidar-odometry-fusion.md) | 人形多传感器融合对照 |
| [FAST-LIO](../../wiki/entities/fast-lio.md) | LIO 基线对照 |

## 对 wiki 的映射

- 论文：[`sources/papers/kilvo_arxiv_2608_05647.md`](../papers/kilvo_arxiv_2608_05647.md)
- 沉淀 **[`wiki/entities/paper-kilvo.md`](../../wiki/entities/paper-kilvo.md)**
