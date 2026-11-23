# KILVO: Kinematic-Inertial-LiDAR-Visual Odometry with Robust Multimodal Adaptation for Humanoid Robots（arXiv:2608.05647）

> 来源归档（ingest）

- **标题：** KILVO: Kinematic-Inertial-LiDAR-Visual Odometry with Robust Multimodal Adaptation for Humanoid Robots
- **缩写 / 框架：** **KILVO**（Kinematic-Inertial-LiDAR-Visual Odometry）
- **类型：** paper / humanoid / odometry / slam / sensor-fusion / esikf
- **arXiv：** <https://arxiv.org/abs/2608.05647>（Submitted 2026-08-06；PDF：<https://arxiv.org/pdf/2608.05647>）
- **DOI：** <https://doi.org/10.1109/TMECH.2026.3721778>（IEEE/ASME Transactions on Mechatronics）
- **代码仓：** <https://github.com/JixinGao/KILVO> — 归档见 [`sources/repos/kilvo.md`](../repos/kilvo.md)
- **作者：** Jixin Gao、Fucheng Liu、Teng Zhang、Fusheng Zha（通讯）
- **机构：** 哈尔滨工业大学机器人技术与系统全国重点实验室（HIT；英文：State Key Laboratory of Robotics and Systems）
- **入库日期：** 2026-08-08
- **复核日期：** 2026-11-21
- **原文下载/核对日：** 2026-11-21
- **通读：** 已完成
- **核对范围：** 全文 2292 行（Abstract、§I–§VIII 方法与因子/ESIKF 推导、传感器退化切换、真机 G1/LIKO 评测表）+ 仓库 `JixinGao/KILVO` 源码（`src/main.cpp`, `src/LIVMapper.cpp`, `launch/mapping_mid360_d455i.launch`）
- **一句话说明：** 面向人形的运动学–惯性–激光–视觉里程计：异步–顺序混合 ESIKF 紧耦合关节编码/IMU/LiDAR/相机，含接触估计与模态自适应，输出可达 1 kHz；**官方源码（GPL-2.0）与 15 段 rosbag 数据集已于 2026-11-02 在 GitHub 公开发布**。

## 开源状态（步骤 2.5）

- **仓库核查（2026-11-21 复核）：** [`JixinGao/KILVO`](https://github.com/JixinGao/KILVO)（GPL-2.0，46 files，`pushed_at: 2026-11-02`）已完整发布 ROS Noetic C++ catkin 源码包（`src/main.cpp` → `LIVMapper`、`kio.cpp`、`vio.cpp`、`voxel_map.cpp`、`IMU_Processing.cpp`）、启动文件（`launch/mapping_mid360_d455i.launch`、`mapping_mid360_mvs.launch`、`mapping_liko.launch`）及 OneDrive 15 段 Unitree G1 人形 rosbag 序列（含 Livox Mid360 10 Hz / 200 Hz IMU、RealSense D455 / Hikvision MVS 10 Hz、`/g1/joint_state` 500 Hz）。
- **结论：** **已开源**（可编译运行 + 数据集已开放；详见 [`sources/repos/kilvo.md`](../repos/kilvo.md)）。

## 摘录 1：问题与主张（§I）

- **痛点：** 纯本体感里程计长时漂；人形冲击/跌倒/光照退化导致传感器失效；接触信息常依赖额外力传感器或学习模块。
- **主张：** 在 **异步–顺序混合 ESIKF** 中：IMU 预测；腿运动学高率异步更新；外感先 LiDAR 几何再视觉光度顺序更新；多模态适配容忍失效；接触估计复用运动学/惯性/地图线索、无额外传感器。
- **贡献：** 框架 + 接触模块 + 人形 SLAM 数据集（15 序列多步态）+ 公开/真机评测。
- **发表：** IEEE/ASME Transactions on Mechatronics（DOI 10.1109/TMECH.2026.3721778）；单位为 HIT 机器人技术与系统全国重点实验室。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-kilvo.md`](../../wiki/entities/paper-kilvo.md)；与 [里程计–激光融合](../../wiki/methods/lidar-odometry-fusion.md)、[FAST-LIO](../../wiki/entities/fast-lio.md)、[LIO/VIO 选型](../../wiki/comparisons/lidar-slam-lio-vio-selection.md) 互链。

## 摘录 2：系统机制（§III–§V）

- **状态估计：** 全模态时紧耦合；编码器失效 → LIV（输出率可降至 10 Hz）；LiDAR 失效 → KI（暂停建图）；相机失效 → KIL（点云去色）等，可恢复后重融合。
- **接触：** 脚–地 patch / 速度线索等紧凑检测；相对 HR²-KILO 接触模块约 **76%** 时延改善（~0.02 ms）；多数序列准确率 >95%，平均 FPR 3.67%。
- **平台数据：** 公共 LIKO（BHR）、HR²-KILO（Unitree 等）；自采 KILVO 数据集含 Unitree G1（Mid360 / IMU / 编码器 1 kHz）等。

**对 wiki 的映射：** 实体页画 ESIKF 异步–顺序数据流与模态降级表。

## 摘录 3：评测（§VI / Table IV–VI）

| 设定 | 要点 |
|------|------|
| LIKO 公共集 | KILVO 平均 ATE RMSE **0.0151 m**（5 序中 3 序最低）；RTE 全序领先 |
| HR²-KILO 集 | Z 轴端到端误差多 **<1 cm** |
| 真机 15 序 | 端到端平移平均误差 **0.0145 m**（全方法最佳均值）；robust 序列相对 LIO-only 大幅更稳 |
| 效率 | 完整处理约十余 ms 量级；**1 kHz** 输出（异步运动学阶段） |
| 失效/恢复 | robust h02* 分段丢编码/LiDAR/图像后仍可完成，端到端约 **0.0165 m** |

**对 wiki 的映射：** 强调「人形冲击 + 模态失效」场景相对 FAST-LIO2 / LIKO / HR²-KILO 的鲁棒读法；源码已按 GPL-2.0 公开。

## 建议 wiki 动作

- 维护 **`wiki/entities/paper-kilvo.md`**、**`sources/repos/kilvo.md`**（同步开源状态、`LIVMapper` 源码时序图与 15 段 rosbag 数据入口）。
- 交叉更新 lidar-odometry-fusion / lidar-slam 选型页。
