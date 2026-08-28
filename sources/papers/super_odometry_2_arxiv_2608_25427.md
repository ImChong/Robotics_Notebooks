# SUPER ODOMETRY 2.0: Resilient Odometry via Hierarchical Adaptation

> 来源归档（ingest）

- **标题：** SUPER ODOMETRY 2.0: Resilient Odometry via Hierarchical Adaptation
- **短名：** Super Odometry 2.0
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.25427>
- **PDF：** <https://arxiv.org/pdf/2608.25427>
- **期刊：** *Science Robotics* 2025，DOI [10.1126/scirobotics.adv1818](https://doi.org/10.1126/scirobotics.adv1818)
- **项目页：** <https://superodometry.com>
- **代码（slim）：** <https://github.com/superxslam/SuperOdom>
- **入库日期：** 2026-08-28
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)（<https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>）
- **一句话说明：** 四级自适应传感器融合，在烟雾、沙尘、积雪与弱光中把 IMU 提升为与相机 / LiDAR 同等的后备。

## 开源状态（步骤 2.5）

- **部分开源**：项目页 Code 指向 [`superxslam/SuperOdom`](https://github.com/superxslam/SuperOdom)（ROS 2 Humble，GPL-3.0）。README 写明这是 **slim version**（LiDAR odometry + IMU odometry 互为约束），不是论文四级自适应 + 100 小时异构学习式惯性模块的完整复现。学习式 IMU / 引擎选择以论文与项目页为准。

## 核心摘录（面向 wiki 编译）

### 摘录 1：分层自适应

- 四级：自适应特征选择 → 状态方向选择 → 引擎选择 → 学习式惯性里程计。
- 惯性模型用超过 **100 小时**异构机器人数据训练；外感知失效时 IMU 作可靠后备。
- 名义条件：因子图给 IMU 网络提供自由位姿标签；退化条件：学习式 IMU 接管。

**对 wiki 的映射：** [paper-super-odometry-2](../../wiki/entities/paper-super-odometry-2.md)、[里程计与激光雷达融合](../../wiki/methods/lidar-odometry-fusion.md)

### 摘录 2：评测

- **200 公里 / 800 小时**，空中、轮式、腿式；13 类连续退化单次跑，腿式终点漂移约 **20 cm / 2966 m**。

**对 wiki 的映射：** [LiDAR / LIO / VIO 选型](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-super-odometry-2.md`](../../wiki/entities/paper-super-odometry-2.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
