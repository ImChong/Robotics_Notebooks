# SuperOdom（superxslam/SuperOdom）

- **URL：** <https://github.com/superxslam/SuperOdom>
- **许可：** GPL-3.0
- **主页：** <https://superodometry.com>
- **配套论文：** Super Odometry 2.0 [arXiv:2608.25427](https://arxiv.org/abs/2608.25427)；v1 IROS 2021

## 状态（2026-08-28）

README 自称为 **slim version**：LiDAR odometry 与 IMU odometry 互为约束。支持 Livox / Velodyne / Ouster，ROS 2 Humble。

```bash
ros2 launch super_odometry livox_mid360.launch.py
```

完整四级自适应与 100 小时学习式 IMU **不在**本 slim 仓范围内。

## wiki

- [`wiki/entities/paper-super-odometry-2.md`](../../wiki/entities/paper-super-odometry-2.md)
