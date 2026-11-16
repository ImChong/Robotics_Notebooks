# Robot Navigation and Obstacle Avoidance from Randomly Positioned External Cameras：作者代码核对

- **类型：** 官方代码仓
- **代码：** https://github.com/raulsteinmetz/vine
- **论文：** https://arxiv.org/abs/2611.16282
- **公开入口：** `~/IsaacLab/isaaclab.sh -p executables/evaluate.py`
- **相关实现：** `source 的障碍图估计器及控制器`
- **产物：** 地图 IoU 和导航评测
- **核对范围：** 官方 README 及对应启动/评测/模块目录的静态核查，未运行实际训练或硬件。
- **通读：** 已完成
- **wiki：** [论文详情](../../wiki/entities/paper-vine-exocentric-nav-2611-16282.md)

**范围限制：** 依赖机器人位姿、近似摄像头位姿和房间尺寸；缺少这些输入不能声称任意布置的摄像头均无需标定。
