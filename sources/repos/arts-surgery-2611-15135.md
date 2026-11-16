# ARTS: An Open-Source, Reproducible Three-Arm Platform for Autonomy Research in Minimally Invasive Surgery：作者代码核对

- **类型：** 官方代码仓
- **代码：** https://github.com/MRE-Lab-UMD/arts
- **论文：** https://arxiv.org/abs/2611.15135
- **公开入口：** `ros2 launch arts_driver arts_bringup.launch.py psm_left:=true psm_right:=true ecm:=true`
- **相关实现：** `arts_driver 和 ros2_control`
- **产物：** 机械臂控制及同步传感器流
- **核对范围：** 官方 README 及对应启动/评测/模块目录的静态核查，未运行实际训练或硬件。
- **通读：** 已完成
- **wiki：** [论文详情](../../wiki/entities/paper-arts-surgery-2611-15135.md)

**范围限制：** 论文测试为台架、遥操作与少量训练任务，不能证明临床安全性；无人从独立实验室完全复现安装链，部分摄像头 SDK 为专有依赖。
