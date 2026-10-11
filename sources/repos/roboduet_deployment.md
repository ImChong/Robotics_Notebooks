# Source: RoboDuet_Deployment

- **Title:** Unitree Go1/Go2 + ARX5 Deployment
- **URL:** https://github.com/locomanip-duet/RoboDuet_Deployment
- **Project page:** https://locomanip-duet.github.io/
- **Related training repository:** https://github.com/locomanip-duet/RoboDuet
- **Paper:** https://arxiv.org/abs/2403.17367
- **License:** MIT
- **Default branch:** master
- **核查日期：** 2026-10-11
- **Type:** Official hardware deployment code

## 仓库职责与运行入口

该仓库单独收录训练后的 Go1/Go2 + ARX5 真机部署路径：

- **Go1 + ARX5：** README 描述基于 Docker 的部署环境；按次序启动低层 `lcm_position_vr`、RL policy（Docker autostart）和 `remote_pub.py` VR receiver。
- **Go2 + ARX5：** README 提供 `scripts/deploy_policy_vr.py`、`remote_pub.py` 与 `lcm_position_vr_go2` 入口；策略可运行在 PC 或可选的 NVIDIA Jetson Orin 上。
- **控制输入：** 远程控制器处理四足速度，Meta Quest 3 / VR 接口发送末端位姿目标。
- **硬件准备：** README 说明 Unitree SDK、LCM、ARX 机械臂 CAN/串口映射与网络配置等依赖。

## 许可和复现边界

仓库标注 MIT License 并公开硬件侧程序。实际部署仍要求匹配机器人/机械臂、SDK、总线、网络与 VR 配置；README 的安装说明不意味着无需校准即可复现论文实验。该仓库不是训练代码的替代品，而是同一 RoboDuet 项目的部署部分。

## 对 wiki 的映射

- 项目实体：[RoboDuet](../../wiki/entities/paper-roboduet-loco-manipulation.md)
- 论文来源：[论文归档](../papers/roboduet_arxiv_2403_17367.md)
- 项目页来源：[项目页归档](../sites/roboduet.md)
- 训练代码：[RoboDuet](./roboduet.md)
