# TRON1 RL Deploy ROS2

> 来源归档（国内具身开源全景）

- **标题：** TRON1 RL Deploy ROS2
- **类型：** repo
- **机构：** 逐际动力
- **链接：** https://github.com/limxdynamics/tron1-rl-deploy-ros2
- **分类：** 仿真环境
- **入库日期：** 2026-09-06
- **一句话说明：** 逐际动力 开源项目 TRON1 RL Deploy ROS2（仿真环境），见 [国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/cn-os-tron1-rl-deploy-ros2.md`](../../wiki/entities/cn-os-tron1-rl-deploy-ros2.md)

## 开源状态

- **已开源**：公开仓库（以 README 与 release 为准）。

## 对 wiki 的映射

- [wiki/entities/cn-os-tron1-rl-deploy-ros2.md](../../wiki/entities/cn-os-tron1-rl-deploy-ros2.md)

## 部署入口补核（2026-10-05）

- **代码：** <https://github.com/limxdynamics/tron1-rl-deploy-ros2>；公开 ROS 2 / ONNX 部署实现，不是完整 RL 训练框架。
- **模块：** `robot_hw` 管理仿真/真机状态与执行，`robot_controllers` 组织策略输入并运行 ONNX；低层依赖 `limxsdk-lowlevel`。
- **入口：** `colcon` 编译；`ros2 launch robot_hw pointfoot_hw_sim.launch.py` 先验证仿真，真机启动与本体配置按 README 核对。
- **范围：** 本轮核查仓库与文档入口，未运行本体仿真/真机策略。
