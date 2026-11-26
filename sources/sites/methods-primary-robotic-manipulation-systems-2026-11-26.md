# Robotic Manipulation Systems · 机器人操作系统与末端执行器：一手资料复核

- **类型：** 方法一手资料归档
- **核对日期：** 2026-11-26
- **通读：** 已完成
- **核对范围：** 以下所列文档、接口和实现范围；未运行实机或评测。
- **对应页面：** [方法页](../../wiki/methods/robotic-manipulation-systems.md)

## 一手资料

- [MoveIt Planning Scene Monitor](https://moveit.picknik.ai/main/doc/concepts/planning_scene_monitor.html)：场景读写、环境监视和占用图输入。
- [MoveIt Motion Planning](https://moveit.picknik.ai/main/doc/concepts/motion_planning.html)：运动规划请求、环境碰撞与时序轨迹。
- [MoveIt Low Level Controllers](https://moveit.picknik.ai/main/doc/examples/controller_configuration/controller_configuration_tutorial.html)：机械臂和抓手控制器动作接口与执行监控。

## 交叉核对

- MoveIt 以规划场景描述机器人与环境，并用 PSM 同步机器人状态、传感器占用图和应用加入的物体。规划请求以当前状态、目标和约束为输入，输出受关节速度及加速度约束的时序轨迹，而不只是路径。
- 规划等效几何与真实软体、摩擦和夹持力不同。几何无碰不等同于稳定抓取；可控性、检测到的接触和急停需由独立硬件及控制链保证。
