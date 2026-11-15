# Kinova Gen3：官方技术资料与运行入口核对

- **类型：** repo / 官方技术文档
- **核对日期：** 2026-11-15
- **核对范围：** `Kinovarobotics/ros_kortex/noetic-devel` 的根 README、`kortex_description/readme.md`、`kortex_driver/readme.md` 全文，以及 `kortex_examples/src/full_arm/example_cartesian_poses_with_notifications.py` 全文；范围限 ROS1 驱动与该笛卡尔示例。
- **通读：** 已完成
- **复现状态：** 本次为文档与源码阅读，未连接硬件、未运行控制程序，不等于实机验证。

## 主要原始资料

- [官方 ROS 仓](https://github.com/Kinovarobotics/ros_kortex/tree/noetic-devel)、[型号描述](https://github.com/Kinovarobotics/ros_kortex/blob/noetic-devel/kortex_description/readme.md)
- [驱动完整说明](https://github.com/Kinovarobotics/ros_kortex/blob/noetic-devel/kortex_driver/readme.md)
- [带通知的笛卡尔动作例程](https://github.com/Kinovarobotics/ros_kortex/blob/noetic-devel/kortex_examples/src/full_arm/example_cartesian_poses_with_notifications.py)。源码树 `c3280d8a`。

## 方法、实现与限制

ROS Noetic/Ubuntu20.04 分支的 `kortex_driver` 将 ROS 服务/话题映射到 Kortex API。Gen3 默认7自由度，支持显式6自由度；Gen3 lite 默认6自由度，夹爪/vision 参数各异。文档区分持续发布的 base_feedback/joint_state 与需显式激活的通知话题，高层 base 命令按25ms处理，不能将其当作1kHz力矩控制接口。例程初始化服务代理和 action_topic 订阅，清故障、读出并执行 Home、设置参考系、激活通知，再依次发送笛卡尔动作并等待 ACTION_END/ACTION_ABORT。所读例程有两处应复核：Home 等待在显式激活通知之前；参考系函数构造 req 却未将其传给 ServiceProxy。这些是阅读发现，未实机验证。BSD-3-Clause 的 ROS 代码许可不等于机械本体 CAD 全开放。

## 对 wiki 的映射

- [Kinova Gen3](../../wiki/entities/kinova-gen3.md)
