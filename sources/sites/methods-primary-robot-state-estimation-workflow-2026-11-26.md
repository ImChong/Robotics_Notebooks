# Robot State Estimation Workflow · 机器人状态估计与定位：一手资料复核

- **类型：** 方法一手资料归档
- **核对日期：** 2026-11-26
- **通读：** 已完成
- **核对范围：** 以下所列文档、接口和实现范围；未运行实机或评测。
- **对应页面：** [方法页](../../wiki/methods/robot-state-estimation-workflow.md)

## 一手资料

- [ROS REP-105](https://www.ros.org/reps/rep-0105.html)：坐标框架、发布责任与里程计连续性。
- [robot_localization State Estimation Nodes](https://github.com/cra-ros-pkg/robot_localization/blob/rolling-devel/doc/state_estimation_nodes.rst)：阅读滤波器类别、传感器时间与协方差设置、world_frame 和已发布变换。

## 交叉核对

- 编码器、视觉或惯性运动信息用于预测或增量约束，按量测坐标系、噪声及有效时间执行修正。可依据约束结构选择滤波或图优化；局部里程计、已有地图定位和联合建图并不是同一问题。
- 滤波器输出频率不等于真实新增信息速率。量测超时后的纯预测可维持输出，误差仍会随时间增大；未标定、不同步或高度相关的输入可能导致看似稳定但有偏的结果。
