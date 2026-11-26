# Robot Navigation Workflow · 机器人导航与路径执行：一手资料复核

- **类型：** 方法一手资料归档
- **核对日期：** 2026-11-26
- **通读：** 已完成
- **核对范围：** 以下所列文档、接口和实现范围；未运行实机或评测。
- **对应页面：** [方法页](../../wiki/methods/robot-navigation-workflow.md)

## 一手资料

- [Nav2 Navigation Concepts](https://docs.nav2.org/rolling/getting_started/navigation_concepts/)：导航服务器、定位、环境表示与行为树的模块划分。
- [Nav2 Environmental Representation](https://docs.nav2.org/rolling/getting_started/navigation_concepts/environmental_representation/)：代价地图分层、传感器来源与空间区域过滤。
- [ROS REP-105](https://www.ros.org/reps/rep-0105.html)：坐标框架、发布责任与里程计连续性。

## 交叉核对

- Nav2 将行为编排与规划、控制、恢复服务解耦。代价地图把占用、未知和障碍膨胀编码进二维网格；感知层可融合激光、深度等障碍来源。全局规划搜索可行路线，控制器读取路线与局部环境并输出控制量。
- 二维占用与膨胀半径不能刻画台阶、坡面和可跨越体积，不应视为复杂地形上的可通行性证明。规划器输出也不能替代硬件急停与安全层。
