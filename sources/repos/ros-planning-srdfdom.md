# ros-planning / srdfdom

> 来源归档

- **标题：** srdfdom — Semantic Robot Description Format (SRDF) parser
- **类型：** repo
- **链接：** https://github.com/ros-planning/srdfdom
- **入库日期：** 2026-06-15
- **一句话说明：** **SRDF** 的 C++/Python 解析与写入库；MoveIt **Setup Assistant** 生成、**move_group** 读取的语义层（规划组、碰撞禁用、末端执行器、命名姿态）均依赖 SRDF，与 URDF 配对使用。

## 为什么值得保留

- **MoveIt 配置三件套中的「语义」真值：** URDF 描述 link/joint 几何与运动学；SRDF 描述 **哪些 link 组成 arm/gripper 规划组**、**哪些 link 对可忽略碰撞**、**ready/home 等 group state**。
- MoveIt 2 文档明确 move_group 需要 **URDF + SRDF + MoveIt configuration**；外部 URDF/SRDF 生成工具产出须与此格式兼容。

## 能力摘录（据 README）

- C++ / Python **parser** 与 C++ **writer**
- 示例：`test/test_parser.cpp`、`test/test.py`；`display_srdf` 命令行工具（ROS 2：`ros2 run srdfdom display_srdf ...`）

## 对 wiki 的映射

- 交叉：[wiki/entities/moveit2.md](../../wiki/entities/moveit2.md)、[wiki/entities/urdf-studio.md](../../wiki/entities/urdf-studio.md)
