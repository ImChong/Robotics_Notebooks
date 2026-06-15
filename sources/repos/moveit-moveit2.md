# moveit / moveit2

> 来源归档

- **标题：** moveit2 — MoveIt Motion Planning Framework for ROS 2
- **类型：** repo
- **维护者：** MoveIt Maintainers；PickNik Robotics 主导开发
- **链接：** https://github.com/moveit/moveit2
- **许可：** BSD-3-Clause
- **入库日期：** 2026-06-15
- **一句话说明：** **MoveIt 2** 官方 C++ 实现：运动规划、碰撞检测、IK、场景监控、ros2_control 接口与 MTC 等；2019 起 ROS 2 移植，ROSIN（Horizon 2020）曾资助移植工作。

## 核心摘录（据 README）

- **定位：** *Easy-to-use open source robotics manipulation platform* — 商业原型、算法基准与科研常用。
- **文档：** https://moveit.picknik.ai/
- **安装：** [Binary](https://moveit.ros.org/install-moveit2/binary/) · [Source](https://moveit.ros.org/install-moveit2/source/)
- **迁移：** [MIGRATION_GUIDE.md](https://github.com/moveit/moveit2/blob/main/doc/MIGRATION_GUIDE.md) · [Migration Progress 表格](https://docs.google.com/spreadsheets/d/1aPb3hNP213iPHQIYgcnCYh9cGFUlZmi_06E_9iTSsOI/edit?usp=sharing)
- **ROS 1 对照仓：** https://github.com/moveit/moveit
- **商业版：** [MoveIt Pro](http://picknik.ai/pro)

## 主要包族（buildfarm 表可见，非穷尽）

| 包 | 角色 |
|----|------|
| `moveit_core` | 核心数据结构、碰撞、运动学 |
| `moveit_ros_move_group` | **move_group** 节点 |
| `moveit_planners_ompl` / `_chomp` / Pilz | 规划器插件 |
| `moveit_ros_planning` / `_planning_interface` | 规划管线与客户端 API |
| `moveit_ros_perception` | 感知与占用地图 |
| `moveit_hybrid_planning` | 全局+局部混合规划 |
| `moveit_configs_utils` | 配置生成辅助 |
| `moveit_msgs` | ROS 2 消息/服务/动作定义 |

## 对 wiki 的映射

- 实体页：[wiki/entities/moveit2.md](../../wiki/entities/moveit2.md)
- 文档：[moveit2-picknik-documentation.md](../sites/moveit2-picknik-documentation.md)
- 门户：[moveit-official-portal.md](../sites/moveit-official-portal.md)
