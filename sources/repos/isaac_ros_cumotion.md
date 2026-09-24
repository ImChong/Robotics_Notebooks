# Isaac ROS cuMotion

- **标题:** isaac_ros_cumotion — GPU 运动规划（MoveIt 2 插件）
- **链接:** [https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion)
- **类型:** repo
- **摘要:** NVIDIA **Isaac ROS** 包族：基于 **cuRobo** 的 **GPU 无碰撞轨迹生成**，经 **`isaac_ros_cumotion_moveit`** 作为 **MoveIt 2 planning plugin**；支持 **静态 planning scene**、**Nvblox ESDF** 世界碰撞与 **object attachment**（携带物体碰撞球/几何）。ROS 2 action：`cumotion/move_group`、`cumotion/motion_plan`、`cumotion/ik` 等。
- **文档:** [Isaac ROS cuMotion](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion/index.html)
- **上游算法:** [NVlabs/curobo](https://github.com/NVlabs/curobo)

## 为什么值得保留

- **部署态** cuRobo 能力的主入口；[ROBOTIS AI Worker cuMotion 集成](https://docs.robotis.com/docs/systems/aiworker/resources/technical_story/isaac_cumotion/) 与 Isaac ROS **5.0** 博客均以此为核心 manipulation 加速组件。

## 对 wiki 的映射

- [wiki/entities/curobo.md](../../wiki/entities/curobo.md)
- [wiki/entities/moveit2.md](../../wiki/entities/moveit2.md)
- [wiki/entities/robotis-ai-worker-isaac-cumotion.md](../../wiki/entities/robotis-ai-worker-isaac-cumotion.md)

---
- **录入日期:** 2026-09-24
