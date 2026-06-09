# PythonRobotics 在线教材（Sphinx Textbook）

> 课程 / 教程来源归档

- **标题：** PythonRobotics documentation（在线教材）
- **类型：** course / tutorial
- **链接：** https://atsushisakai.github.io/PythonRobotics/
- **源码仓库：** https://github.com/AtsushiSakai/PythonRobotics（`docs/` + Sphinx 构建）
- **入库日期：** 2026-06-09
- **一句话说明：** 与 PythonRobotics 代码同仓库维护的 HTML 教材：按定位→建图→SLAM→规划→跟踪→机械臂→无人机组织，配动画与最小公式，适合算法直觉建立。
- **沉淀到 wiki：** [python-robotics](../../wiki/entities/python-robotics.md)

---

## 为什么值得保留

- **代码与文档一体**：算法实现与讲解同步更新，避免「文档漂移」。
- **模块完整覆盖经典移动机器人栈**：从 KF 定位到 Frenet 轨迹、Stanley/MPC 跟踪，是 Nav2 等工程栈的 **算法预习**。
- **低门槛**：相对 *Probabilistic Robotics* 与 *Modern Robotics* 大部头，更适合先跑通仿真再读理论。

---

## 教材目录结构（主干）

| 章节 | 主题 | 典型算法 |
|------|------|----------|
| Getting Started | 项目哲学、运行方式 | — |
| Localization | 定位 | EKF、粒子滤波、直方图滤波 |
| Mapping | 建图 | 栅格地图、LiDAR 转栅格、聚类 |
| SLAM | 同步定位建图 | ICP、FastSLAM |
| Path Planning | 路径规划 | DWA、A*/D*、PRM、RRT*、Frenet |
| Path Tracking | 路径跟踪 | Stanley、LQR、MPC |
| Arm Navigation | 机械臂 | N 关节 IK、避障 |
| Aerial Navigation | 空中 | 四旋翼轨迹、着陆 |
| Bipedal | 双足 | 倒立摆步态修正 |

动画资源独立存放于 [PythonRoboticsGifs](https://github.com/AtsushiSakai/PythonRoboticsGifs)。

---

## 对 wiki 的映射

- 实体页：[python-robotics](../../wiki/entities/python-robotics.md)
- 导航栈总览：[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
- 状态估计概念：[state-estimation](../../wiki/concepts/state-estimation.md)
- 轨迹优化：[trajectory-optimization](../../wiki/methods/trajectory-optimization.md)
