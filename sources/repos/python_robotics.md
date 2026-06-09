# PythonRobotics

> 来源归档

- **标题：** PythonRobotics
- **类型：** repo
- **链接：** https://github.com/AtsushiSakai/PythonRobotics
- **在线教材：** https://atsushisakai.github.io/PythonRobotics/
- **Stars：** ~29.7k（2026-06）
- **License：** MIT
- **入库日期：** 2026-06-09
- **一句话说明：** 面向自主导航算法入门的高星 Python 代码集 + Sphinx 在线教材：定位、建图、SLAM、路径规划、路径跟踪、机械臂与无人机示例，依赖极少、动画直观。
- **沉淀到 wiki：** [python-robotics](../../wiki/entities/python-robotics.md)、[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)

---

## 核心定位

PythonRobotics 是 Atsushi Sakai 等人维护的 **机器人算法教学型开源项目**，同时提供：

1. **可运行 Python3 示例代码**（每算法独立目录，最小依赖）
2. **Sphinx 在线教材**（与代码同仓库维护）
3. **动画 GIF / YouTube**（[PythonRoboticsGifs](https://github.com/AtsushiSakai/PythonRoboticsGifs)）

三大设计哲学：**易读**（Python + 直观可视化）、**实用**（学界与工业常用算法）、**最小依赖**（运行仅需 Python / NumPy / SciPy / Matplotlib / cvxpy）。

---

## 模块地图（与经典自主导航栈对应）

| 模块 | 代表算法 | 与工程栈关系 |
|------|----------|--------------|
| **Localization** | EKF、粒子滤波、直方图滤波 | 理解 AMCL / 融合定位的算法原型 |
| **Mapping** | 高斯栅格、射线投射、LiDAR→栅格、k-means 聚类 | 理解 occupancy grid 与感知预处理 |
| **SLAM** | ICP、FastSLAM 1.0 | 理解 scan matching 与粒子 SLAM 直觉 |
| **Path Planning** | DWA、Dijkstra/A*/D*/D* Lite、势场、PRM、RRT/RRT*、Frenet 最优轨迹、状态格 | Nav2 / Autoware 规划器的算法背景 |
| **Path Tracking** | Pure Pursuit、Stanley、LQR、MPC、C-GMRES NMPC | 局部控制器与 MPC 跟踪入门 |
| **Arm Navigation** | N 关节点到点、避障 | 与操作栈独立，偏运动学直觉 |
| **Aerial Navigation** | 四旋翼 3D 轨迹、火箭着陆 | 与多旋翼控制教材互补 |
| **Bipedal** | 倒立摆步态修正 | 与腿式/人形主线弱相关，仅作概念演示 |

---

## 运行与开发依赖

**运行示例：** Python 3.12+、NumPy、SciPy、Matplotlib、cvxpy。

**开发/文档：** pytest、pytest-xdist、mypy、Sphinx、ruff（或 pycodestyle）。

安装：`conda env create -f requirements/environment.yml` 或 `pip install -r requirements/requirements.txt`。

---

## 关联原始资料

- 项目论文：[arXiv:1808.10703](../papers/python_robotics_arxiv_1808_10703.md)
- 在线教材归档：[sources/courses/python_robotics_textbook.md](../courses/python_robotics_textbook.md)
- 动画仓库：<https://github.com/AtsushiSakai/PythonRoboticsGifs>
- 音频概览：<https://www.youtube.com/watch?v=uMeRnNoJAfU>

---

## 对 wiki 的映射

- 实体页：[python-robotics](../../wiki/entities/python-robotics.md)
- 导航栈总览（算法入门对照）：[navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md)
- 卡尔曼滤波形式化：[kalman-filter](../../wiki/formalizations/kalman-filter.md)
- 轨迹优化方法：[trajectory-optimization](../../wiki/methods/trajectory-optimization.md)
- Modern Robotics 教材（理论互补）：[modern-robotics-book](../../wiki/entities/modern-robotics-book.md)
