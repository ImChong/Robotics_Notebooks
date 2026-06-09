# PythonRobotics: a Python code collection of robotics algorithms（arXiv:1808.10703）

> 论文来源归档（ingest）

- **标题：** PythonRobotics: a Python code collection of robotics algorithms
- **类型：** paper / open-source / autonomous-navigation / education
- **arXiv：** <https://arxiv.org/abs/1808.10703> · PDF：<https://arxiv.org/pdf/1808.10703.pdf>
- **作者：** Atsushi Sakai, Daniel Ingram, Joseph Dinius, Karan Chawla, Antonin Raffin, Alexis Paques
- **TeX 源：** <https://github.com/AtsushiSakai/PythonRoboticsPaper>
- **代码仓库：** <https://github.com/AtsushiSakai/PythonRobotics>
- **入库日期：** 2026-06-09
- **一句话说明：** 2018 年介绍 PythonRobotics 开源项目的短文（8 页）：强调用 Python3 最小依赖实现常用自主导航算法，并配以动画帮助初学者建立直觉。

## 核心摘录（面向 wiki 编译）

### 1) 项目定位与目标读者

- **要点：** OSS 项目，聚焦 **autonomous navigation**；目标读者为 **机器人初学者**，通过可读代码理解各算法基本思想，而非追求工业级性能或 ROS 集成。
- **对 wiki 的映射：** [`wiki/entities/python-robotics.md`](../../wiki/entities/python-robotics.md)

### 2) 算法选型原则

- **要点：** 选取 **学界与工业界广泛使用** 的实用算法：如定位中的 Kalman / 粒子滤波、建图中的栅格地图、规划中的动态规划与采样法（RRT 等）、跟踪中的最优控制。
- **对 wiki 的映射：** 同上；与 [navigation-slam-autonomy-stack](../../wiki/overview/navigation-slam-autonomy-stack.md) 各层算法对照阅读。

### 3) 实现与可复现性

- **要点：** 每例 **Python3 + 标准科学计算库**；附带 **直观动画** 展示仿真行为；降低环境搭建门槛。
- **对 wiki 的映射：** [`sources/repos/python_robotics.md`](../repos/python_robotics.md)、[`sources/courses/python_robotics_textbook.md`](../courses/python_robotics_textbook.md)

### 4) 与工程导航栈的关系

- **要点：** 论文与仓库 **不提供 ROS/ROS 2 节点封装**；价值在于 **算法原型与教学**，工程落地需迁移到 Nav2、Autoware 等框架。
- **对 wiki 的映射：** [`wiki/entities/navigation2.md`](../../wiki/entities/navigation2.md)、[`wiki/overview/navigation-slam-autonomy-stack.md`](../../wiki/overview/navigation-slam-autonomy-stack.md)

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 与在线教材、GitHub README 模块列表交叉核对
