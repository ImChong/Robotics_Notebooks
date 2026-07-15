# 开源机器人学学习指南（qqfly）

> 课程 / 教程来源归档

- **标题：** 开源机器人学学习指南（Open Source Robotics Learning Guide）
- **类型：** course / tutorial
- **链接：** https://learn-robotics.qqfly.net/
- **英文版：** https://en.learn-robotics.qqfly.net/（AI 辅助翻译、作者校订）
- **源码仓库：** https://github.com/qqfly/how-to-learn-robotics
- **作者：** qqfly
- **许可：** CC BY 4.0
- **入库日期：** 2026-07-15
- **站点最近更新：** 2026-07-14（commit e2e41d3）
- **一句话说明：** 面向「非科班出身」读者的中文机器人学自学路线图：先修 → Craig 工业臂入门 → 现代机器人学 / 3D 视觉 / 自主规划进阶 → 配套编程与 ROS 实践；具身智能章节整理中。
- **沉淀到 wiki：** [learn-robotics-qqfly-guide](../../wiki/entities/learn-robotics-qqfly-guide.md)

---

## 为什么值得保留

- **补国内教学体系缺口**：前言从作者面试与博士经历出发，指出大陆多数机器人研究生缺乏运动学逆解、轨迹规划、控制闭环等系统训练；本书按自学路径组织，与本库 [运动控制成长路线](../../roadmap/motion-control.md) 的 L−1–L3 打底形成互补（本库偏人形 / 浮基；本书偏工业臂与传统规控）。
- **工业臂 + 规控主线清晰**：入门以 John Craig《Introduction to Robotics》+ Khatib 斯坦福视频为轴，覆盖 DH、雅可比数值 IK、牛顿-欧拉动力学、PID + 前馈、标定辨识；进阶三条线（Modern Robotics 李群语言、3D 视觉、C-Space 运动规划）与 Penn Robotics 专项、MoveIt、OpenCV 实践清单对齐。
- **实践导向强**：入门 / 进阶均配「Get your hands dirty」——Robotics Toolbox for Python、单轴伺服辨识、RobotStudio 示教、MoveIt、手眼标定等可执行清单。
- **开源可 PR**：MkDocs Material 站点 + GitHub 仓库，CC BY 4.0，便于社区修正与交叉引用。

---

## 站点结构（主干章节）

| 部分 | 页面 | 核心内容 |
|------|------|----------|
| 前言 | preface | 非科班背景、国内机器人教育碎片化问题 |
| 先修知识 | prerequisite | 英文、线代（Strang）、微积分、理论力学、Matlab/Python、控制基础、数电模电、单片机、Linux/C、3D 设计 |
| 入门 | getting-started | Craig 路线：空间变换、DH 正逆解、雅可比、动力学、PID/轨迹、标定辨识 |
| 入门实践 | dirty-your-hands | Robotics Toolbox 编程练习、比赛/硬件、ROS 2 学习路径 |
| 进阶导言 | advanced | Springer Handbook 索引；三条主线边界 |
| 现代机器人学 | modern-robotics | PoE/旋量、李群插值、姿态 Bezier、约束流形构造（端水搬运） |
| 3D 视觉 | 3d-vision | 标定、手眼 AX=XB、位姿估计、视觉伺服；SLAM 仅指路 |
| 自主规划 | motion-planning | C-Space、图搜索/优化/采样/学习、TOPP、零空间约束规划、经验路图、MPC 接口 |
| 具身智能 | embodied-ai | **整理中**；暂迁 ML / RL 入门（Sutton、吴恩达、Hinton） |
| 进阶实践 | advanced-practice | Penn Robotics 专项、Corke 视觉控制、姿态/标定/MoveIt 动手清单 |
| 勇者斗恶龙 | dragon-quest | （站点保留章节，待读） |
| 参考文献 | references | Craig、Khalil、Springer Handbook、Modern Robotics、Choset、LaValle、Sutton 等 |

---

## 核心教材与外部资源映射

| 本书章节引用 | 资料 | 本库相关页 |
|-------------|------|-----------|
| [1] Craig | *Introduction to Robotics* | 入门 DH / PID / 工业臂直觉 |
| [2] Khalil | *Modeling, Identification and Control of Robots* | [system-identification](../../wiki/concepts/system-identification.md) |
| [3] Springer Handbook | 各领域章节索引 | 进阶查阅 |
| [5] Lynch & Park | *Modern Robotics* | [modern-robotics-book](../../wiki/entities/modern-robotics-book.md) |
| [8] Choset et al. | *Principles of Robot Motion* | C-Space 理论 |
| [9] LaValle | *Planning Algorithms* | 规划算法 |
| [13] Sutton & Barto | *Introduction to RL* | [reinforcement-learning](../../wiki/methods/reinforcement-learning.md) |

**配套公开课：** Penn Coursera Robotics 专项（Perception / Estimation / Aerial / Motion Planning / Mobility）、Modern Robotics Coursera、Strang 线代、Brian Douglas 控制理论。

**工具链：** Peter Corke Robotics Toolbox、OpenCV 标定、PCL、MoveIt、ViSP、ROS 2 Humble+。

---

## 对 wiki 的映射

- 实体页：[learn-robotics-qqfly-guide](../../wiki/entities/learn-robotics-qqfly-guide.md)
- 传统教材：[modern-robotics-book](../../wiki/entities/modern-robotics-book.md)、[python-robotics](../../wiki/entities/python-robotics.md)
- 运动控制路线：[motion-control](../../roadmap/motion-control.md) L−1–L3 互补阅读
- 概念： [ros2-basics](../../wiki/concepts/ros2-basics.md)
- 方法： [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[trajectory-optimization](../../wiki/methods/trajectory-optimization.md)、[model-predictive-control](../../wiki/methods/model-predictive-control.md)
- 实体： [moveit2](../../wiki/entities/moveit2.md)
