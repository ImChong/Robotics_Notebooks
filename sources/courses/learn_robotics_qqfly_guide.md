# 开源机器人学学习指南（qqfly）

> 课程 / 教程来源归档

- **标题：** 开源机器人学学习指南（Open Source Robotics Learning Guide）
- **类型：** course / tutorial
- **链接：** https://learn-robotics.qqfly.net/
- **英文版：** https://en.learn-robotics.qqfly.net/（AI 辅助翻译、作者校订）
- **源码仓库：** https://github.com/qqfly/how-to-learn-robotics
- **作者：** qqfly
- **许可：** CC BY 4.0
- **入库日期：** 2026-07-15（2026-11-07 逐章通读复核）
- **版本：** GitHub `qqfly/how-to-learn-robotics` master 分支 `docs/*.md` 源文件（2026-11-07 经 raw.githubusercontent.com 获取；GitHub API 在本环境不可达，未能核对 commit，沿用上条记载的 e2e41d3 仅作参考）
- **核对范围：** 通读 `README.md`、`mkdocs.yml` 头部配置，以及 `docs/` 下全部正文章节：index、preface、prerequisite、getting-started、dirty-your-hands、advanced、modern-robotics、3d-vision、motion-planning、embodied-ai、advanced-practice、dragon-quest、references；`license.md`、英文站与 `images/` 图片、各章外链视频 / 教材本身未读。未运行任何练习。
- **通读：** 已完成
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

## 2026-11-07 通读复核补充（出自各章原文）

- **先修（12 条）：** 英文、线代（Strang）、微积分（Strang《Highlights of Calculus》）、理论力学、Matlab/Python、控制理论（Brian Douglas）、数电模电、单片机、Linux 与 C、3D 设计、B 站、AI 工具；作者自述这些大致是「自动化 / 机电专业大三学生应有水平」。
- **入门（Craig + Khatib 视频）：** 空间变换 → Modified DH 四参数（$a_{i-1},\alpha_{i-1},d_i,\theta_i$）→ 正逆解（要求亲手推导并编程实现六轴逆解）→ 雅可比（$\dot x=J\dot q$，数值逆解 $\Delta q=J^\dagger\Delta x$，列出四个缺点：慢、单解、依赖初值、奇异点不收敛）→ 动力学（作者称 80% 的人在此放弃；建议会拉格朗日三轴、牛顿-欧拉编程实现）→ 控制（PID → 轨迹规划 T / S 曲线 → 动力学前馈 + PID）→ 标定与辨识（多点触碰运动学标定 $\Delta p=J\Delta\phi$；动力学对参数线性 $\tau=A(q,\dot q,\ddot q)\Theta$ 最小二乘，注意激励轨迹须满秩）。
- **作者自述实验：** 单轴直驱平台上，SolidWorks 读取的参数算出的力矩与实测误差约 50%，辨识后降到 8% 以内；并用电流估计外力矩做碰撞检测与拖动示教。大减速比（100 kg 负载、减速比 86 的行星传动）下不建模摩擦则完全无法拟合——「摩擦力，入坑需谨慎」，因此 KUKA iiwa、Franka 等在关节加力矩传感器。
- **入门实践：** 用 Peter Corke 的 Robotics Toolbox for Python 对答案（`fkine` / `ikine` / `rne`）；比赛建议有两次完整经历后转向理论；ROS 直接学 ROS 2（Noetic 已于 2025 年停止维护），先补 Linux、Git、C++、数据结构。
- **进阶三线：** 现代机器人学（位置属欧式空间而姿态不是；⊕ / ⊖ / ⊙ 群语言；统一插值 $A(u)=A_1\oplus u\odot((\ominus A_1)\oplus A_2)$；李群 Bezier 过渡；平均旋转；约束流形 $R(3)\oplus SO(3)$ 搬水）；3D 视觉（物体识别 / 位姿估计 / 标定三件事，手眼 $AX=XB$，位姿估计从模板、SIFT、LineMod 到 ICP / NDT，视觉伺服闭环，SLAM 仅提及）；自主规划（示教再现 → 任务级编程；路径 / 轨迹 / 运动规划区分；完备性与最优性；C-Space 与 $60^6\approx4.7\times10^{10}$ 的离散化代价；图搜索 / 优化 / 采样 / 学习四类；时间参数化；零空间 $\dot q=J^\dagger\dot x+(I-J^\dagger J)v$ 与流形约束三类做法；经验路图与运动策略、MPC；MDP 视角）。
- **进阶实践：** Penn Robotics 专项五门（Perception / Estimation and Learning / Aerial Robotics / Computational Motion Planning / Mobility）+ Corke《Robotics: Vision and Control》；动手清单 5 项：姿态插值、平均旋转、相机与手眼标定、点云滤波、MoveIt 规划。
- **具身智能章：** 标注「本部分整理中」；现有内容仅为机器学习（吴恩达、Hinton、PyTorch）与强化学习（Sutton 教材、Bellman 方程、MDP 可视作运动规划的另一种表述）入门，作者「坚信强化学习是机器人的未来方向」并举宇树等足式机器人为例。
- **勇者斗恶龙章：** 提醒「真空中的球形龙」——学到的东西离工业级强度还远，ROS 崩溃、Eigen Alignment、缺少好用 3D 传感器等；强调团队与工程化。
