# modern_robotics_textbook

> 来源归档（ingest）

- **标题：** Modern Robotics: Mechanics, Planning, and Control
- **类型：** book / textbook
- **作者：** Kevin M. Lynch (Northwestern University) & Frank C. Park (Seoul National University)
- **出版：** Cambridge University Press, 2017
- **ISBN：** 9781107156302
- **官方资源：** https://hades.mech.northwestern.edu/index.php/Modern_Robotics
- **PDF 镜像：** https://hades.mech.northwestern.edu/images/7/7f/MR.pdf
- **配套 Coursera 专项课程：** Modern Robotics 6 门课
- **入库日期：** 2026-04-30
- **沉淀到 wiki：** 是 → [`wiki/entities/modern-robotics-book.md`](../../wiki/entities/modern-robotics-book.md)

## 一句话说明

经典的现代机器人学教材，独特之处是**全程使用李群（Lie group）/ 螺旋理论（screw theory）**作为统一数学语言来描述刚体运动、运动学、动力学与控制，覆盖从配置空间到全身控制、抓取、移动机器人的完整体系。

## 为什么值得保留

1. **唯一系统教完螺旋理论的本科级教材**：传统教材（Craig、Spong、Siciliano）多用 D-H 参数；本书用 PoE（Product of Exponentials）公式，更几何、更简洁。
2. **配套 Coursera 专项课程 + Python/MATLAB/Mathematica 库**：可以代码验证。
3. **覆盖广度**：13 章覆盖配置空间 → 运动学 → 动力学 → 轨迹 → 规划 → 控制 → 抓取 → 移动机器人，是 RL/LLM 时代之前传统机器人学的「最大公约数」教材。
4. **官方 PDF 永久免费**：作者直接提供，无需购买。

## 章节目录

| 章节 | 标题 | 主要主题 |
|------|------|---------|
| Ch 1 | Preview | 全书路线图 |
| Ch 2 | Configuration Space | 自由度、C-space、拓扑、约束 |
| Ch 3 | **Rigid-Body Motions** | SO(3)/SE(3)、旋转矩阵、齐次变换、**twist / wrench / 指数坐标** |
| Ch 4 | **Forward Kinematics** | **PoE 公式（Product of Exponentials）**、空间形式 vs 物体形式 |
| Ch 5 | **Velocity Kinematics and Statics** | 雅可比矩阵（空间/物体）、奇异性、可操作度椭球 |
| Ch 6 | Inverse Kinematics | 解析解、数值解（牛顿-拉夫森） |
| Ch 7 | Kinematics of Closed Chains | 并联机构、Stewart 平台、Grübler 公式 |
| Ch 8 | **Dynamics of Open Chains** | Lagrange / Newton-Euler、惯性矩阵 M(q)、动力学方程 |
| Ch 9 | Trajectory Generation | 多项式时间律、最短路径、最优时间轨迹 |
| Ch 10 | Motion Planning | C-space 障碍、网格、采样规划（PRM/RRT）、虚拟势场 |
| Ch 11 | **Robot Control** | 关节空间 PD/PID、计算力矩控制、任务空间控制、力控、阻抗控制 |
| Ch 12 | Grasping and Manipulation | 接触模型、力封闭、形封闭、操作平面 |
| Ch 13 | Wheeled Mobile Robots | 全向 / 非完整、里程计、控制 |

加粗章节是与本知识库现有 wiki 页面**直接对应**的核心章节。

## 关键术语对应

教材主要术语已在对应 wiki 实体页 [`wiki/entities/modern-robotics-book.md`](../../wiki/entities/modern-robotics-book.md) 的"章节地图"中给出全部映射。本 source 文件不再单独列出具体 wiki 链接，避免双重维护。

## 与其他教材的关系

| 维度 | Modern Robotics (Lynch-Park) | Craig | Siciliano | Featherstone |
|------|------------------------------|-------|-----------|--------------|
| 数学语言 | **李群 / 螺旋理论** | D-H 参数 | 兼有 | 空间向量代数 |
| 本科可读 | ✅ | ✅ | △ | ❌（研究级） |
| 涵盖动力学 | 中等 | 浅 | 深 | 极深（递推算法） |
| 覆盖控制 | 中等 | 中等 | 深 | ❌ |
| 涵盖规划/抓取 | ✅ | ❌ | △ | ❌ |
| 配套代码 | ✅ Python/MATLAB | ❌ | △ | ❌ |
| 在线视频 | ✅ Coursera | ❌ | ❌ | ❌ |

## 当前提炼状态

- [x] 章节目录与教材定位
- [x] 与本项目已有 wiki 页面的术语对照
- [ ] 后续可专题深挖：PoE Forward Kinematics 形式化、Wheeled Mobile Robots 单页（当前 wiki 未覆盖）、传统采样规划（RRT/PRM）单页
