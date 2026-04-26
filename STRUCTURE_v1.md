# 机器人技术栈项目目录结构 v1

> 版本：v1.0
> 日期：2026-04-08
> 定位：面向机器人全栈成长路径的技术栈导航项目
> 终极目标：全栈机器人工程师
> 当前切入口：机器人运动控制算法工程师

---

## 一、项目定位

本项目以**机器人运动控制**为切入口，重点覆盖强化学习、模仿学习与人形机器人相关能力，逐步扩展至感知、规划、软件系统与整机集成，目标是构建通向机器人全栈工程实践的成长地图。

**一句话定位：**
> 面向人形机器人运动控制、强化学习与模仿学习的技术栈地图与学习路线图，最终通向机器人全栈工程能力。

---

## 二、顶层结构（三大模块）

```
Robotics_Notebooks/
├── wiki/                   # 🌟 核心：结构化知识页面（持续建设区）
├── sources/                # 原始资料与资源索引（沉淀区）
├── roadmap/                # 🆕 v1：成长路线与阶段导航
├── tech-map/               # 🆕 v1：技术栈地图与依赖关系
├── references/            # 🆕 v1：论文导航、开源生态、项目索引
├── exports/                # 网页/思维导图数据导出层
├── docs/                   # GitHub Pages 静态网页输出
├── schema/                 # 知识库维护规范
├── log.md                  # 重构变更日志
├── README.md               # 项目主入口
└── index.md                # 知识索引首页
```

---

## 三、目录结构详表

### 3.1 `wiki/` — 核心知识页面

按"内圈主攻 + 外圈拓展"组织，内圈打深，外圈逐步扩充骨架。

#### 内圈：当前主攻区（重点建设）

```
wiki/
├── math/                          # 数学基础
│   ├── linear-algebra.md          # 线性代数（机器人场景重点）
│   ├── calculus.md                # 微积分
│   ├── probability-statistics.md  # 概率统计
│   └── optimization.md            # 优化基础
│
├── robotics-fundamentals/         # 机器人学基础骨架
│   ├── rigid-body-motion/        # 刚体运动与坐标系变换
│   │   ├── rotation-representation.md      # 旋转表示（欧拉角/四元数/旋量）
│   │   ├── homogeneous-transform.md        #齐次变换
│   │   └── twist-wrench.md                # Twist & Wrench
│   ├── kinematics/               # 运动学
│   │   ├── forward-kinematics.md          # 正运动学
│   │   ├── inverse-kinematics.md          # 逆运动学
│   │   └── differential-kinematics.md      # 微分运动学（雅可比）
│   ├── dynamics/                 # 动力学 ⭐ 对控制极其重要
│   │   ├── newton-euler.md               # 牛顿欧拉法
│   │   ├── lagrangian.md                 # 拉格朗日法
│   │   ├── floating-base-dynamics.md      # 浮动基动力学
│   │   └── contact-modeling.md           # 接触建模
│   └── state-estimation/          # 状态估计
│       ├── kalman-filter.md               # 卡尔曼滤波
│       ├── complementary-filter.md        # 互补滤波
│       └── imu-fusion.md                  # IMU数据融合
│
├── control/                        # 控制主线
│   ├── classical-control/          # 经典控制
│   │   ├── pid.md                         # PID控制
│   │   └── lqr.md                         # LQR线性二次调节
│   ├── optimal-control/            # 最优控制 ⭐
│   │   ├── ocp-formulation.md             # OCP问题构建
│   │   ├── ipm-qp.md                      # 内点法与QP求解
│   │   └── direct-indirect.md             # 直接法与间接法
│   ├── mpc/                        # 模型预测控制 ⭐⭐
│   │   ├── linear-mpc.md                   # 线性MPC
│   │   ├── nonlinear-mpc.md                # 非线性MPC
│   │   └── convex-mpc.md                   # 凸MPC（人形常用）
│   ├── whole-body-control/         # 全身控制 ⭐⭐
│   │   ├── task-space-control.md           # 任务空间控制
│   │   ├── tsid.md                         # 任务空间逆动力学（TSID）
│   │   └── qp-wbc.md                       # QP式全身控制
│   └── humanoid-locomotion/        # 人形运动控制 ⭐⭐⭐ 核心出口
│       ├── bipedal-walking.md              # 双足步行
│       ├── gait-balance.md                 # 步态与平衡
│       ├── push-recovery.md                # 扰动恢复
│       ├── centroidal-dynamics.md           # 质心动力学
│       └── whole-body-locomotion.md         # 全身步态综合
│
├── reinforcement-learning/         # 强化学习主线 ⭐⭐
│   ├── rl-foundations/            # RL基础
│   │   ├── mdp-value-policy.md            # MDP/价值函数/策略
│   │   ├── on-policy-off-policy.md         # on/off-policy
│   │   ├── ppo.md                          # PPO（机器人控制最常用）
│   │   ├── sac.md                          # SAC
│   │   └── td3-sac.md                      # TD3/SAC
│   ├── rl-for-robotics/           # 机器人RL
│   │   ├── rl-locomotion.md               # RL运动控制
│   │   ├── sim2real.md                    # sim2real核心方法
│   │   ├── domain-randomization.md         # 域随机化
│   │   └── reward-design.md                # Reward设计
│   └── advanced-rl/                # 前沿方向
│       ├── model-based-rl.md               # 基于模型的RL
│       ├── offline-rl.md                   # Offline RL
│       └── hierarchical-rl.md               # 分层RL
│
├── imitation-learning/             # 模仿学习主线 ⭐⭐
│   ├── behavior-cloning.md                # 行为克隆
│   ├── dagger.md                          # DAgger
│   ├── generative-models/           # 生成式方法
│   │   ├── gail.md                         # GAIL
│   │   └── diffusion-policy.md             # Diffusion Policy
│   ├── motion-retarget/            # 运动重定向 ⭐⭐⭐
│   │   ├── smpl-modeling.md               # SMPL人体模型
│   │   ├── retarget-overview.md           # 重定向方法概览
│   │   └── retarget-practice.md           # 工程实践
│   └── contact-retarget.md                  # 接触感知的重定向
│
└── humanoid-special/               # 人形机器人专项 ⭐⭐⭐
    ├── humanoid-overview.md               # 人形机器人概述
    ├── hardware-architecture.md          # 硬件架构认知
    ├── actuator-drive.md                  # 执行器与驱动
    ├── sensor-suite.md                    # 传感器配置
    ├── whole-body-skill.md                # 全身技能学习
    └── deployment-practice.md             # 部署实践
```

#### 外圈：全栈延展区（骨架先行，内容逐步补）

```
wiki/ (continued)
├── perception/                     # 感知（全栈扩展）
│   ├── vision-basics.md                   # 视觉基础
│   ├── depth-perception.md                # 深度感知
│   ├── point-cloud.md                     # 点云处理
│   └── tactile-sensing.md                 # 触觉感知
│
├── planning/                        # 规划（全栈扩展）
│   ├── motion-planning.md                  # 运动规划
│   ├── task-planning.md                   # 任务规划
│   └── navigation.md                       # 导航
│
├── software-stack/                  # 软件工程（全栈扩展）
│   ├── ros2-basics.md                      # ROS2基础
│   ├── ros2-control.md                     # ros2_control
│   ├── middleware.md                        # 中间件与通信
│   ├── real-time-systems.md                # 实时系统
│   └── testing-debugging.md                # 测试与调试
│
├── system-integration/               # 系统集成（全栈扩展）
│   ├── hardware-interface.md                # 硬件接口
│   ├── calibration.md                       # 标定
│   ├── safety-reliability.md               # 安全性与可靠性
│   └── deployment-pipeline.md               # 部署流水线
│
└── benchmarks/                      # Benchmark与评测
    ├── locomotion-benchmarks.md            # 运动控制Benchmark
    ├── rl-benchmarks.md                    # RL评测环境
    └── humanoid-benchmarks.md              # 人形机器人Benchmark
```

---

### 3.2 `roadmap/` — 成长路线导航 🆕 v1

```
roadmap/
├── README.md                        # 路线总览
│
├── motion-control.md        # 主路线：运动控制算法工程师成长路线 ⭐
│   # 按学习阶段排列，从数学基础到人形机器人RL
│   # 每阶段含：目标 + 核心模块 + 最小落地任务 + 推荐资源
│
└── learning-paths/                  # 按目标分支的学习路径
    ├── if-goal-locomotion-rl.md     # 目标：人形RL运动控制
    ├── if-goal-imitation-learning.md # 目标：模仿学习与技能迁移
    ├── if-goal-whole-body-control.md # 目标：全身控制与优化
    └── if-goal-generalist.md       # 目标：全栈通用能力
```

**主路线（运动控制切入口）学习阶段：**

| 阶段 | 主题 | 核心模块 | 最小落地任务 |
|------|------|----------|--------------|
| L0 | 数学基础 | 线代、微积分、概率、优化 | 能推导雅可比矩阵、搞懂概率分布 |
| L1 | 机器人学骨架 | 刚体运动、运动学、动力学 | 用Pinocchio跑通一个人形模型正逆动力学 |
| L2 | 经典控制 | PID、LQR、MPC、QP | 写一个倒立摆的LQR控制器 |
| L3 | 人形运动控制 | 质心动力学、WBC、人形locomotion | 跑通一个平坦地面行走baseline |
| L4 | 强化学习 | PPO/SAC、sim2real、domain randomization | 用IsaacGym训练一个人形行走策略 |
| L5 | 模仿学习 | BC/DAgger/Diffusion Policy、Retarget | 把一个人体动作数据集重定向到人形模型 |
| L6 | 综合实战 | 完整训练+部署闭环 | 训练+sim2real部署一个简单技能 |

---

### 3.3 `tech-map/` — 技术栈地图 🆕 v1

```
tech-map/
├── README.md                        # 技术栈地图说明
│
├── overview.md                      # 全栈技术域总览
│   # 图：数学 → 机器人基础 → 控制 → 学习 → 系统 → 全栈
│
├── dependency-graph.md              # 模块依赖关系图 ⭐ 核心价值
│   # 说明各模块之间的前置依赖和学习顺序
│
├── modules/                         # 各模块的标准化信息卡
│   ├── module-template.md           # 统一模板（解释/前置/产出/资源/难度）
│   │
│   ├── math/
│   │   └── linear-algebra.md        # 线代在机器人场景的重点标注
│   │
│   ├── robotics/
│   │   ├── rigid-body-motion.md
│   │   ├── kinematics.md
│   │   └── dynamics.md
│   │
│   ├── control/
│   │   ├── mpc.md
│   │   ├── whole-body-control.md
│   │   └── humanoid-locomotion.md
│   │
│   ├── rl/
│   │   ├── ppo.md
│   │   ├── sim2real.md
│   │   └── humanoid-rl.md
│   │
│   ├── il/
│   │   ├── behavior-cloning.md
│   │   ├── diffusion-policy.md
│   │   └── motion-retarget.md
│   │
│   └── system/
│       ├── ros2.md
│       ├── simulation.md
│       └── deployment.md
│
└── research-directions/            # 研究方向导航 ⭐
    # 按问题组织，而非按学科
    # 例如：如何让人形走得更稳？如何提升扰动恢复？
    # 每个问题下挂：关键方法、代表论文、常用Benchmark
```

---

### 3.4 `references/` — 论文导航与开源生态 🆕 v1

```
references/
├── README.md
│
├── papers/                          # 论文分类索引
│   ├── locomotion-rl.md             # locomotion RL 论文列表
│   ├── imitation-learning.md        # 模仿学习论文列表
│   ├── whole-body-control.md        # 全身控制论文列表
│   ├── humanoid-hardware.md         # 人形机器人硬件相关
│   ├── sim2real.md                 # sim2real论文列表
│   └── survey-papers.md             # 综述论文
│
├── repos/                          # 开源项目索引
│   ├── simulation.md                # 仿真平台
│   ├── rl-frameworks.md             # RL训练框架
│   ├── humanoid-projects.md         # 人形机器人相关项目
│   ├── retarget-tools.md            # 重定向工具
│   └── utilities.md                 # 工具库
│
└── benchmarks/                     # Benchmark索引
    ├── locomotion-benchmarks.md
    └── humanoid-environments.md
```

---

### 3.5 `sources/` — 原始资料（保持现状，逐步扩充）

```
sources/
├── README.md                        # 资源总索引
├── blogs/                           # 技术博客
├── courses/                         # 课程
├── notes/                           # 笔记归档
├── papers/                          # 论文归档
├── repos/                           # 仓库索引
└── videos/                          # 视频课程
```

---

### 3.6 `exports/` — 网页/思维导图数据导出

```
exports/
# 未来存放 JSON/YAML 格式的标准化知识数据
# 用于渲染成交互式技术栈网页或思维导图
# 保持结构化，便于前端消费
```

---

### 3.7 `schema/` — 知识库维护规范

```
schema/
├── naming.md                        # 命名规范
├── page-types.md                    # 页面类型定义
├── linking.md                       # 页面互链规范
└── ingest-workflow.md               # 新内容入库流程
```

---

### 3.8 `docs/` — GitHub Pages 静态网页输出

```
docs/
# Jekyll/Docusaurus/VitePress 构建输出目录
# 暂保持现状，不改动
```

---

## 四、Wiki 模块标准模板

每个 wiki 知识页统一使用以下结构：

```markdown
# 模块名称

## 这个模块解决什么问题
（1-2句话）

## 为什么对人形机器人重要
（和你的主线目标强绑定）

## 前置依赖
- 依赖模块A
- 依赖模块B

## 核心内容
### 子主题1
### 子主题2

## 学完应该会什么
（能做什么，不能做什么）

## 最小落地任务
（1-2个可执行的任务）

## 推荐资源
- 课程/书籍
- 论文
- 开源代码

## 相关模块
（关联的其他wiki页面）

## 难度 / 优先级
- 难度：X/5
- 优先级：X/5（当前切入口视角）
```

---

## 五、执行优先级（v1 阶段）

### 第一批（核心，切入口相关）
1. 创建 `roadmap/` 目录及单一主路线
2. 创建 `tech-map/dependency-graph.md`
3. 创建 `wiki/control/humanoid-locomotion/` 相关页面
4. 创建 `wiki/reinforcement-learning/rl-for-robotics/sim2real.md`
5. 创建 `wiki/imitation-learning/motion-retarget/`
6. 更新 `README.md` 反映新结构

### 第二批（全栈扩展骨架）
7. 创建 `wiki/perception/` 基础页面
8. 创建 `wiki/planning/` 基础页面
9. 创建 `wiki/software-stack/` 基础页面
10. 创建 `references/` 论文导航

### 第三批（充实内容）
11. 各模块详细内容页
12. `sources/` 资源整理
13. `exports/` 数据导出格式设计

---

## 六、与另外两个项目的边界

| 项目 | 职责 |
|------|------|
| **Humanoid_Robot_Learning_Paper_Notebooks** | 单篇论文深读笔记 |
| **Robotics_Notebooks** | 跨模块知识组织、学习路线、技术栈地图 |
| **ImChong.github.io** | 个人简历与对外展示 |

> 笔记项目负责"点"，技术栈项目负责"线和面"，简历项目负责"展示"。

---

*本文件为 v1 结构规范，后续按 roadmap 执行逐步落地。*
*更新日志见 `log.md`*
