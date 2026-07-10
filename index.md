# Robotics Notebooks Index

本项目以运动控制为切入口，通通向机器人全栈工程能力。

这是知识入口总索引。**如果你第一次来，从 [README](README.md) 开始**，那里有完整的使用说明。

---

## 快速入口

| 你的目标 | 从这里进入 |
|---------|-----------|
| 想有一条学习路线照着走 | [主路线：运动控制成长路线](roadmap/motion-control.md) |
| 想用强化学习做 locomotion | [RL 纵深路线](roadmap/depth-rl-locomotion.md) |
| 想学模仿学习与技能迁移 | [模仿学习纵深路线](roadmap/depth-imitation-learning.md) |
| 想学安全控制（CLF/CBF）| [安全控制纵深路线](roadmap/depth-safe-control.md) |
| 想做接触丰富的操作任务 | [接触操作纵深路线](roadmap/depth-contact-manipulation.md) |
| 想看知识概念和方法 | [浏览完整页面目录](catalog.md) |
| 想看模块关系和依赖 | [tech-map 总览](tech-map/overview.md) |

---

## 四个模块的分工

如果你想知道每个目录负责什么：

### wiki/ — 结构化知识页
回答"某个概念或方法是什么"。

重点页面：
- [Optimal Control (OCP)](wiki/concepts/optimal-control.md)
- [LIP / ZMP](wiki/concepts/lip-zmp.md)
- [Centroidal Dynamics](wiki/concepts/centroidal-dynamics.md)
- [TSID](wiki/concepts/tsid.md)
- [Whole-Body Control](wiki/concepts/whole-body-control.md)
- [State Estimation](wiki/concepts/state-estimation.md)
- [System Identification](wiki/concepts/system-identification.md)
- [Floating Base Dynamics](wiki/concepts/floating-base-dynamics.md)
- [Contact Dynamics](wiki/concepts/contact-dynamics.md)
- [Capture Point / DCM](wiki/concepts/capture-point-dcm.md)
- [Sim2Real](wiki/concepts/sim2real.md)
- [Reinforcement Learning](wiki/methods/reinforcement-learning.md)
- [PyTorch](wiki/entities/pytorch.md)
- [TensorFlow](wiki/entities/tensorflow.md)
- [Imitation Learning](wiki/methods/imitation-learning.md)
- [World Action Models（WAM）](wiki/concepts/world-action-models.md)（联合未来–动作分布的具身策略范式；综述与 Awesome-WAM 资源入口）
- [Pelican-Unified 1.0（UEI）](wiki/methods/pelican-unified-1.md)（Qwen3-VL 推理末态 \(z\) + Wan 系 UFG：同一扩散去噪联合未来视频与动作；arXiv:2605.15153）
- [mimic-video（VAM）](wiki/methods/mimic-video.md)（互联网视频潜计划 + 流匹配动作解码；arXiv:2512.15692）
- [DeFI（解耦前向/逆动力学 VLA）](wiki/methods/defi-decoupled-dynamics-vla.md)（GFDM + GIDM 分阶段预训练；arXiv:2604.16391）
- [MINT（Mimic Intent, Not Just Trajectories）](wiki/entities/paper-mint-vla.md)（RSS 2026：频域意图分词 + 单样本迁移；arXiv:2602.08602）
- [EgoScale](wiki/methods/egoscale.md)（2 万小时级 egocentric 人视频预训练 VLA + 对齐 mid-training；arXiv:2602.16710）
- [ENPIRE](wiki/methods/enpire.md)（coding agent 真机策略自改进闭环：EN–PI–R–E 四模块 + AutoEnvBench；NVIDIA GEAR 2026）
- [Model Predictive Control (MPC)](wiki/methods/model-predictive-control.md)
- [Trajectory Optimization](wiki/methods/trajectory-optimization.md)
- [Locomotion](wiki/tasks/locomotion.md)
- [WBC vs RL](wiki/comparisons/wbc-vs-rl.md)
- [Isaac Gym / Isaac Lab](wiki/entities/isaac-gym-isaac-lab.md)
- [Robotic World Model（ETH RSL，RWM / RWM-U）](wiki/entities/robotic-world-model-eth-rsl.md)（Isaac Lab 扩展 + Lite 离线仓；集成 RNN 动力学与想象 rollout 的 MBRL 参考实现）
- [SAGE（执行器 Sim2Real 间隙估计）](wiki/entities/sage-sim2real-actuator-gap-estimator.md)（Isaac 重放与真机对齐、关节级 gap 指标与成对数据）
- [Genesis (仿真器)](wiki/entities/genesis-sim.md)
- [MuJoCo](wiki/entities/mujoco.md)
- [dm_control / Control Suite](wiki/entities/dm-control.md)
- [MuJoCo MJX](wiki/entities/mujoco-mjx.md)（JAX/XLA 版 MuJoCo，GPU 批量与可微 rollout）
- [Brax](wiki/entities/brax.md)（JAX 可微物理与 RL 训练；与 MJX、MuJoCo Playground 的官方组合指引）
- [legged_gym](wiki/entities/legged-gym.md)
- [Pinocchio](wiki/entities/pinocchio.md)
- [AprilTag](wiki/entities/april-tag.md)
- [Agent Reach](wiki/entities/agent-reach.md)（编码代理互联网接入脚手架：CLI + doctor + 可插拔渠道；上游 yt-dlp / gh / MCP 等）
- [AnyGrasp（抓取感知 SDK）](wiki/entities/anygrasp.md)
- [MoveIt 2](wiki/entities/moveit2.md)（ROS 2 机械臂运动规划、Planning Scene、OMPL/Pilz/CHOMP 与 MTC pick-and-place）
- [Grasp Pose Estimation（抓取位姿估计）](wiki/methods/grasp-pose-estimation.md)（6-DoF 抓取检测谱系：GraspNet → Contact-GraspNet → GSNet/Graspness/AnyGrasp，含点云/RGBD 输入与评测指标）
- [Crocoddyl](wiki/entities/crocoddyl.md)
- [cuRobo](wiki/entities/curobo.md)
- [Unitree](wiki/entities/unitree.md)
- [市面知名机器人平台纵览](wiki/overview/notable-commercial-robot-platforms.md)（人形 / 四足高频品牌索引）
- [机器人开源宝库（微信策展第01期）](wiki/overview/robot-open-source-wechat-issue01-curator.md)（10 个开源整机/平台：傅利叶 N1、智元 X1、天工、ODRI、BHL、Orca、TurtleBot3、ROBOTIS 等）
- [机器人开源宝库（微信策展第02期）](wiki/overview/robot-open-source-wechat-issue02-curator.md)（Reachy2、Poppy、InMoov、Doggo/Pupper、myCobot 320、myAGV、TidyBot2、Kinova Gen3、Franka R3、PAROL6）
- [四足机器人（Quadruped Robot）](wiki/entities/quadruped-robot.md)
- [轮足四足机器人（四轮足）](wiki/concepts/wheel-legged-quadruped.md)（Go2W 类混合滚动–步态）
- [MotionCode™](wiki/entities/motioncode.md)（人体运动数据与 Mind 线人形训练叙事）
- [HumanNet](wiki/entities/humannet.md)（百万小时级人中心视频语料；VLA/IL 人类侧预训练参照）
- [Xiaomi-Robotics-0](wiki/entities/xiaomi-robotics-0.md)（小米开源 VLA；Qwen3-VL + DiT flow matching，异步 action chunk 部署）
- [Project Instinct](wiki/entities/project-instinct.md)（人形全身动态控制研究站群：接触丰富 Shadowing、感知跑酷、野外徒步）
- [人形腿部行星滚柱丝杠直线驱动（PRS）](wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md)（Optimus 类腿部直线执行器 + 连杆：负载/自锁/力控 vs 动态带宽的权衡）
- [人形机器人并联关节解算](wiki/concepts/humanoid-parallel-joint-kinematics.md)（闭链踝、力分配与仿真接口分层）
- [文字生成 CAD（Text-to-CAD）](wiki/concepts/text-to-cad.md)（可用早期：概念件 / 脚本参数化 / STEP 衔接；整机装配仍以专业 CAD 为主）
- [CAD Skills（earthtojake/text-to-cad）](wiki/entities/cad-skills.md)（Agent Skills：build123d STEP-first + URDF/SRDF/SDF + 制造/打印全链路）
- [Articraft](wiki/entities/articraft.md)（agent + SDK + harness：可关节 3D 仿真资产与 Articraft-10K；arXiv:2605.15187）
- [World Labs](wiki/entities/world-labs.md)（空间智能：Marble 生成式 3D 世界 + Spark 流式 3DGS Web 渲染）
- [Superpowers（obra）](wiki/entities/superpowers-obra.md)（编码代理可组合技能 + TDD / worktree / 子代理交付；与 LLM Wiki 规约对照）
- [Skills For Real Engineers（mattpocock）](wiki/entities/mattpocock-skills.md)（轻量可组合工程技能：grill、CONTEXT.md、TDD、架构卫生；skills.sh 安装）
- [SenseNova-Skills（OpenSenseNova）](wiki/entities/sensenova-skills.md)（Agent Skills 办公技能库：信息图/PPT/Excel/深度研究；Hermes/OpenClaw + SenseNova API）
- [Hermes Agent（Nous Research）](wiki/entities/hermes-agent.md)（常驻自主代理：网关 + 记忆/技能闭环 + 多沙箱 + cron/子代理；MIT）

### roadmap/ — 成长路线
回答"应该先学什么、再学什么、学完输出什么"。

核心路线：
- [主路线：运动控制成长路线](roadmap/motion-control.md)（含 RL / IL / 安全 / 接触操作等 [可选纵深](roadmap/motion-control.md#depth-optional-index)）

### tech-map/ — 技术栈地图
回答"模块之间是什么关系"。

- [技术栈模块总览](tech-map/overview.md)
- [模块依赖关系图](tech-map/dependency-graph.md)

### references/ — 论文导航
按主题整理的论文列表，补充 wiki 的深度阅读需求。

---

## 当前推荐阅读顺序

如果你从人形机器人运动控制切入，建议顺序：

1. [LIP / ZMP](wiki/concepts/lip-zmp.md)
2. [Centroidal Dynamics](wiki/concepts/centroidal-dynamics.md)
3. [Trajectory Optimization](wiki/methods/trajectory-optimization.md)
4. [Model Predictive Control (MPC)](wiki/methods/model-predictive-control.md)
5. [TSID](wiki/concepts/tsid.md)
6. [Whole-Body Control](wiki/concepts/whole-body-control.md)
7. [State Estimation](wiki/concepts/state-estimation.md)
8. [System Identification](wiki/concepts/system-identification.md)
9. [Sim2Real](wiki/concepts/sim2real.md)
10. [Locomotion](wiki/tasks/locomotion.md)

---

## 当前主线知识链

```
LIP / ZMP
  ↓
Centroidal Dynamics
  ↓
Trajectory Optimization / MPC
  ↓
TSID / WBC
  ↓
State Estimation / System Identification / Sim2Real
```

---

## Obsidian Dataview 动态查询

> 以下查询块在 Obsidian + Dataview 插件中可直接使用。在 GitHub/普通 Markdown 渲染器中显示为代码块。

**查看所有 concept 页面及状态：**
```dataview
TABLE status, tags FROM "wiki/concepts"
SORT file.name ASC
```

**查看所有 draft 状态的页面（待完善）：**
```dataview
TABLE type, file.folder FROM "wiki"
WHERE status = "draft"
SORT type ASC
```

**查看带 rl 标签的所有页面：**
```dataview
LIST FROM "wiki"
WHERE contains(tags, "rl")
SORT type ASC
```

---

## 与其他项目的边界

- [`Humanoid_Robot_Learning_Paper_Notebooks`](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks)：单篇论文深读
- [`Robotics_Notebooks`](https://github.com/ImChong/Robotics_Notebooks)：跨模块知识组织
- [`ImChong.github.io`](https://github.com/ImChong/ImChong.github.io)：个人简历

> 论文项目负责点，技术栈项目负责线和面，个人主页负责展示。

---

---

## 完整页面目录

自动生成的全量页面清单已迁移到 **[catalog.md](catalog.md)**。本页只保留核心入口、推荐顺序与主线知识链，便于第一次访问时快速定位。

维护者运行 `make catalog` 即可重新生成完整目录。
