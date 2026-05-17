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
| 想看知识概念和方法 | 直接翻下面的 wiki 目录 |
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
- [Imitation Learning](wiki/methods/imitation-learning.md)
- [World Action Models（WAM）](wiki/concepts/world-action-models.md)（联合未来–动作分布的具身策略范式；综述与 Awesome-WAM 资源入口）
- [Pelican-Unified 1.0（UEI）](wiki/methods/pelican-unified-1.md)（Qwen3-VL 推理末态 \(z\) + Wan 系 UFG：同一扩散去噪联合未来视频与动作；arXiv:2605.15153）
- [mimic-video（VAM）](wiki/methods/mimic-video.md)（互联网视频潜计划 + 流匹配动作解码；arXiv:2512.15692）
- [EgoScale](wiki/methods/egoscale.md)（2 万小时级 egocentric 人视频预训练 VLA + 对齐 mid-training；arXiv:2602.16710）
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
- [legged_gym](wiki/entities/legged-gym.md)
- [Pinocchio](wiki/entities/pinocchio.md)
- [AprilTag](wiki/entities/april-tag.md)
- [AnyGrasp（抓取感知 SDK）](wiki/entities/anygrasp.md)
- [Crocoddyl](wiki/entities/crocoddyl.md)
- [cuRobo](wiki/entities/curobo.md)
- [Unitree](wiki/entities/unitree.md)
- [市面知名机器人平台纵览](wiki/overview/notable-commercial-robot-platforms.md)（人形 / 四足高频品牌索引）
- [四足机器人（Quadruped Robot）](wiki/entities/quadruped-robot.md)
- [轮足四足机器人（四轮足）](wiki/concepts/wheel-legged-quadruped.md)（Go2W 类混合滚动–步态）
- [MotionCode™](wiki/entities/motioncode.md)（人体运动数据与 Mind 线人形训练叙事）
- [HumanNet](wiki/entities/humannet.md)（百万小时级人中心视频语料；VLA/IL 人类侧预训练参照）
- [Xiaomi-Robotics-0](wiki/entities/xiaomi-robotics-0.md)（小米开源 VLA；Qwen3-VL + DiT flow matching，异步 action chunk 部署）
- [Project Instinct](wiki/entities/project-instinct.md)（人形全身动态控制研究站群：接触丰富 Shadowing、感知跑酷、野外徒步）
- [人形机器人并联关节解算](wiki/concepts/humanoid-parallel-joint-kinematics.md)（闭链踝、力分配与仿真接口分层）
- [文字生成 CAD（Text-to-CAD）](wiki/concepts/text-to-cad.md)（可用早期：概念件 / 脚本参数化 / STEP 衔接；整机装配仍以专业 CAD 为主）
- [Articraft](wiki/entities/articraft.md)（agent + SDK + harness：可关节 3D 仿真资产与 Articraft-10K；arXiv:2605.15187）
- [World Labs](wiki/entities/world-labs.md)（空间智能：Marble 生成式 3D 世界 + Spark 流式 3DGS Web 渲染）
- [Superpowers（obra）](wiki/entities/superpowers-obra.md)（编码代理可组合技能 + TDD / worktree / 子代理交付；与 LLM Wiki 规约对照）

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







### Entities（实体页）

- [1X Technologies](entities/1x-technologies.md) — 1X Technologies（前身为 Halodi Robotics）是一家挪威与美国双总部的通用人形机器人公司，产品覆盖轮式人形 EVE 与面向家庭场景的双足 NEO，强调真实世界部署数据与 AI 能力迭代。 `📅2026-05-07` `[entity_page]`
- [Allegro Hand (灵巧手)](entities/allegro-hand.md) — Allegro Hand** 是由 Wonik Robotics 开发的一款高性能四指灵巧手（Dexterous Hand）。它在机器人科研界（特别是强化学习和模仿学习领域）享有极高的普及率，被视 `📅unknown` `[entity_page]`
- [AnyGrasp（抓取感知 SDK）](entities/anygrasp.md) — AnyGrasp** 是面向平行夹爪的稠密 7-DoF 抓取感知与跨帧跟踪管线：单前向从深度点云预测大量候选并做时空关联，官方以 SDK（预编译库 + License）形式发布。 `📅2026-05-13` `[entity_page]`
- [ANYmal 四足机器人](entities/anymal.md) — ANYmal** 是由苏黎世联邦理工学院（ETH Zurich）的机器人系统实验室（Robotic Systems Lab, RSL）研发，并随后由衍生公司 ANYbotics 成功商业化的高性能 `📅unknown` `[entity_page]`
- [AMASS（统一 SMPL 人体动捕档案）](entities/amass.md) — AMASS 将多份光学标记动捕拟合到 SMPL 参数序列，是动画、可视化与机器人学习里常见的大规模人体参考运动来源之一；下载与引用以 MPI-IS 站点许可为准。 `📅2026-05-15` `[entity_page]`
- [Asimov v1（开源人形机器人仓库）](entities/asimov-v1.md) — Asimov v1 由 asimovinc 在单仓内开放机械与电气 CAD、MuJoCo 模型及板载软件，配套 DIY Kit 与自采 BOM，适合作为全栈对齐与 Sim2Real 研究的硬件参考平台 `📅unknown` `[entity_page]`
- [Atom01 Deploy](entities/atom01-deploy.md) — atom01_deploy** 负责 Atom01 的真机部署链路，连接训练策略与机器人执行系统，是 Sim2Real 落地关键环节。 `📅unknown` `[entity_page]`
- [Atom01 Description](entities/atom01-description.md) — atom01_description** 是 Atom01 的机器人描述仓库，主要提供 URDF、网格和模型配置，用于连接硬件实体与仿真/控制软件。 `📅unknown` `[entity_page]`
- [Atom01 Firmware](entities/atom01-firmware.md) — atom01_firmware** 是 Atom01 的底层固件与板端运行仓库，负责通信、设备驱动与基础运行时能力。 `📅unknown` `[entity_page]`
- [Atom01 Hardware](entities/atom01-hardware.md) — Atom01_hardware** 是 Roboparty Atom01 机器人的硬件主仓库，负责承载机械结构、电子设计与物料清单等“实体可复现”资产。 `📅unknown` `[entity_page]`
- [Atom01 Train](entities/atom01-train.md) — atom01_train** 是 Roboparty Atom01 项目的训练主仓库，聚焦 IsaacLab 场景下的策略学习、实验配置与迁移链路。 `📅unknown` `[entity_page]`
- [Booster Robotics RoboCup Demo](entities/booster-robocup-demo.md) — Booster Robotics RoboCup Demo** 是由 [Booster Robotics](https://github.com/BoosterRobotics) 官方维护的开源项 `📅unknown` `[entity_page]`
- [Boston Dynamics（波士顿动力）](entities/boston-dynamics.md) — Boston Dynamics** 是一家全球顶尖的机器人工程公司，以其在足式机器人运动控制、平衡和动力学领域的卓越成就而闻名。从 1992 年从 MIT 的 Leg Laboratory 独立至 `📅unknown` `[entity_page]`
- [BotLab / MotionCanvas（浏览器内策略–仿真编排）](entities/botlab-motioncanvas.md) — BotLab（站点内产品名 MotionCanvas）是地瓜机器人提供的浏览器端节点图工具：在网页里编排观测、历史堆叠、ONNX 策略与 MuJoCo 步进，并支持 MSCP 图与 Netron 模型预览。 `📅2026-05-12` `[entity_page]`
- [Crocoddyl](entities/crocoddyl.md) — Crocoddyl** 是一个面向机器人最优控制与轨迹优化的开源工具箱，长期由 **LAAS-CNRS / INRIA / Gepetto / Stack-of-Tasks** 这条学术与开源路线 `📅unknown` `[entity_page]`
- [cuRobo（NVIDIA GPU 运动生成库）](entities/curobo.md) — cuRobo** 在 CUDA 上并行化无碰撞 IK、几何规划与多样本轨迹优化，cuRoboV2 引入 B 样条+力矩约束与深度融合 ESDF，面向双臂与人形等高自由度系统。 `📅2026-05-16` `[entity_page]`
- [Drake (机器人工具箱)](entities/drake.md) — Drake** 是由丰田研究院（Toyota Research Institute, TRI）主导开发，由 Russ Tedrake（MIT 教授）团队深度参与的核心开源机器人软件库。它并非单纯的 `📅unknown` `[entity_page]`
- [DoorMan（论文：人形 RGB 开门视觉 Sim2Real）](entities/paper-doorman-opening-sim2real-door.md) — DoorMan（arXiv:2512.01061，CVPR 2026）以人形纯 RGB 开门为 loco-manipulation 基准：Isaac Lab 中特权教师 PPO 配合分阶段重置探索，经 DAgger 蒸馏到视觉学生，再用 GRPO 微调缓解部分可观测性；大规模物理与 PBR/光照随机化支撑真机零样本泛化。 `📅2026-05-17` `[entity_page]`
- [E-SDS（论文：环境感知 VLM 自动奖励与人形地形 RL）](entities/paper-e-sds-environment-aware-humanoid-locomotion-rl.md) — E-SDS（arXiv:2512.16446，UCL）在 SDS 式视频分解上加入仿真采样的地形统计以生成调用高度图与 LiDAR 的奖励，双候选 PPO 与 VLM rollout 反馈三轮精炼；Isaac Lab + Unitree G1 四类地形上报告相对手工感知基线显著降速跟误差，论文称唯其完成楼梯下降。 `📅2026-05-17` `[entity_page]`
- [EWMBench（具身世界模型生成评测）](entities/ewmbench.md) — EWMBench 面向具身世界模型（EWM）视频生成：在 Agibot-World 子集上从场景守恒、末端轨迹与语义/逻辑对齐三轴评测候选模型，并开源数据与评测工具链。 `📅2026-05-16` `[entity_page]`
- [FastStair（论文：人形高速上楼梯）](entities/paper-faststair-humanoid-stair-ascent.md) — FastStair（arXiv:2601.10365）用 GPU 并行 DCM 落脚点离散搜索嵌入 Isaac Lab RL，强 foothold-tracking 预训练后分训高速/低速专家并以 LoRA 合成单策略，在 LimX Oli 上报告约 1.65 m/s 指令速度与长螺旋梯实爬。 `📅2026-05-17` `[entity_page]`
- [URDD（论文：Universal Robot Description Directory）](entities/paper-urdd-universal-robot-description-directory.md) — URDD（arXiv:2512.23135）把从 URDF 反复派生的 DOF 映射、链结构、凸分解与形状近似等整理为版本化 JSON/YAML 模块目录，配套 Rust/Bevy 与 Three.js 检视器及示例资产库，以减少跨栈重复预处理。 `📅2026-05-17` `[entity_page]`
- [Figure AI](entities/figure-ai.md) — Figure AI 是美国人形机器人公司，以 Figure 02 整机与自研 Helix 视觉-语言-动作（VLA）模型为核心，强调全身协同与端侧推理，代表「垂直整合具身 AI」的一条主流工程路线。 `📅2026-05-07` `[entity_page]`
- [GENE-26.5（Genesis AI 操作基础模型）](entities/gene-26-5-genesis-ai.md) — GENE-26.5 是 Genesis AI 公司的机器人操作基础模型产品品牌；与开源 Genesis 物理仿真器（Genesis-Embodied-AI）名称相近但主体不同。 `📅2026-05-07` `[entity_page]`
- [Genesis (仿真器)](entities/genesis-sim.md) — **Genesis** 是具身智能领域新兴的高性能物理仿真与数据生成平台，常与 Isaac Gym / Isaac Lab 并列，强调多物理场与 GPU 大规模并行。 `📅2026-05-07` `[entity_page]`
- [GR00T-WholeBodyControl（人形全身控制平台）](entities/gr00t-wholebodycontrol.md) — NVlabs 单仓聚合解耦 WBC、GEAR-SONIC（SONIC）与 MotionBricks 预览的训练/部署/VLA 教程与 C++ 推理栈。 `📅2026-05-14` `[entity_page]`
- [人形机器人（Humanoid Robot）](entities/humanoid-robot.md) — 人形机器人是具有双足步行能力和类人形态（躯干 + 双臂 + 双腿）的机器人平台，兼顾移动能力与操作能力，是当前具身智能研究的核心载体。 `📅unknown` `[entity_page]`
- [HumanNet](entities/humannet.md) — HumanNet 是面向具身学习的大规模人中心视频语料（论文宣称约一百万小时），一三人称混合并配套交互导向标注与可扩展策展管线；适合作为 VLA/表示学习的人类侧数据基础设施参照。 `📅2026-05-14` `[entity_page]`
- [InterPrior（论文：物理 HOI 生成式控制）](entities/paper-interprior.md) — InterPrior（arXiv:2602.06035，CVPR 2026 Highlight）用 InterMimic+ 全参考 PPO 专家、掩码多模态目标变分蒸馏与带正则的 RL 微调，学习可泛化的物理人–物全身交互先验；报告 G1 sim-to-sim 与交互控制演示。 `📅2026-05-17` `[entity_page]`
- [Isaac Gym / Isaac Lab](entities/isaac-gym-isaac-lab.md) — Isaac Gym** 和 **Isaac Lab** 是 NVIDIA 机器人强化学习生态里的两代核心框架。 `📅unknown` `[entity_page]`
- [Kimodo（运动扩散模型）](entities/kimodo.md) — NVIDIA nv-tlabs 开源：大规模动捕上训练的运动扩散模型，支持文本与运动学约束生成 SOMA / G1 / SMPL-X 轨迹，附 CLI、时间线 Demo 与公开基准。 `📅2026-05-14` `[entity_page]`
- [LaFAN1（Ubisoft 动捕数据集）](entities/lafan1-dataset.md) — Ubisoft La Forge 发布的 BVH 棚拍动捕与过渡任务基准（SIGGRAPH 2020 配套）；许可为 CC BY-NC-ND 4.0，克隆需 Git LFS。 `📅2026-05-15` `[entity_page]`
- [LIFT（人形 SAC + 物理知情世界模型管线）](entities/lift-humanoid.md) — BIGAI 开源：MuJoCo Playground 大规模并行 JAX SAC 预训练，Brax 拉格朗日 + 残差世界模型离线拟合，微调阶段真机确定性采集、随机探索仅发生在模型 rollout（arXiv:2601.21363）。 `📅2026-05-17` `[entity_page]`
- [legged_gym](entities/legged-gym.md) — legged_gym** 是 ETH Zurich RSL（Robotic Systems Lab）开源的足式机器人强化学习训练框架，建立在 **Isaac Gym** 之上。 `📅unknown` `[entity_page]`
- [MuJoCo (物理引擎)](entities/mujoco.md) — MuJoCo (Multi-Joint dynamics with Contact)** 是一款专为机器人、生物力学和控制研究开发的高性能物理引擎。自被 DeepMind 收购并完全开源（Apac `📅unknown` `[entity_page]`
- [NVIDIA Omniverse (具身仿真底座)](entities/nvidia-omniverse.md) — NVIDIA Omniverse** 并非一个简单的物理引擎，而是一个庞大的**实时协作仿真平台**。在机器人领域，它是目前最强物理仿真器 **Isaac Sim** 的运行底座。通过利用光线追踪 `📅unknown` `[entity_page]`
- [Meta Quest (Oculus) 遥操作](entities/oculust-quest-teleop.md) — 在机器人模仿学习（Imitation Learning）和 VLA 模型训练中，**Meta Quest (原 Oculus Quest)** 系列 VR 头显已成为获取大规模高质量人类演示数据的核心 `📅unknown` `[entity_page]`
- [Mixamo（Adobe 角色与动画库）](entities/mixamo.md) — Adobe 在线绑定角色与全身动画下载服务；适合美术与原型管线，商业许可以官方为准，与科研向 MoCap 档案在可追溯性与再分发边界上不同。 `📅2026-05-15` `[entity_page]`
- [MotionCode™](entities/motioncode.md) — MotionCode 是宣称专注人体运动数字化的公司，公开分为 Move（采集与标注）、Media（内容）、Mind（机器人与 AI 运动学习）三线，并强调数据可对接 Isaac、MuJoCo 及主流 DCC/实时引擎管线。 `📅2026-05-07` `[entity_page]`
- [开源人形机器人“大脑” (主控电脑) 选型](entities/open-source-humanoid-brains.md) — 对于人形机器人，其“大脑”需要承担两类截然不同的计算任务：一是需要极高确定性的底层 **运控循环 (1kHz+)**；二是需要海量算力的 **感知与大模型推理 (5-30Hz)**。 `📅unknown` `[entity_page]`
- [开源人形机器人硬件方案对比](entities/open-source-humanoid-hardware.md) — 随着具身智能的爆发，人形机器人的硬件门槛正在迅速降低。对于预算有限的实验室或个人研究者，**开源硬件方案 (Open-source Humanoid Hardware)** 是验证算法的首选。 `📅unknown` `[entity_page]`
- [Pinocchio (刚体动力学库)](entities/pinocchio.md) — Pinocchio** 是一个由法国国家信息与自动化研究所（INRIA）开源的，专注于**高计算效率**和**分析导数 (Analytical Derivatives)** 的刚体动力学（Rigi `📅unknown` `[entity_page]`
- [PyTorch](entities/pytorch.md) — PyTorch 是以 Python 优先、动态图友好的开源深度学习框架，覆盖从研究训练到服务端与边端部署的完整工具链，是机器人学习与仿真的主流张量后端之一。 `📅2026-05-15` `[entity_page]`
- [Project Instinct](entities/project-instinct.md) — 本页汇总 Project Instinct 公开站点与子课题主张；定量结论与实现细节以各论文 PDF 与代码仓库为准。 `📅2026-05-12` `[entity_page]`
- [四足机器人（Quadruped Robot）](entities/quadruped-robot.md) — 四足机器人是以四条腿与环境形成间歇接触的腿足平台，侧重崎岖地形移动与户外部署，常与强化学习 locomotion、Sim2Real 及分层导航结合。 `📅unknown` `[entity_page]`
- [RLDX-1](entities/rldx-1.md) — RLDX-1** 是 RLWRLD 开源的灵巧操作 VLA：Qwen3-VL 骨干 + MSAT 扩散动作头，可选运动模块、时序记忆与触觉/力矩物理流，三阶段训练与图捕获/RTC 推理栈。 `📅2026-05-11` `[entity_page]`
- [Robotic World Model（ETH RSL，RWM / RWM-U）](entities/robotic-world-model-eth-rsl.md) — ETH RSL 开源的神经动力学世界模型：Isaac Lab 扩展支持在线/离线想象 rollout 训练模型基策略；Lite 仓剥离仿真仅保留离线管线与 Colab 入口（ANYmal D 参考任务）。 `📅2026-05-17` `[entity_page]`
- [RIO（Robot I/O）](entities/robot-io-rio.md) — RIO（Robot I/O）** 是一套面向**真实机器人**的 **Python 实时 I/O** 与编排框架，目标是把「换一套机械臂 / 人形 / 相机 / 遥操作设备就要重写控制栈」的摩擦降 `📅2026-05-15` `[entity_page]`
- [robot_lab (IsaacLab 扩展框架)](entities/robot-lab.md) — robot_lab** 是由开发者 `fan-ziqi` 维护的一个建立在 NVIDIA **IsaacLab** 之上的强化学习 (RL) 扩展库。它允许用户在隔离的仓库中开发机器人资产、环境和 `📅unknown` `[entity_page]`
- [机器人关键帧与运动编辑工具（选型入口）](entities/robot-motion-keyframe-editors.md) — 本页把三条 **公开仓库** 上的运动编辑工具放在一起对照：它们都解决「已有轨迹 / 姿态序列 → 人工修正 → 再导出」的问题，但 **绑定仿真栈、文件格式与是否纯前端** 差异很大，选型时应先确定下游是 **真机 CSV**、**MuJoCo qpos** 还是 **NumPy 归档**。 `📅2026-05-17` `[entity_page]`
- [Roboto Origin（开源人形机器人基线）](entities/roboto-origin.md) — Roboto Origin** 是 Roboparty 发布的“全链路开源”人形机器人项目入口页，目标不是只给一个仓库，而是提供从硬件到训练再到部署的可复现工程路径。 `📅unknown` `[entity_page]`
- [SceneVerse++（3D 场景理解数据集）](entities/sceneverse-pp.md) — SceneVerse++ 从互联网无标注视频自动重建与标注大规模真实室内 3D 场景，为检测分割、3D 空间 VQA 与视觉–语言导航等任务提供训练数据，代码与数据开源。 `📅2026-05-07` `[entity_page]`
- [Shadow Hand (灵巧手)](entities/shadow-hand.md) — Shadow Hand** 由英国 Shadow Robot Company 开发，是目前世界上最接近人类手部功能的灵巧手平台之一。它拥有 5 根手指和 20 个主动驱动关节（总计 24 个自由度 `📅unknown` `[entity_page]`
- [Tairan He（何泰然）](entities/tairan-he.md) — 何泰然（Tairan He）为 CMU RI 博士、NVIDIA GEAR 实习背景，研究聚焦人形规模化学习与视觉 Sim2Real；代表作含 OmniH2O、HOVER、ASAP、VIRAL / DoorMan 等，主页为论文与项目总索引。 `📅2026-05-14` `[entity_page]`
- [Unitree G1 (人形机器人)](entities/unitree-g1.md) — Unitree G1** 是宇树科技 (Unitree) 在 H1 之后推出的一款量产型、高性价比的人形机器人平台。其设计初衷是降低人形机器人研究的门槛，使其能够大规模进入实验室、高校和家庭场景。 `📅unknown` `[entity_page]`
- [unitree_ros（Unitree 官方 ROS1 / Gazebo 栈）](entities/unitree-ros.md) — unitree_ros 是 Unitree 官方 ROS1（Melodic/Kinetic）+ Gazebo8 的机器人描述与关节级仿真栈；真机 ROS 控制依赖 unitree_ros_to_real 与 unitree_legged_msgs。README 明确 Gazebo 侧不做高层行走，与 MuJoCo 上的 unitree_rl_mjlab 形成并行官方路线。 `📅2026-05-17` `[entity_page]`
- [Unitree](entities/unitree.md) — Unitree Robotics（宇树科技）** 是当前腿式机器人和人形机器人领域最有影响力的公司之一。 `📅unknown` `[entity_page]`
- [World Labs（空间智能与世界生成）](entities/world-labs.md) — World Labs 是公开以「空间智能」与三维世界模型为主线的公司与产品团队：浏览器产品 Marble 从文本/图像/视频/全景生成可编辑的持久 3D 世界，并开源 Spark（Web 端 3DGS 渲染与流式大场景技术栈）支撑交互式展示与管线集成。 `📅2026-05-16` `[entity_page]`
- [Xiaomi-Robotics-0](entities/xiaomi-robotics-0.md) — 小米开源的约 4.7B 参数 VLA（Qwen3-VL-4B + DiT flow matching），强调跨本体预训练、异步 action chunk 执行与仿真/双臂真机评测。 `📅2026-05-15` `[entity_page]`
- [Xue Bin Peng（彭学斌）](entities/xue-bin-peng.md) — Xue Bin Peng** 是 SFU 助理教授兼 NVIDIA 研究科学家，博士师从 Levine / Abbeel；以 DeepMimic、AMP、ASE、动力学随机化 Sim2Real 等工作定义了物理角色与腿式机器人 RL 运动控制的一条主干研究线，并通过 MimicKit 统一开源实现。 `📅2026-05-14` `[entity_page]`
- [Zhengyi Luo（罗正宜）](entities/zhengyi-luo.md) — Zhengyi Luo** 为 NVIDIA GEAR Lab 高级研究科学家、CMU RI 博士（Kris Kitani）；工作横跨人形通用低层控制、人–人形遥操作、视觉 Sim2Real 与交互感知，是 HOVER、ASAP、OmniH2O、PDC 与 SONIC 等社区关键论文的核心作者之一。 `📅2026-05-14` `[entity_page]`
- [舞肌科技（上海舞肌科技有限公司）](entities/wuji-robotics.md) — 舞肌科技（上海舞肌）面向具身机器人提供关节级电机（F 系列 / Pan Motor 叙事）与官方文档确认的五指灵巧手 Wuji Hand，配套 SDK、ROS2、MuJoCo 描述与遥操作 Retargeting；研发在上海、量产在常州。 `📅2026-05-12` `[entity_page]`

### Wiki Concepts（概念页）

- [3D 空间 VQA（3D Spatial Visual Question Answering）](concepts/3d-spatial-vqa.md) — **3D 空间 VQA**：在 **三维室内场景** 条件下，模型需要结合视觉观测与自然语言问题，推理物体间 **几何关系**（远近、相对方位、计数、尺度、路径顺序等）并给出答案——常见形式包括选择题与数值题。 `📅2026-05-07` `[wiki_page]`
- [Armature Modeling（电枢惯量建模）](concepts/armature-modeling.md) — 在机器人动力学和仿真中，**Armature** 指的是电机内部旋转部件（转子）的转动惯量，经过减速比放大后，对关节端产生的等效惯性效应。 `📅unknown` `[wiki_page]`
- [Capture Point / DCM](concepts/capture-point-dcm.md) — Capture Point（捕获点）** 和 **DCM（Divergent Component of Motion，发散运动分量）** 是腿式机器人动态平衡与步态控制里两个非常关键的概念，用来描 `📅unknown` `[wiki_page]`
- [Centroidal Dynamics](concepts/centroidal-dynamics.md) — Centroidal Dynamics（质心动力学）**：用机器人整体质心的线动量和角动量来描述全身动力学的一种中层建模方式。 `📅unknown` `[wiki_page]`
- [Contact Dynamics](concepts/contact-dynamics.md) — Contact Dynamics（接触动力学）**：研究机器人与地面、物体、墙面等环境发生接触时，接触力、约束和系统运动之间关系的动力学框架。 `📅unknown` `[wiki_page]`
- [Contact Estimation（接触估计）](concepts/contact-estimation.md) — Contact Estimation 是指在机器人运动过程中，**实时判断哪个足/末端执行器处于接触状态（与地面或物体接触）**，并尽可能估计接触力的大小和方向。 `📅unknown` `[wiki_page]`
- [Contact-Rich Manipulation（接触丰富型操作）](concepts/contact-rich-manipulation.md) — Contact-Rich Manipulation**：那些必须利用接触力、摩擦、约束和接触序列本身才能完成的操作任务，例如插拔、拧瓶盖、推门、卡扣装配、双手推箱等。 `📅unknown` `[wiki_page]`
- [Control Barrier Function（控制屏障函数）](concepts/control-barrier-function.md) — 控制屏障函数（Control Barrier Function，CBF）**：一种将系统安全约束转化为可在控制层实时强制执行的数学工具。通过定义一个标量函数 $h(x)$，使得 $h(x) \ge `📅unknown` `[wiki_page]`
- [Curriculum Learning（课程学习）](concepts/curriculum-learning.md) — Curriculum Learning 是一种训练策略：在学习早期提供更简单的任务或环境，随着策略能力提升逐渐增加难度，模拟人类"从简单到复杂"的学习过程。 `📅unknown` `[wiki_page]`
- [Dexterous Kinematics (灵巧手运动学)](concepts/dexterous-kinematics.md) — 灵巧手运动学 (Dexterous Kinematics)** 是机器人学中研究多指协同操作的理论基础。与传统的单臂串联运动学不同，灵巧手在抓取物体时，多个手指通过接触点与物体共同构成了一个**闭 `📅unknown` `[wiki_page]`
- [Domain Randomization](concepts/domain-randomization.md) — 域随机化**：在仿真训练中主动随机化物理参数、视觉纹理、环境设置，让策略被迫学会适应各种变化的泛化能力，从而实现零样本从仿真迁移到现实。 `📅unknown` `[wiki_page]`
- [Embodied Data Cleaning (具身数据清洗)](concepts/embodied-data-cleaning.md) — 具身数据清洗**：在具身智能（Embodied AI）中，将人类示教或自动采集的原始“脏数据”转化为高质量、可用于训练的专家演示轨迹（Expert Trajectories）的过程。 `📅unknown` `[wiki_page]`
- [Embodied Scaling Laws (具身规模法则)](concepts/embodied-scaling-laws.md) — 具身规模法则**：在机器人学习中，随着训练数据（演示轨迹、仿真经验）、模型参数量和计算资源的增加，模型在未见任务、未见物体和未见环境上的表现呈现出可预测的性能提升趋势（通常遵循幂律分布）。 `📅unknown` `[wiki_page]`
- [EtherCAT 协议基础](concepts/ethercat-protocol.md) — EtherCAT (Ethernet for Control Automation Technology)** 是目前人形机器人底层总线的首选协议。它解决了标准以太网因冲突检测（CSMA/CD）而 `📅unknown` `[wiki_page]`
- [Floating Base Dynamics](concepts/floating-base-dynamics.md) — Floating Base Dynamics（浮动基动力学）**：描述机器人在基座不固定于世界坐标系时，其整体动力学如何建模与控制的框架。 `📅unknown` `[wiki_page]`
- [Footstep Planning（步位规划）](concepts/footstep-planning.md) — Footstep Planning** 是腿式机器人运动规划中的核心子问题：在给定运动目标和地形约束下，**决定每一步脚应该落在哪里、何时落下**。步位规划的输出是一个时序接触点序列（contac `📅unknown` `[wiki_page]`
- [Force Control Basics (力控制基础)](concepts/force-control-basics.md) — 在人形机器人和操作任务中，**力控制 (Force Control)** 是实现物理交互的基石。与传统工业机器人仅跟踪位置轨迹（Position Control）不同，力控制允许机器人感知并调节它对环 `📅unknown` `[wiki_page]`
- [Foundation Policy（基础策略模型）](concepts/foundation-policy.md) — Foundation Policy（基础策略模型）**：在大规模多任务、多机器人形态演示数据上预训练的通用机器人策略，通过"规模化预训练 + 任务微调"范式，将跨任务泛化能力迁移到新场景——是 N `📅unknown` `[wiki_page]`
- [Gait Generation（步态生成）](concepts/gait-generation.md) — Gait Generation** 是腿式机器人运动控制中负责**决定步态模式（gait pattern）的模块**：确定各腿的支撑/摆动相时序、步频、步幅范围，为步位规划和质心轨迹优化提供时序框 `📅unknown` `[wiki_page]`
- [HQP（Hierarchical QP）](concepts/hqp.md) — 分层二次规划（Hierarchical Quadratic Programming，HQP）**：全身控制（WBC）中处理多任务优先级冲突的优化框架，通过将任务按优先级分层求解，确保高优先级任务精 `📅unknown` `[wiki_page]`
- [Hybrid Force-Position Control（力位混合控制）](concepts/hybrid-force-position-control.md) — 力位混合控制**：把任务空间拆成“该控位置的方向”和“该控力的方向”，让机器人在一个子空间内严格跟踪几何目标，在另一个子空间内稳定施加期望接触力。 `📅unknown` `[wiki_page]`
- [人形与腿式策略的网络架构（Policy Network Architecture）](concepts/humanoid-policy-network-architecture.md) — 人形与腿式模仿/强化策略里，网络“长什么样”往往被写在论文 Method：从浅层 MLP 到 AMP 判别器、MoE、Transformer/Diffusion chunk，再到 VLA 与 WAM；真机强项常是小 MLP，难点在观测、奖励与 sim2real。 `📅2026-05-15` `[wiki_page]`
- [Impedance Control（阻抗控制）](concepts/impedance-control.md) — 阻抗控制**：不直接要求机器人“精确走到某个位姿”，而是规定当机器人与环境之间出现位置误差或接触力时，系统应该表现出怎样的 **质量-弹簧-阻尼（Mass-Spring-Damper）** 响应。 `📅unknown` `[wiki_page]`
- [Latent Imagination (潜空间想象)](concepts/latent-imagination.md) — 潜空间想象 (Latent Imagination)** 是现代 Model-Based 强化学习（尤其是 **Dreamer** 系列）的灵魂。它彻底改变了机器人学习的范式：不再是在真实世界或沉 `📅unknown` `[wiki_page]`
- [LCM (Lightweight Communications and Marshalling) 基础](concepts/lcm-basics.md) — LCM** 是一款由 MIT 团队开发的通信库，专门针对**高频、低延迟、高带宽**的机器人控制场景设计。在人形机器人和四足机器人的“脊髓级”控制中，LCM 是优于 ROS 2 的首选方案。 `📅unknown` `[wiki_page]`
- [LIP / ZMP](concepts/lip-zmp.md) — LIP（Linear Inverted Pendulum, 线性倒立摆）** 和 **ZMP（Zero Moment Point, 零力矩点）** 是双足机器人行走控制里最经典的一对基础模型与稳定 `📅unknown` `[wiki_page]`
- [Motion Retargeting Pipeline（动作重定向流水线）](concepts/motion-retargeting-pipeline.md) — Motion Retargeting Pipeline** 关注的不是「某一个重定向算法」，而是把**异构来源的人体动作**（MoCap、单目视频估计、生成模型、遥操作流）落到**机器人可执行参考 `📅unknown` `[wiki_page]`
- [Motion Retargeting（动作重定向）](concepts/motion-retargeting.md) — Motion Retargeting 是将一个运动序列（通常来自人类或动物）**转换为适合目标机器人执行的动作序列**的过程。 `📅unknown` `[wiki_page]`
- [MPC 与 WBC 集成：人形机器人 locomotion 的典型控制架构](concepts/mpc-wbc-integration.md) — MPC 负责"大尺度规划"（质心往哪走、落脚点放哪），WBC 负责"全身执行"（怎么协调关节力矩来跟踪 MPC 发出的指令）**——两者分层配合，组成当前人形机器人 locomotion 最主流的 `📅unknown` `[wiki_page]`
- [Optimal Control (OCP)](concepts/optimal-control.md) — 最优控制**：给定一个动力学系统和一个代价函数，求解在有限或无限时域内使得代价最小的控制输入序列的理论框架。 `📅unknown` `[wiki_page]`
- [Privileged Training（特权信息训练）](concepts/privileged-training.md) — 特权训练**（Privileged Training / Teacher-Student Training）：训练阶段提供给策略额外的、在真实部署时无法获取的信息，再通过知识蒸馏将能力迁移给仅使用 `📅unknown` `[wiki_page]`
- [Reward Design](concepts/reward-design.md) — 奖励函数设计（Reward Design）**：强化学习中定义智能体优化目标的核心环节。奖励函数的好坏直接决定策略能不能学出来、学出来后的行为是否符合预期。 `📅unknown` `[wiki_page]`
- [ROS 2 (Robot Operating System 2) 基础](concepts/ros2-basics.md) — ROS 2** 是全球机器人社区中最广泛使用的开源框架。它并非真正的操作系统，而是一套运行在 Linux 之上的**中间件 (Middleware)**，提供了标准化的通信协议、开发工具和海量的算 `📅unknown` `[wiki_page]`
- [Safety Filter（安全过滤器）](concepts/safety-filter.md) — Safety Filter**：位于高层策略和低层控制器之间的一层在线修正模块。它接收一个“名义动作”或“候选控制输入”，在尽量少改动原动作的前提下，强制满足安全约束，例如关节限位、碰撞距离、接触 `📅unknown` `[wiki_page]`
- [传感器融合（Sensor Fusion）](concepts/sensor-fusion.md) — 传感器融合**：将来自多个传感器（IMU、摄像头、激光雷达、腿部运动学）的测量值在概率框架下统一融合，估计机器人的位姿、速度和接触状态，为上层控制（MPC / WBC）提供实时、精确的状态输入。 `📅unknown` `[wiki_page]`
- [Sim2Real](concepts/sim2real.md) — Sim2Real**（仿真到现实迁移）：在仿真环境训练控制策略，然后部署到真实机器人上。 `📅unknown` `[wiki_page]`
- [State Estimation](concepts/state-estimation.md) — State Estimation（状态估计）**：根据传感器观测、机器人模型和历史信息，估计机器人当前最可能真实状态的过程。 `📅unknown` `[wiki_page]`
- [System Identification](concepts/system-identification.md) — System Identification（系统辨识 / SysID）**：通过实验数据估计机器人动力学、执行器、摩擦、延迟等模型参数，使模型更接近真实系统的过程。 `📅unknown` `[wiki_page]`
- [Tactile Sensing（触觉感知）](concepts/tactile-sensing.md) — 触觉感知 (Tactile Sensing)** 是机器人感知系统中的重要组成部分。如果说视觉（Vision）赋予了机器人远距离和全局的场景理解能力，那么触觉则是机器人与物理世界发生**直接物理交 `📅unknown` `[wiki_page]`
- [Terrain Adaptation（地形适应）](concepts/terrain-adaptation.md) — Terrain Adaptation**：让腿式或人形机器人根据地形感知结果，动态调整落脚点、身体姿态、接触时序和控制参数，从而在楼梯、碎石、草地、台阶和坡面上稳定行走。 `📅unknown` `[wiki_page]`
- [文字生成 CAD（Text-to-CAD）](concepts/text-to-cad.md) — 文字生成 CAD 已从纯研究演示进入可用早期：适合概念件与参数化初稿，复杂装配与生产级 DFM 仍依赖专业 CAD；机器人方向更稳的是 LLM + CadQuery/OpenSCAD 参数化脚本再导出 STEP。 `📅2026-05-14` `[wiki_page]`
- [TSID](concepts/tsid.md) — TSID（Task Space Inverse Dynamics，任务空间逆动力学）** 是一种典型的人形机器人全身控制方法，用来在满足动力学与接触约束的前提下，把任务空间目标转成可执行的关节加速 `📅unknown` `[wiki_page]`
- [Video-as-Simulation (视频即仿真)](concepts/video-as-simulation.md) — 视频即仿真 (Video-as-Simulation)** 是具身智能领域最激进也最前沿的技术范式。它的核心假设是：如果一个生成模型能够完美预测“给定当前动作后，下一帧图像应该长什么样”，那么这个 `📅unknown` `[wiki_page]`
- [视触觉融合（Visuo-Tactile Fusion）](concepts/visuo-tactile-fusion.md) — 视触觉融合 (Visuo-Tactile Fusion)** 研究如何在一次操作的不同阶段，让机器人在「视觉全局信息」与「触觉局部信息」之间动态切换权重，特别关注**接触瞬间**这一最难的窗口期。 `📅unknown` `[wiki_page]`
- [Whole-Body Control (WBC)](concepts/whole-body-control.md) — 全身控制**：对人形机器人等复杂系统，同时协调多个肢体/关节完成全身任务的控制方法。 `📅unknown` `[wiki_page]`
- [Whole-Body Coordination（全身协调控制）](concepts/whole-body-coordination.md) — 全身协调控制（Whole-Body Coordination）**：研究高自由度机器人系统（尤其是人形机器人）如何将全身多个肢体、链接的运动在时间和空间上进行统一协调，使不同子系统的运动相互配合， `📅unknown` `[wiki_page]`
- [World Action Models（WAM，世界–动作模型）](concepts/world-action-models.md) — World Action Models（WAM）把环境前向预测与可执行动作生成耦合在同一具身策略里，以联合分布 p(o',a|o,l) 为对象，区别于纯反应式 VLA 与单独的世界模型。 `📅2026-05-14` `[wiki_page]`
- [轮足四足机器人（四轮足 / Wheel-Legged Quadruped）](concepts/wheel-legged-quadruped.md) — 轮足四足机器人在四条腿末端集成驱动轮，平地偏滚动效率与能效，崎岖地形仍依赖足式步态；典型量产如 Unitree Go2W / B2W，仿真资产参见 robot_lab。 `📅unknown` `[wiki_page]`

### Wiki Methods（方法页）

- [Action Chunking（动作块输出）](methods/action-chunking.md) — Action Chunking**：让策略一次预测未来连续若干步动作，而不是每个控制周期只吐一帧动作。它最早在模仿学习和双臂操作场景里被广泛采用，现在也成为 VLA 与低层控制器结合时处理推理延迟 `📅unknown` `[method_page]`
- [Actuator Network (执行器网络)](methods/actuator-network.md) — 执行器网络 (Actuator Network)** 是一种在机器人仿真中用于模拟物理驱动器（如电控伺服电机、SEA 驱动器）真实物理行为的深度学习模型。它是解决**足式机器人 Sim2Real  `📅unknown` `[method_page]`
- [AMP & HumanX: 判别器驱动的风格学习](methods/amp-reward.md) — 在机器人动作模仿中，单纯的轨迹跟踪奖励（如关节角度 MSE）往往会导致机器人出现高频抖动、抽搐或不自然的步态。**AMP** 引入了生成对抗的思想来提升运动质量，而 **HumanX** 将其扩展到了 `📅unknown` `[method_page]`
- [AMS: 物理可行性过滤与混合奖励](methods/ams.md) — 在人形机器人动作合成与学习中，很多参考动捕数据（MoCap）由于人体与机器人的动力学差异，直接模仿会导致物理不可行。**AMS** 提供了一套系统化的数据清洗与奖励设计方法。 `📅unknown` `[method_page]`
- [Any2Track & RGMT: 增强型自适应与时序建模](methods/any2track.md) — 在复杂的人形机器人控制任务（如多步动作模仿或动态越障）中，传统的单帧马尔可夫决策过程 (MDP) 往往不足以捕捉环境的非平稳性（如地面滑移、外力扰动）。**Any2Track** 与 **RGMT `📅unknown` `[method_page]`
- [Auto-labeling Pipelines (自动化标注流水线)](methods/auto-labeling-pipelines.md) — 自动化标注流水线** 是构建具身基础模型（Foundation Models）的关键基础设施。它解决了具身智能中最昂贵的环节：**让数据具备语义（Semantic Grounding）**。 `📅unknown` `[method_page]`
- [Behavior Cloning with Transformer](methods/bc-with-transformer.md) — 在模仿学习（IL）中，传统的基于 MLP 或 CNN 的行为克隆往往难以处理**多模态动作**（例如专家有时左绕，有时右绕）和**长时间依赖**。将 **Transformer** 引入 BC 架构， `📅unknown` `[method_page]`
- [Behavior Cloning（行为克隆）](methods/behavior-cloning.md) — Behavior Cloning, BC**：把专家演示数据当作监督学习数据集，直接学习从观测到动作的映射，是模仿学习最直接的做法。 `📅unknown` `[method_page]`
- [Being-H0.7（潜空间世界–动作模型）](methods/being-h07.md) — Being-H0.7 用大规模第一人称视频与机器人演示，在潜空间对齐未来感知监督，得到可部署的「世界–动作」先验；推理不滚未来像素，直接输出动作 chunk。 `📅2026-05-12` `[method_page]`
- [BeyondMimic](methods/beyondmimic.md) — BeyondMimic** 是由 Hybrid Robotics 等团队开发的高性能机器人动作模仿框架。相比早期的 DeepMimic 或 AMP，BeyondMimic 更侧重于从仿真到真实物理 `📅unknown` `[method_page]`
- [CLAW (宇树 G1 全身动作数据生成管线)](methods/claw.md) — CLAW** (Composable Language-Annotated Whole-Body Motion Data Generation) 是一种面向人形机器人的模块化数据生成方案。它通过将 `📅unknown` `[method_page]`
- [ContactNet](methods/contact-net.md) — ContactNet** 解决了“在杂乱无章的堆叠物中，手手该按在哪”的问题。它直接输入原始点云，输出稠密的接触成功概率图。 `📅unknown` `[method_page]`
- [CRISP（Contact-guided Real2Sim）](methods/crisp-real2sim.md) — CRISP（ICLR 2026）从单目视频用凸平面场景原语与人–场景接触补全构造可物理仿真的人形运动与场景，并以 RL 人形闭环提高跟踪成功率与仿真吞吐。 `📅2026-05-17` `[method_page]`
- [DAgger（Dataset Aggregation）](methods/dagger.md) — DAgger**：一种交互式模仿学习方法，让当前策略先去“自己跑”，再由专家为这些真实访问到的状态打标签，并把新数据持续并入训练集。 `📅unknown` `[method_page]`
- [Diffusion-based Motion Generation (基于扩散模型的运动生成)](methods/diffusion-motion-generation.md) — Diffusion-based Motion Generation**：利用扩散概率模型（Diffusion Probabilistic Models）生成机器人关节空间或笛卡尔空间的连续运动序列 `📅unknown` `[method_page]`
- [Diffusion Policy](methods/diffusion-policy.md) — Diffusion Policy**：将扩散生成模型（Diffusion Model）用于机器人模仿学习，通过逆扩散过程从噪声中生成动作序列的策略学习方法。 `📅unknown` `[method_page]`
- [Disney Olaf 角色机器人（实机动画角色）](methods/disney-olaf-character-robot.md) — Disney Research 将动画角色 Olaf 做成实机：非对称腿+泡沫裙隐藏机构、动画参考驱动的 PPO 策略，并把执行器温度与落地噪声显式写进观测与奖励（含 CBF 风格约束项）。 `📅2026-05-13` `[method_page]`
- [Dynamic Movement Primitives (DMP)](methods/dmp.md) — DMP** 是一种用于轨迹建模和控制的方法。它将复杂的运动路径表示为一个非线性动力学系统，其核心是一个受迫振荡器，可以通过调整参数来改变运动的速度和目标位置，而不需要重新规划。 `📅unknown` `[method_page]`
- [DWM（Dexterous World Models，灵巧世界模型）](methods/dwm.md) — DWM（Dexterous World Models）在已知静态 3D 场景上，用场景–动作条件视频扩散把第一人称相机轨迹与手部网格轨迹转成可信的交互视频，通过修复模型恒等先验学残差动力学，并以合成精确对齐数据混合固定机位真实视频补全物理真实性。 `📅2026-05-17` `[method_page]`
- [EFGCL（External Force-Guided Curriculum Learning）](methods/efgcl.md) — EFGCL** 是一种面向腿足机器人**高动态全身动作**的 **guided RL / 物理引导探索** 训练范式：在仿真里对机器人施加**外部辅助力**，使其在课程早期就能反复完成目标动作；再按**成功率自适应衰减**辅助强度直至为零，使策略在**无参考轨迹、弱奖励塑形**的条件下仍能学会并**迁移到实机**。 `📅2026-05-13` `[method_page]`
- [EGM（Efficient General Mimic，高效通用模仿跟踪）](methods/egm-efficient-general-mimic.md) — EGM（Efficient General Mimic）：以 Bin 级跨动作课程自适应采样、上下身分组的复合解耦 MoE（CDMoE）与三阶段教师–学生训练，在少量高质量重定向动捕上学习可泛化的人形高动态全身跟踪策略。 `📅2026-05-17` `[method_page]`
- [EgoScale（海量 egocentric 人视频 + 人–机对齐 mid-training 的 VLA）](methods/egoscale.md) — EgoScale：在超两万小时带腕与重定向灵巧手标签的第一人称人视频上预训练流式 VLA，实证人数据规模与验证损失近 log-linear 缩放且与真机灵巧表现强相关，再以小规模视点对齐的人–机 mid-training 锚定机器人感知与控制（arXiv:2602.16710）。 `📅2026-05-17` `[method_page]`
- [ExoActor (视频生成驱动的交互式人形控制)](methods/exoactor.md) — ExoActor** 把"第三人称（exocentric）视频生成"作为人形机器人 **交互动力学的统一接口**：给定任务指令与场景观测，先让大型视频生成模型"想象"出一段任务执行视频， `📅2026-05-03` `[method_page]`
- [Generalized Advantage Estimation (GAE)](methods/gae.md) — GAE** 解决了强化学习中一个核心痛点：如何准确估计一个动作比平均水平“好多少”（即优势函数 $A(s, a)$），同时保持低方差。 `📅unknown` `[method_page]`
- [Generative Data Augmentation (生成式数据增强)](methods/generative-data-augmentation.md) — 在具身智能训练中，**生成式数据增强** 是解决“长尾效应 (Long-tail Distribution)”的关键。虽然我们可以轻易采集到成千上万条成功的“拿杯子”演示，但“杯子滑落”、“手部剧烈抖 `📅unknown` `[method_page]`
- [Generative World Models (生成式世界模型)](methods/generative-world-models.md) — 生成式世界模型** 是具身智能（Embodied AI）领域的下一代物理引擎替代者。不同于 Drake 或 MuJoCo 等基于严谨几何和力学方程的解析引擎，生成式世界模型直接利用**生成式 AI `📅unknown` `[method_page]`
- [GENMO（统一人体运动估计与生成）](methods/genmo.md) — GENMO（官方代码常以 GEM 名义发布）把人体运动估计表述为带多模态条件的约束式扩散生成，统一视频、2D 关键点、文本、音频与 SMPL 关键帧等输入下的轨迹恢复与合成。 `📅2026-05-07` `[method_page]`
- [HAIC: 基于世界模型的教师-学生训练](methods/haic.md) — 在复杂的物体交互任务（如搬运、协作、精细操作）中，机器人不仅要模仿姿态，还要实时预测物体状态和外力。**HAIC** 提出了一种创新的训练范式，通过世界模型（World Model）将特权信息（Pri `📅unknown` `[method_page]`
- [Hindsight Experience Replay (HER)](methods/her.md) — HER** 是一种处理“稀疏奖励（Sparse Reward）”任务的绝佳技巧。在抓取或装配任务中，如果机器人只有在完美完成任务时才得到 1 分奖励，它很难通过随机探索学到任何东西。 `📅unknown` `[method_page]`
- [HY-Motion 1.0（文本→SMPL-H 流匹配 DiT）](methods/hy-motion-1.md) — HY-Motion 1.0：腾讯混元将 DiT+流匹配的人体文本→3D 运动生成扩至十亿级参数，配套大规模预训练、高质量微调与 DPO/Flow-GRPO 对齐，并开源代码与权重（arXiv:2512.23464）。 `📅2026-05-17` `[method_page]`
- [HiPAN（Hierarchical Posture-Adaptive Navigation）](methods/hipan.md) — HiPAN** 是面向**四足机器人**在**非结构化三维环境**（窄通道、限高、死胡同、半封闭房间）中的导航框架：部署时**不依赖显式三维地图**，仅用**机载深度**做感知，通过**分层强化学习**把“往哪走、身体缩多低”与“关节怎么动”拆开，并用 **Path-Guided Curriculum Learning（PGCL）** 让高层策略从短视界反应式行为过渡到长视界目标导向导航。 `📅unknown` `[method_page]`
- [htwk-gym](methods/htwk-gym.md) — htwk-gym** 是一个开源的强化学习（RL）框架，专门针对人形机器人足球（Humanoid Soccer）竞赛设计。该框架由 RoboCup 强队 HTWK Leipzig 维护，在 **B `📅unknown` `[method_page]`
- [Imitation Learning (IL, 模仿学习)](methods/imitation-learning.md) — 模仿学习 (Imitation Learning)**：通过专家演示数据（行为克隆等），让机器人学会从状态到动作的映射，核心是“抄”。 `📅unknown` `[method_page]`
- [In-hand Reorientation (手内重定向)](methods/in-hand-reorientation.md) — 手内重定向 (In-hand Reorientation)** 是灵巧操作（Dexterous Manipulation）领域中最具挑战性的任务之一。它的目标是让多指灵巧手（如 Allegro H `📅unknown` `[method_page]`
- [Intentional Updates for Streaming RL（意图更新与流式强化学习）](methods/intentional-updates-streaming-rl.md) — 意图更新（intentional updates）指：不显式固定「参数空间步长」，而是先规定**当前这一步在关心的输出量**（价值预测、动作 log-probability 等）上希望达到的变化，再用 `📅2026-05-10` `[method_page]`
- [LingBot-Map (Streaming 3D Reconstruction Foundation Model)](methods/lingbot-map.md) — LingBot-Map** 是一种新型的 3D 基础模型，旨在解决从连续视频流中进行高效、鲁棒的**流式 3D 重建**问题。 `📅unknown` `[method_page]`
- [LQR / iLQR 算法详解](methods/lqr-ilqr.md) — LQR (Linear Quadratic Regulator)** 是线性最优控制的解析基石，而 **iLQR (iterative LQR)** 是其在非线性系统上的威力延伸。它们通过贝尔曼最 `📅unknown` `[method_page]`
- [LWD（Learning while Deploying）](methods/lwd.md) — LWD（Learning while Deploying）** 是 AGIBOT Research 在 2026 年提出的**车队级（fleet-scale）offline-to-online 强化学习后训练框架**，把部署中的成功/失败/人为干预统一喂给同一个 RL 学习器，让单一通用 VLA 策略持续改进。 `📅2026-05-06` `[method_page]`
- [Multi-Agent Reinforcement Learning (MARL)](methods/marl.md) — MARL** 扩展了单智能体 RL，处理多个机器人在同一空间协作或竞争的问题（如机器人足球、多臂流水线）。 `📅unknown` `[method_page]`
- [mimic-video（Video-Action Model, VAM）](methods/mimic-video.md) — mimic-video 是一类把互联网规模视频生成模型当作操作语义与物理动力学先验的通用操作策略：先在视频潜空间里形成与语言指令一致的视觉动力学计划，再以流匹配动作头输出机器人动作块。 `📅2026-05-17` `[method_page]`
- [Model-Based RL（基于模型的强化学习）](methods/model-based-rl.md) — Model-Based RL（MBRL）**：在强化学习中，智能体显式学习或利用环境的动力学模型，通过在模型中规划或生成虚拟经验来提升样本效率。 `📅unknown` `[method_page]`
- [Model Predictive Control (MPC)](methods/model-predictive-control.md) — 模型预测控制**：一种基于滚动时域优化的控制方法，在每个时刻求解一个有限时域的最优控制问题，只执行第一步，然后重复。 `📅unknown` `[method_page]`
- [GMR: 通用动作重定向](methods/motion-retargeting-gmr.md) — GMR (General Motion Retargeting)** 是运动控制流程中的“前端”模块，负责将人类或其他来源的动作序列转换为机器人可理解的关节角度序列。 `📅unknown` `[method_page]`
- [NMR（神经运动重定向与人形全身控制）](methods/neural-motion-retargeting-nmr.md) — NMR** 用 CEPR 管线（筛选、GMR、聚类 RL 专家、仿真 rollout）构造物理一致的人机配对监督，训练 CNN–Transformer 将 SMPL 序列映射为 Unitree G1 等平台的全身参考轨迹，缓解优化式重定向的非凸与噪声传播。 `📅2026-05-08` `[method_page]`
- [ReActor（物理感知 RL 运动重定向）](methods/reactor-physics-aware-motion-retargeting.md) — ReActor** 将跨具身运动重定向建成物理仿真中的双层问题：上层优化稀疏语义刚体对应下的参数化参考，下层用 RL 跟踪；以结构化近似上层梯度避免 Hessian 逆，面向少脚滑/自碰的参考与下游模仿学习（SIGGRAPH 2026 预印本 arXiv:2605.06593）。 `📅2026-05-13` `[method_page]`
- [SPIDER（物理感知采样式灵巧重定向）](methods/spider-physics-informed-dexterous-retargeting.md) — SPIDER** 用并行物理仿真中的采样式轨迹优化，把人体+物体的运动学参考 refinement 成动力学可行、接触意图保留的机器人轨迹；课程式虚拟接触力降低接触歧义（arXiv:2511.09484）。 `📅2026-05-17` `[method_page]`
- [Model Predictive Path Integral (MPPI)](methods/mppi.md) — MPPI** 是一种基于采样（Sampling-based）的随机最优控制算法。它不依赖于对动力学方程进行求导（与 DDP/iLQR 不同），而是通过在 GPU 上并行生成成千上万条随机轨迹，并根 `📅unknown` `[method_page]`
- [PAiD Framework](methods/paid-framework.md) — PAiD (Perception-Action integrated Decision-making)** 是由 TeleHuman 研究团队提出的一种针对人形机器人足球技能的渐进式学习框架。其核 `📅unknown` `[method_page]`
- [Pelican-Unified 1.0（统一具身智能 UEI）](methods/pelican-unified-1.md) — Pelican-Unified 1.0 将 Qwen3-VL 的语义理解与链式推理末态 \(z\)，与 Wan 系扩散 UFG 耦合，使未来视频与动作块在同一去噪轨迹中联合生成。 `📅2026-05-16` `[method_page]`
- [Policy Optimization](methods/policy-optimization.md) — 策略优化**：通过直接对策略参数做梯度上升或近似优化，使期望累积奖励最大化的一类强化学习方法。 `📅unknown` `[method_page]`
- [Reinforcement Learning (RL, 强化学习)](methods/reinforcement-learning.md) — 强化学习 (Reinforcement Learning)**：通过与环境交互，以最大化累积奖励 (Reward) 为目标学习决策策略的机器学习范式。 `📅unknown` `[method_page]`
- [Safe RL（安全强化学习）](methods/safe-rl.md) — 安全强化学习（Safe Reinforcement Learning, Safe RL）** 是近年来强化学习领域发展最快、在机器人实体部署中最为核心的一个分支。其根本宗旨在于：在智能体（Agen `📅unknown` `[method_page]`
- [SONIC（规模化运动跟踪人形控制）](methods/sonic-motion-tracking.md) — SONIC 将规模化运动跟踪作为人形低层控制的统一预训练目标；论文主张网络容量、MoCap 数据与算力三轴 scaling，并以统一 token 接口接入 VR、视频、VLA 等上游。 `📅2026-05-14` `[method_page]`
- [StarVLA](methods/star-vla.md) — StarVLA**（尤其是其首个技术报告版本 **StarVLA-$\alpha$**）是一个旨在降低 Vision-Language-Action (VLA) 系统复杂性的开源基准模型与框架。 `📅unknown` `[method_page]`
- [Trajectory Optimization（轨迹优化）](methods/trajectory-optimization.md) — 轨迹优化 (Trajectory Optimization, TO)** 是一种基于动力学模型和约束条件，通过数值非线性规划（NLP）技术来自动搜索最优运动序列的计算方法。在足式机器人领域，不论是 `📅unknown` `[method_page]`
- [Unified Multimodal Tokens (统一多模态 Token)](methods/unified-multimodal-tokens.md) — 统一多模态 Token** 是一种先进的具身智能架构设计。它摒弃了为每种感官模态设计专用神经网络分支的传统做法，转而将所有输入（图像、语言、状态、动作）全部转换为格式一致的 Token 序列，并在 `📅unknown` `[method_page]`
- [Visual Servoing（视觉伺服控制）](methods/visual-servoing.md) — 视觉伺服 (Visual Servoing)** 是一门将计算机视觉（Computer Vision）与经典控制理论（Control Theory）深度融合的技术。它不依赖于将图像构建为复杂的 3 `📅unknown` `[method_page]`
- [WiLoR（野外 3D 手部定位与重建）](methods/wilor.md) — WiLoR** 是端到端的野外手部检测定位 + Transformer 3D 手部网格重建网络，支持单图像与逐帧视频管线，为下游灵巧操作或人形双手轨迹提供单 RGB 估计。 `📅2026-05-07` `[method_page]`
- [VLA（Vision-Language-Action）](methods/vla.md) — VLA**：把视觉、语言和机器人动作统一到同一个模型里，让策略不只“看见状态后输出动作”，还能够显式理解任务指令和语义约束。 `📅unknown` `[method_page]`
- [π₀ (Pi-zero) 策略模型](methods/π0-policy.md) — π₀ (Pi-zero)** 是具身智能大模型（VLA）领域的最新突破，由 Physical Intelligence 团队于 2024 年提出。它旨在打破“一个机器人一个模型”的限制，通过单一的 `📅unknown` `[method_page]`

### Wiki Tasks（任务页）

- [Balance Recovery（平衡恢复）](tasks/balance-recovery.md) — 平衡恢复**（Balance Recovery / Push Recovery）：机器人在受到外部扰动（推力、地形突变、碰撞等）后，从失衡状态恢复到稳定姿态并继续任务的能力。 `📅unknown` `[task_page]`
- [Bimanual Manipulation（双臂协调操作）](tasks/bimanual-manipulation.md) — 双臂协调操作（Bimanual Manipulation）**：同时使用两只手臂协同完成一个任务，两臂之间存在物理或时序上的依赖关系——典型任务包括双手递接、拧瓶盖、折叠衣服、组装零件。单臂独立控 `📅unknown` `[task_page]`
- [Humanoid Locomotion (人形机器人移动)](tasks/humanoid-locomotion.md) — Humanoid Locomotion**：使双足类人机器人能够在复杂、非结构化的地形中，保持平衡的同时实现高效、鲁棒的位移，并具备全身协调（Whole-body Coordination）能力。 `📅unknown` `[task_page]`
- [Humanoid Soccer](tasks/humanoid-soccer.md) — 人形机器人足球**：人形机器人在动态、竞争性环境下的综合表现。作为 RoboCup 的核心项目，它被认为是衡量人形机器人自主能力的重要基准。 `📅unknown` `[task_page]`
- [Loco-Manipulation (移动操作)](tasks/loco-manipulation.md) — 移动操作（Loco-Manipulation）**：机器人在运动（行走/移动）的同时执行操作任务（抓取/推动/交互），要求同时具备行走能力和上肢操作能力。 `📅unknown` `[task_page]`
- [Locomotion](tasks/locomotion.md) — 运动/行走**：让机器人（尤其人形/足式）实现稳定、高效、多地形移动的能力。 `📅unknown` `[task_page]`
- [Manipulation](tasks/manipulation.md) — 操作**：让机器人的手/末端执行器抓取、移动、操作物体。 `📅unknown` `[task_page]`
- [Teleoperation（遥操作）](tasks/teleoperation.md) — 一句话定义**：操作员通过外部设备实时远程控制机器人完成任务，同时采集高质量示范数据用于后续策略学习。 `📅unknown` `[task_page]`
- [ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation](tasks/ultra-survey.md) — 统一多模态控制：实现人形机器人自主全身移动操作 `📅unknown` `[task_page]`
- [视觉–语言导航（Vision-and-Language Navigation, VLN）](tasks/vision-language-navigation.md) — **VLN**：智能体接收 **自然语言导航指令** 与 **第一人称（egocentric）视觉观测**（渲染视图或真实相机图像），在离散或连续动作空间中决策，最终到达指令描述的目标位置或物体。**语言–视觉接地** 与 **路径效率** 是核心评价维度。 `📅2026-05-07` `[task_page]`

### Wiki Formalizations（形式化基础）

- [Behavior Cloning Loss (行为克隆损失函数)](formalizations/behavior-cloning-loss.md) — 行为克隆 (Behavior Cloning, BC)** 是模仿学习（Imitation Learning）中最简单且最广泛使用的形式。它的核心思想是：给定一个由专家（人类操作员或最优控制器）生 `📅unknown` `[formalization_page]`
- [Bellman 方程](formalizations/bellman-equation.md) — Bellman 方程**：值函数的递归关系，揭示了"未来奖励"与"当前决策"之间的数学联系，是几乎所有强化学习算法的理论基础。 `📅unknown` `[formalization_page]`
- [Constrained MDP (CMDP)](formalizations/cmdp.md) — 约束马尔可夫决策过程 (Constrained Markov Decision Process, CMDP)** 是一种在运筹学和强化学习中极其重要的数学形式化框架。当我们在构建真实物理世界的机器 `📅unknown` `[formalization_page]`
- [Contact Complementarity（接触互补约束）](formalizations/contact-complementarity.md) — 接触互补约束（Contact Complementarity Conditions）** 是描述刚体与环境接触物理行为的数学框架：**接触力与接触间隙必须互补为零**——要么接触面有力（接触中）， `📅unknown` `[formalization_page]`
- [接触力旋量锥 (Contact Wrench Cone)](formalizations/contact-wrench-cone.md) — 接触力旋量锥 (Contact Wrench Cone, CWC)** 把 [摩擦锥 (Friction Cone)](./friction-cone.md) 从“单点三维力”推广到“面接触的六维 `📅unknown` `[formalization_page]`
- [Control Lyapunov Function（控制李雅普诺夫函数）](formalizations/control-lyapunov-function.md) — Control Lyapunov Function（CLF）**：一种用于设计使系统渐近稳定的控制律的数学工具。通过找到一个正定标量函数 $V(x)$，使得在某个控制输入下 $\dot{V}(x `📅unknown` `[formalization_page]`
- [Cross-modal Attention (跨模态注意力)](formalizations/cross-modal-attention.md) — 在具身大模型（VLA）中，**跨模态注意力 (Cross-modal Attention)** 是实现“理解指令并根据视觉反馈执行动作”的核心数学机制。它允许模型在处理 Token 序列时，显式地计算 `📅unknown` `[formalization_page]`
- [Extended Kalman Filter (EKF)](formalizations/ekf.md) — 扩展卡尔曼滤波（EKF）**：将标准卡尔曼滤波推广到非线性系统的经典状态估计方法，通过每步线性化（一阶 Taylor 展开）在非线性系统上近似应用 Kalman 递推公式。 `📅unknown` `[formalization_page]`
- [Foundation Policy Alignment (基础策略对齐)](formalizations/foundation-policy-alignment.md) — 在具身基础模型（Foundation Policy）中，**对齐 (Alignment)** 是指将来自不同机器人形态（如四足、双足、机械臂）、不同传感器配置和不同任务目标的异构数据，映射到一个统一的 `📅unknown` `[formalization_page]`
- [摩擦锥 (Friction Cone)](formalizations/friction-cone.md) — 摩擦锥** 是机器人学中描述接触力物理约束的核心数学模型。它规定了接触力 $\mathbf{f}$ 必须满足的范围，以确保机器人脚部或手部与支撑环境之间不发生滑动。 `📅unknown` `[formalization_page]`
- [GAE（广义优势估计）](formalizations/gae.md) — GAE（Generalized Advantage Estimation）** 是估计策略梯度中优势函数 $A(s,a)$ 的标准方法，通过参数 $\lambda \in [0,1]$ 在**偏差 `📅unknown` `[formalization_page]`
- [HJB 方程（Hamilton-Jacobi-Bellman）](formalizations/hjb.md) — HJB 方程**是连续时间最优控制的基本方程，给出了最优值函数 $V^*(x,t)$ 满足的偏微分方程（PDE）。它是 Bellman 最优方程在连续时间域的推广。 `📅unknown` `[formalization_page]`
- [LQR / iLQR](formalizations/lqr.md) — LQR（Linear Quadratic Regulator，线性二次调节器）**：最优控制中最经典的解析解，针对线性系统 + 二次代价函数，给出最优状态反馈增益的闭式解。**iLQR（itera `📅unknown` `[formalization_page]`
- [Lyapunov 稳定性](formalizations/lyapunov.md) — Lyapunov 稳定性**：通过构造一个随系统状态变化的标量函数 $V(x)$ 来判断平衡点附近的误差是否收敛。对机器人控制来说，它回答的是："这个控制器不仅能把误差压小，而且能持续保持稳定吗？ `📅unknown` `[formalization_page]`
- [Markov Decision Process (MDP)](formalizations/mdp.md) — 马尔可夫决策过程**：在离散时间步中，智能体根据当前状态选择动作，环境根据转移概率回应新状态和奖励的数学框架，是强化学习的理论基础。 `📅unknown` `[formalization_page]`
- [Partially Observable MDP (POMDP)](formalizations/pomdp.md) — 在真实的机器人应用中，我们永远无法获取完美的、全知全能的状态 $s$。传感器噪声、视觉遮挡和未知的物理参数使得系统处于**部分可观测 (Partial Observability)** 状态。 `📅unknown` `[formalization_page]`
- [Probability Flow (概率流形式化)](formalizations/probability-flow.md) — 在具身智能的生成式动作建模（如 **π₀** 或 **Diffusion Policy**）中，**概率流 (Probability Flow)** 是连接噪声分布与真实动作分布的数学“传送带”。它将 `📅unknown` `[formalization_page]`
- [SE(3) Representation (位姿表示形式化)](formalizations/se3-representation.md) — 在机器人学与具身智能中，如何表示物体的**位姿（Pose）**——即位置与姿态的组合，是感知与控制的基础。**SE(3)** (Special Euclidean Group) 描述了三维空间中的刚体 `📅unknown` `[formalization_page]`
- [Task Space Inverse Dynamics (TSID) 形式化](formalizations/tsid-formulation.md) — TSID** 是一种在保持机器人物理一致性的前提下，实现多任务并行控制的数学框架。它将复杂的运动指令转换为底层的电机力矩。 `📅unknown` `[formalization_page]`
- [Variational Objective (变分目标函数)](formalizations/variational-objective.md) — 在构建具身智能的世界模型（World Models）时，我们面临的核心数学挑战是如何从高维、嘈杂的观测（图像）中提取紧凑的、具有预测性的隐变量表示。**变分目标函数 (Variational Obje `📅unknown` `[formalization_page]`
- [Action Tokenization (动作分词)](formalizations/vla-tokenization.md) — 在具身智能大模型（VLA）中，**动作分词 (Action Tokenization)** 是连接符号推理（语言模型）与物理执行（机器人控制）的数学枢纽。它解决了 LLM 架构本质上是离散序列预测器， `📅unknown` `[formalization_page]`
- [ZMP + LIP 形式化](formalizations/zmp-lip.md) — LIP (Linear Inverted Pendulum)** 模型是将人形机器人简化为在固定高度平面内运动的质点，而 **ZMP (Zero Moment Point)** 则是判断该系统是否 `📅unknown` `[formalization_page]`

### Wiki Comparisons（对比页）

- [CLF vs CBF：稳定性与安全性的对偶工具](comparisons/clf-vs-cbf.md) —  工具 | 全称 | 核心角色  `📅unknown` `[comparison_page]`
- [数据手套 vs 视觉遥操作 (灵巧数据采集选型)](comparisons/data-gloves-vs-vision-teleop.md) — 在训练灵巧手（Dexterous Hand）执行复杂任务时，获取高质量的人类演示数据是第一步。目前，**穿戴式数据手套 (Data Gloves)** 和 **基于视觉的遥操作 (Vision-bas `📅unknown` `[comparison_page]`
- [Kalman Filter vs. Optimization-based Estimation (状态估计选型)](comparisons/kalman-filter-vs-optimization-based-estimation.md) — 在机器人（特别是人形和四足机器人）中，实时估计 Base 的位置、速度和姿态是所有算法的基础。目前主要存在两大技术路线：以 **EKF** 为代表的递归滤波派，和以 **滑窗优化 (Sliding W `📅unknown` `[comparison_page]`
- [Model-Based vs Model-Free RL 对比](comparisons/model-based-vs-model-free.md) —  维度 | Model-Free RL | Model-Based RL  `📅unknown` `[comparison_page]`
- [MPC vs RL：控制策略选型对比](comparisons/mpc-vs-rl.md) — 背景**：MPC（模型预测控制）和 RL（强化学习）是当前机器人运动控制领域的两大主流范式。MPC 基于显式动力学模型在线求解最优控制，RL 离线学习隐式策略。两者在假设、计算模式和适用场景上都有 `📅unknown` `[comparison_page]`
- [MuJoCo vs Isaac Lab：仿真器选型对比](comparisons/mujoco-vs-isaac-lab.md) — 背景**：MuJoCo 和 Isaac Lab 是当前 locomotion RL 领域最常用的两款仿真平台。MuJoCo 代表学术研究生态的经典底座，以物理精度和 API 友好性著称；Isaac `📅unknown` `[comparison_page]`
- [MuJoCo vs Isaac Sim (物理引擎选型)](comparisons/mujoco-vs-isaac-sim.md) — 在机器人强化学习和仿真部署领域，**MuJoCo**（由 DeepMind 维护）和 **Isaac Sim / Isaac Gym**（由 NVIDIA 维护）是目前最主流的两大物理引擎阵营。它们的 `📅unknown` `[comparison_page]`
- [Online RL vs Offline RL](comparisons/online-vs-offline-rl.md) — Online RL 和 Offline RL 是两种根本不同的学习范式。两者都在优化同一个目标（累积奖励），但对**数据来源**的要求截然不同，导致适用场景和瓶颈完全不同。 `📅unknown` `[comparison_page]`
- [PPO vs SAC (vs BRRL/BPO)：机器人 RL 算法选型](comparisons/ppo-vs-sac.md) — 背景**：PPO（Proximal Policy Optimization）和 SAC（Soft Actor-Critic）是机器人 RL 领域最主流的两种连续控制算法。两者都已在真实机器人上取得 `📅unknown` `[comparison_page]`
- [RL vs 模仿学习（Imitation Learning）](comparisons/rl-vs-il.md) — RL 和 IL 是机器人策略学习的两条主干路线。两者都在学"策略 $\pi(a|s)$"，但监督信号、数据需求、能达到的行为质量完全不同。 `📅unknown` `[comparison_page]`
- [ROS 2 vs LCM (机器人中间件选型)](comparisons/ros2-vs-lcm.md) — 在机器人真机部署中，如何让分布在不同进程（甚至不同计算板）上的节点进行可靠、低延迟的数据通信？**ROS 2 (Robot Operating System 2)** 和 **LCM (Lightwe `📅unknown` `[comparison_page]`
- [Sim2Real 方法横向对比](comparisons/sim2real-approaches.md) — Sim2Real gap 的应对策略有三大类：**Domain Randomization（仿真端随机化）**、**Domain Adaptation（领域自适应）**、**Real-World Fi `📅unknown` `[comparison_page]`
- [Trajectory Optimization vs Reinforcement Learning](comparisons/trajectory-opt-vs-rl.md) — 在足式机器人运动控制领域，**轨迹优化 (Trajectory Optimization, TO)** 和 **强化学习 (Reinforcement Learning, RL)** 是两种截然不同但 `📅unknown` `[comparison_page]`
- [WBC vs RL: Whole-Body Control vs Reinforcement Learning](comparisons/wbc-vs-rl.md) — 人形机器人运动控制领域最常见的两种路线对比。 `📅unknown` `[comparison_page]`

### Wiki Overview（总览）

- [人形机器人运动控制 Know-How](overview/humanoid-motion-control-know-how.md) — - **高频振动**：足端撞击地面的瞬时振动会通过骨架传导至 IMU。如果固定不牢，加速度计会因共振产生巨大偏置。 `📅unknown` `[overview_page]`
- [市面知名机器人平台纵览](overview/notable-commercial-robot-platforms.md) — 本页回答：**除了少数明星项目外，产业与新闻里还经常出现哪些人形、四足与腿足平台**，它们大致属于哪条技术–商业路线，以及在本知识库里应去哪里深挖。 `📅2026-05-12` `[overview_page]`
- [Robot Learning Overview](overview/robot-learning-overview.md) — 机器人学习**：让机器人通过数据学会完成复杂任务的方法集合，核心是把”如何做”从人工编程转向从经验中学习。 `📅unknown` `[overview_page]`

### Roadmaps（路线页）

- [成长路线总览](README.md) — 本目录用于承载 `Robotics_Notebooks` 的成长路线设计。 `📅unknown` `[wiki_page]`
- [主路线：运动控制算法工程师成长路线](motion-control.md) — 1. **先把传统控制主干打通**：LIP/ZMP → Centroidal → MPC → TSID/WBC。 `📅unknown` `[wiki_page]`

### Tech-map Nodes（技术栈节点）

- [技术栈地图总览](README.md) — 本目录用于承载 `Robotics_Notebooks` 的技术栈地图、模块依赖关系、标准化模块卡片，以及研究方向导航。 `📅unknown` `[wiki_page]`
- [模块依赖关系图](dependency-graph.md) — 本页的目标不是做花哨图，而是先把 `Robotics_Notebooks` 当前最重要的依赖关系讲清楚。 `📅unknown` `[wiki_page]`
- [Humanoid Locomotion](modules/control/humanoid-locomotion.md) — 人形双足步行、平衡与扰动恢复是当前主攻方向之一。 `📅unknown` `[wiki_page]`
- [MPC](modules/control/mpc.md) — 模型预测控制是连接模型、约束与优化求解的重要方法。 `📅unknown` `[wiki_page]`
- [Whole-Body Control](modules/control/whole-body-control.md) — 全身控制是人形机器人运动控制的重要枢纽。 `📅unknown` `[wiki_page]`
- [Behavior Cloning](modules/il/behavior-cloning.md) — 模仿学习最基础的切入口。 `📅unknown` `[wiki_page]`
- [Diffusion Policy](modules/il/diffusion-policy.md) — 当前模仿学习中的重要生成式方法之一。 `📅unknown` `[wiki_page]`
- [人形策略网络架构](modules/il/policy-network-architecture.md) — 从浅层 MLP（DeepMimic / AMP）到 Diffusion chunk、VLA 与 WAM 的策略骨干演化；强调论文 Method 常见披露项与「小 MLP + 强系统」的真机现实。 `📅unknown` `[wiki_page]`
- [Motion Retarget](modules/il/motion-retarget.md) — 连接人体动作数据与人形机器人技能迁移的关键模块。 `📅unknown` `[wiki_page]`
- [线性代数](modules/math/linear-algebra.md) — 机器人场景重点：向量空间、矩阵变换、特征值分解、最小二乘、雅可比相关计算。 `📅unknown` `[wiki_page]`
- [模块模板](modules/module-template.md) — - `📅unknown` `[wiki_page]`
- [Humanoid RL](modules/rl/humanoid-rl.md) — 聚焦人形机器人 locomotion 与 skill learning 中的强化学习问题。 `📅unknown` `[wiki_page]`
- [PPO](modules/rl/ppo.md) — PPO 是机器人强化学习中最常见、最实用的基础算法之一。 `📅unknown` `[wiki_page]`
- [Sim2Real](modules/rl/sim2real.md) — 重点关注域随机化、系统辨识、鲁棒训练与部署闭环。 `📅unknown` `[wiki_page]`
- [动力学](modules/robotics/dynamics.md) — 重点包括牛顿欧拉法、拉格朗日法、浮动基动力学、接触建模。 `📅unknown` `[wiki_page]`
- [运动学](modules/robotics/kinematics.md) — 重点包括正逆运动学、雅可比、微分运动学。 `📅unknown` `[wiki_page]`
- [刚体运动](modules/robotics/rigid-body-motion.md) — 重点包括旋转表示、坐标变换、SE(3)、Twist/Wrench。 `📅unknown` `[wiki_page]`
- [Deployment](modules/system/deployment.md) — 部署阶段关注控制频率、硬件接口、安全性与调试闭环。 `📅unknown` `[wiki_page]`
- [ROS2](modules/system/ros2.md) — 机器人系统工程和集成阶段的重要基础设施。 `📅unknown` `[wiki_page]`
- [Simulation](modules/system/simulation.md) — 仿真是控制、学习与部署之间的中介层。 `📅unknown` `[wiki_page]`
- [全栈技术域总览](overview.md) — 本页不是单纯列方向，而是 `Robotics_Notebooks` 的技术栈模块入口页。 `📅unknown` `[wiki_page]`
- [研究方向导航](research-directions/README.md) — 后续将按“问题驱动”组织研究方向，而不是按学科简单罗列。 `📅unknown` `[wiki_page]`

### References（参考资料页）

- [参考导航 / References](references/README.md) — 这里不是原始资料堆，也不是知识页正文。 `📅unknown` `[reference_page]`
- [Benchmark 索引 / Benchmarks](references/benchmarks/README.md) — 这里用于整理 locomotion、humanoid、learning control 等方向最常见的 benchmark 与环境。 `📅unknown` `[reference_page]`
- [Humanoid Environments](references/benchmarks/humanoid-environments.md) — 用于整理人形机器人常用训练环境与评测场景。 `📅unknown` `[reference_page]`
- [Locomotion Benchmarks](references/benchmarks/locomotion-benchmarks.md) — 用于整理平地行走、越障、跑酷、复杂地形运动等 benchmark。 `📅unknown` `[reference_page]`
- [论文导航 / Papers](references/papers/README.md) — 这里不是论文全文仓库，也不是逐篇精读笔记区。 `📅unknown` `[reference_page]`
- [Humanoid Hardware](references/papers/humanoid-hardware.md) — 聚焦人形机器人硬件架构、执行器设计、传感器集成与系统工程。 `📅unknown` `[reference_page]`
- [Imitation Learning](references/papers/imitation-learning.md) — 聚焦行为克隆、DAgger、Diffusion Policy、动作块输出与技能嵌入相关论文。 `📅unknown` `[reference_page]`
- [Latent Skill Prior](references/papers/latent-skill-prior.md) — 聚焦如何将运动技能压缩到隐空间（Latent Space），并通过先验（Prior）引导强化学习生成既稳定又自然的行为。 `📅unknown` `[reference_page]`
- [Locomotion RL](references/papers/locomotion-rl.md) — 聚焦人形/腿足机器人 locomotion 中的强化学习论文。 `📅unknown` `[reference_page]`
- [MPC (Model Predictive Control)](references/papers/mpc.md) — 聚焦模型预测控制在机器人（特别是腿式/人形）中的理论、工程实现与应用论文。 `📅unknown` `[reference_page]`
- [Optimal Control](references/papers/optimal-control.md) — 最优控制理论基础、动态规划与轨迹优化奠基工作。 `📅unknown` `[reference_page]`
- [Sim2Real](references/papers/sim2real.md) — 聚焦域随机化、系统辨识、鲁棒训练、部署经验与真实机器人迁移相关论文。 `📅unknown` `[reference_page]`
- [Survey Papers](references/papers/survey-papers.md) — 用于汇总机器人学习、运动控制、人形机器人、模仿学习等方向的领域综述（Review/Survey）。 `📅unknown` `[reference_page]`
- [System Identification](references/papers/system-identification.md) — 聚焦机器人动力学辨识、执行器建模与在线参数估计论文。 `📅unknown` `[reference_page]`
- [Whole-Body Control](references/papers/whole-body-control.md) — 聚焦任务空间控制、TSID、QP-WBC、人形全身运动控制相关论文。 `📅unknown` `[reference_page]`
- [开源生态 / Repos](references/repos/README.md) — 这里不是代码仓库镜像，而是开源项目与工具链的导航层。 `📅unknown` `[reference_page]`
- [Humanoid Projects](references/repos/humanoid-projects.md) — 聚焦人形机器人运动控制、模仿学习、感知与部署相关开源项目。 `📅unknown` `[reference_page]`
- [Manipulation Perception](references/repos/manipulation-perception.md) — bin picking / 平行夹爪抓取位姿与 GraspNet 系数据、SDK 导航。 `📅2026-05-13` `[reference_page]`
- [Retarget Tools](references/repos/retarget-tools.md) — 聚焦人体动作到机器人动作的重定向工具与项目。 `📅unknown` `[reference_page]`
- [RL Frameworks](references/repos/rl-frameworks.md) — 人形/腿足机器人 RL 训练常用开源框架。 `📅unknown` `[reference_page]`
- [Simulation](references/repos/simulation.md) — 当前重点平台： `📅unknown` `[reference_page]`
- [Utilities](references/repos/utilities.md) — 收录 Pinocchio、RBDL、Drake、curobo 等通用工具链。 `📅unknown` `[reference_page]`
