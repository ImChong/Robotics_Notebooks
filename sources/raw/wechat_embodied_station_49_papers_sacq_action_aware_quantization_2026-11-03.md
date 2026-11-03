# 机器人不仅要完成任务，还要知道哪里出了错

> 原始抓取（2026-11-03，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/KBuNOCHIWGw905j4IpOhZw

机器人不仅要完成任务，还要知道哪里出了错
具身智能小站 具身智能小站
 Nov 3, 2026, 6:00 AM

阅读路线

今天更新 49 篇具身智能新论文，覆盖机器人策略与部署、操作评测与数据、人形与灵巧控制、导航与优化等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进46 篇扫读

先看 SACQ 的动作感知量化，再跟进四足课程统一评测与仿真驱动膝关节外骨骼；其余按阅读优先级扫读。这里的新论文指本次新增索引，其中两项为正文注明的历史发表工作。

先看这一篇

1低比特压缩要对齐最终动作

SACQ: Structure-Aware Codebook Quantization for Ultra-Low-Bit Vision-Language-Action Models

 

一句话判断：把视觉语言动作模型部署到有限内存的平台，不能只追求更低位宽：量化误差最终会变成动作偏差。SACQ 将下游计算结构与最终动作误差纳入压缩流程，提供了同时检查模型存储、位宽口径和闭环成功率的方法，适合做部署方案评估。

你可以带走的两点

• 量化目标对齐动作：结合结构感知码本、动作敏感度位宽分配与两阶段校准，避免只优化单层重建误差。

• 核对压缩口径：表 1 中 GR00T N1.7 在四个 LIBERO 套件平均成功率为 93.85%，BF16 为 94.10%；1.50 位指语言与动作候选层，模型内存为 1.175 GB，并非全部参数均为该位宽。

方法亮点：把不同投影层的下游作用引入码本拟合，并按最终动作敏感度决定压缩预算，使存储优化直接关联策略输出。

论文：https://arxiv.org/pdf/2610.41408

代码：https://github.com/haesunglee116/SACQ

项目页：https://haesunglee116.github.io/SACQ_page/

另外 2 篇值得跟进

2采样可达性估计有内在代价

Fundamental Limits of Sampling-Based Reachability: Geometry and Flow Distortion

 

看点：从几何误差和流映射形变分析轨迹终点采样的复杂度，揭示维数、时间跨度与初始集合条件的影响。理论结论依赖明确假设，不能当作所有验证器的下界。

结果：比较必须统一预算：相同 PPO 学生与环境步数下比较课程，分别看随机地形与保留结构化地形，避免用单一排名指导所有机器人。

论文：https://arxiv.org/pdf/2610.40473

代码：https://github.com/JeanLew01/reachapprox

3用仿真训练外骨骼辅助控制

Design and Validation of a Robotic Knee Exoskeleton with Musculoskeletal Simulation-Derived Biological Torque Control

 

看点：从肌骨仿真提取膝关节力矩并蒸馏成传感器估计器，减少新硬件专门采集训练数据的需求。人体实验针对健康参与者下蹲，不代表临床疗效。

结果：减少硬件数据依赖：由肌骨仿真生成训练信号，再蒸馏为仅使用机载传感器的力矩估计器；七名健康参与者的下蹲结果不能外推至康复人群。

论文：https://arxiv.org/pdf/2610.41981

代码：https://github.com/MetaMobilityLabCMU/Knee-Exoskeleton-Project

46 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04让机器人在失败现场继续执行

Auriga Technical Report: A Task-Agnostic Agent Harness for Zero-Shot Embodied Manipulation

Auriga 将三维测量、动作预览与执行检查组织成统一控制接口，使模型在偏差发生后原地恢复。当前报告的主要成绩来自仿真操作基准。

论文：https://arxiv.org/pdf/2611.01112

代码：https://github.com/lose4578/Auriga

项目页：https://auriga.stelledge.com/

05持续适配不必无限增加适配器

ResMerge: Residual Policy Learning and Merging for Continual Adaptation of Pre-trained Robot Policies

ResMerge 冻结基础策略，以强化学习训练残差，再依据残差场一致性合并专家。适合关注持续适配与参数规模的读者，但训练仍需要环境交互与奖励。

论文：https://arxiv.org/pdf/2610.41663

代码：https://github.com/zhucoffee/resmerge-official

项目页：https://resmerge.github.io/

06让接触动力学梯度服务控制学习

∇Mimic: Ultra-Efficient Humanoid Control with Differentiable Simulation

∇Mimic 用逐样本梯度裁剪和动作目标回归缓解接触梯度不稳定，配合专用可微引擎提升训练效率。应结合指定机器人和动作跟踪任务理解收益。

论文：https://arxiv.org/pdf/2610.41654

代码：https://github.com/OamIcs/NablaMimic

项目页：https://oamics.github.io/NablaMimic/

07操作常识不能只靠干净场景评估

MolmoSpaces-CommonSense: Evaluating Commonsense in Robotic Manipulation Policies

MolmoSpaces-CommonSense 将遮挡、容纳、支撑和抓取部位等常识纳入操作基准，暴露通用策略在杂乱环境中的弱点，适合补充纯动作执行评测。

论文：https://arxiv.org/pdf/2610.41377

代码：https://github.com/allenai/molmospaces

08把批量二次规划搬上 GPU

CuPIQP: A Batched and Differentiable QP Solver on GPUs for Robot Learning and Control

CuPIQP 支持稠密、稀疏和多阶段问题，并对各问题独立更新和停止，服务机器人学习中的批量优化。收益需要结合批大小、结构和比较实现判断。

论文：https://arxiv.org/pdf/2610.41021

代码：https://github.com/PREDICT-EPFL/cupiqp

09无需再训练，调整动作表征

STAR: Steering Action Representations with Proprioception for VLA Inference

STAR 对后层残差更新做指数滑动融合，并用近期本体状态调整混合强度，让动作表征继承较早层的物体信息。适合探索推理阶段的策略改进。

论文：https://arxiv.org/pdf/2610.41009

代码：https://github.com/PKU-VaLuE-Lab/STAR

10训练能写控制代码的机器人智能体

CaPtain: A Pipeline for Post-Training Code-as-Policy Robot Agents

CaPtain 将监督微调与多轮强化学习用于代码策略智能体，使小型视觉语言模型学习调用机器人接口。需区分训练阶段特权信息与测试时可用观测。

论文：https://arxiv.org/pdf/2610.40799

代码：https://github.com/waltstephen/CaPtain

项目页：https://waltstephen.github.io/captain/

11让代码智能体构建物理仿真

Gym-Anything for Robotics: Scaling physical simulations with digital agents

Gym-Anything 以创建与审查循环生成多行业机器人仿真任务，并研究如何通过交互经验提升智能体表现。仿真覆盖扩大仍需独立核验物理有效性。

论文：https://arxiv.org/pdf/2610.40565

代码：https://github.com/gym-anything/robotics

项目页：https://gym-anything.github.io/robotics/

12给机器人数据补上原子动作标签

RoboAtom: Large-scale Atomic-Action Supervision for Robot Learning

RoboAtom 为多来源轨迹补充统一时间对齐的技能标签，并构建识别和时间边界评测。其价值在于支持技能组合与分层执行，需留意原始数据的来源条件。

论文：https://arxiv.org/pdf/2610.40207

代码：https://github.com/InternRobotics/RoboAtom

项目页：https://internrobotics.github.io/RoboAtom/

13黑盒机械臂也能自动调参

Differentiable Iterative Learning Control for Black-Box Position-Commanded Manipulators

通过可微约束优化更新迭代学习控制参数，在不访问底层控制器的条件下改善重复轨迹跟踪。实机验证集中于单台机械臂和指定轨迹。

论文：https://arxiv.org/pdf/2611.01220

项目页：https://cortx-lab.github.io/projects/differentiable-ilc/

14无人机集群的运行时碰撞验证

Robust Safety Assurance for Robotic Swarms Using Decentralized Collision Verification

把可达性验证放到户外无人机集群，分析通信、时序和算力约束如何影响安全机制。价值在于真实部署中的假设检验，而非只看数值仿真。

论文：https://arxiv.org/pdf/2611.01081

代码：https://github.com/unl-nimbus-lab/arl-swarm/tree/verifier_2025

15验证损失低，不等于真机成功

Does Validation Loss Predict Real-Robot Success? Evidence from Four Imitation-Learning Policies on a Low-Cost Arm

在低成本机械臂上比较多类模仿策略的离线误差与闭环成功表现，揭示静止不动等失败无法由专家示范验证集充分暴露，提醒慎用损失跨模型排名。

论文：https://arxiv.org/pdf/2611.00283

项目页：https://huggingface.co/datasets/fysical/greenball_pool_225_v1

16跌倒后，先腾出起身空间

ROOF: Learning Humanoid Fall Recovery under Overhead Constraints

ROOF 学习头顶受限时的恢复动作，通过接触支撑和几何逃逸引导协调清障与起身。真机实验覆盖多种障碍，说明仿真恢复率不能替代现场成功率。

论文：https://arxiv.org/pdf/2610.42097

项目页：https://lu-yidan.github.io/ROOF/

17新任务视频如何转成可执行动作

WorldBridge: Bridging Video and Action for Task Generalization in World Action Models

WorldBridge 分开建模视频和动作专家，用结构化注意力连接二者。其迁移实验区分仅增加任务视频与再增加少量动作标签，避免混淆监督条件。

论文：https://arxiv.org/pdf/2610.40748

项目页：https://worldbridge-jd.github.io/

18用触觉对齐人手与机器手

TouchBridge: Tactile-Conditioned Human-to-Robot Transfer for Dexterous Manipulation

TouchBridge 通过对应的视觉和触觉接口保留接触位置与空间关系，无需物体网格即可重定向示范。结果限于指定真实接触任务及相同策略训练设置。

论文：https://arxiv.org/pdf/2610.40349

项目页：https://touchbridge2026.github.io/

19成功之外，评估操作质量

Shades of Success: Evaluating Manipulation Quality from Object-Centric Constraints

从成功示范中学习物体相对位姿约束，按任务进度衡量执行偏差。适合补充二元成功率，重点区分刚体操作的几何质量与任务是否完成。

论文：https://arxiv.org/pdf/2611.01250

项目页：https://spectra-manip.github.io/

20长任务记忆要记住状态变化

MOST: Memory of Object States and Task Progress for Long-Horizon Mobile Manipulation

MOST 用物体状态和子任务进度监督时序记忆，使动作生成能够利用已经离开视野的信息。阅读时应区分任务成功率和最大任务进度两种指标。

论文：https://arxiv.org/pdf/2611.00734

代码：https://github.com/diqihe666/MOST-vla

项目页：https://most-vla.github.io/

21不改指令，也能改变执行目标

Editing the Goal: Per-Token SAE Steering and Its Depth Limits in π_0.5

通过替换目标词对应的稀疏特征或激活向量，研究机器人策略内部的目标表示。实验是特定仿真场景下的存在性证据，效果依赖网络层与编辑方式。

论文：https://arxiv.org/pdf/2611.00602

项目页：https://doubleblind-sae-pi05.github.io/

22找东西要同时理解语义和遮挡

Knowing Where to Look: Interactive Object Search via Spatial-Semantic Fusion

把视觉语言模型的物体位置判断与几何占据信息融合为三维信念场，驱动换视角、移动和开柜搜索。系统在双臂移动人形平台上验证。

论文：https://arxiv.org/pdf/2611.00499

项目页：https://skymanaditya1.github.io/objsearch-website/

23触觉应当理解交互过程

TacMind: Benchmarking and Modeling Tactile Interaction States for Robotic Manipulation

TacMind 围绕精细操作、识别和动态交互构建仿真任务，利用历史触觉预测未来状态并修正候选动作，适合研究触觉在不同扰动下的贡献。

论文：https://arxiv.org/pdf/2611.00455

代码：https://github.com/Tacmind-Bench/TacMind

项目页：https://tacmind-bench.github.io/

24从部署轨迹中定位失败模式

Behavior Graphs: Unsupervised Graph Structures for Interpreting Robot Deployment Data

Behavior Graphs 将无结构执行数据压缩成可解释行为图，帮助识别易失败阶段并指导补采示范。其价值是把部署分析接回策略改进流程。

论文：https://arxiv.org/pdf/2611.00405

项目页：https://try.behavior-graphs.com/

25通过对话积累可复用技能

Teach, Abstract, Apply: A Framework for Dialogue-Driven Task Learning with Collaborative Robots

将自然语言教学、流程抽象和技能复用串联起来，在协作搭建任务中检验交互成本与成功表现；人参与实验仍属初步验证，需关注任务范围。

论文：https://arxiv.org/pdf/2611.00297

代码：https://github.com/clp-research/cobotprogsynthesis-coderep

26用弹琴统一检验灵巧控制

OmniPiano: Diverse Dexterous Piano-Playing Challenges for Standard, Robust, Safe, and Multi-Agent RL

OmniPiano 在同一任务族中覆盖多手协作、扰动和安全约束，适合比较强化学习方法。任务难度与手数可配置，结果不宜直接外推到通用灵巧操作。

论文：https://arxiv.org/pdf/2611.00243

代码：https://github.com/SAIL-Research-Lab/omnipiano

项目页：https://sail-research-lab.github.io/omnipiano/

27全身操作需要精度和稳定性

Anchor 1: Precise and Stable Humanoid Control for Whole-Body Manipulation

Anchor 1 使用校准后的稀疏动作参考与独立误差反馈，保留全身协调的自由度；评测同时考虑精度、持续稳定和可达性，避免只看动作相似度。

论文：https://arxiv.org/pdf/2611.00193

项目页：https://archon.tech/research/anchor1

28把通用能力接到显式验证结构

Scaling Trusted Autonomy: Towards Generally Capable Verifiable Robots

这篇学位论文串联语言约束落地、可编辑场景状态与技能效果预测，以神经符号结构支持规划和恢复。其保证仍依赖模型假设，不能等同无条件可靠。

论文：https://arxiv.org/pdf/2611.00158

代码：https://github.com/benedictquartey/robotlimp

项目页：https://robotlimp.github.io/

29自动改进冻结策略的执行接口

RoboArchitect: Automating the Design ofRobotic Agents with Persistent Memory

RoboArchitect 让代码模型读取执行轨迹、诊断失败并迭代修改控制接口，同时积累持久记忆。评估区分优化用回合与保留场景，避免把接口调优误作零适配。

论文：https://arxiv.org/pdf/2610.42141

代码：https://github.com/roboarchitect-ai/RoboArchitect

项目页：https://roboarchitect-ai.github.io/

30统一手术机器人多模态数据流程

Sensorium: An Open-Source Multimodal Data Collection and Curation Library for Embodied Surgical AI

Sensorium 基于 ROS 与 ROS 2 串联记录、处理、语义标注和回合数据导出，覆盖模型、离体与动物实验数据；主要贡献是可配置的数据工程流程。

论文：https://arxiv.org/pdf/2610.41946

代码：https://github.com/nct-tso-robotics/tso_sensorium

31从人类视频扩展人形动作模型

Movens-Zero: Scaling Humanoid Motion Foundation Models with Large-Scale Human Videos

Movens-Zero 将大规模视频处理、物理感知动作重定向与执行检验连接起来，研究多模态条件下的人形动作生成；需分别看语义一致性和实际执行质量。

论文：https://arxiv.org/pdf/2610.41877

项目页：https://movens-zero.github.io/

32失败可能来自仿真接触伪影

When the Simulator Fails, Not the Policy:Auditing Contact Artifactsin Simulated Robot Benchmarks

SimuGuard 记录物理步状态并重放异常，通过单参数干预定位接触碰撞错误。它提醒评测者先检查仿真器造成的物体弹飞，再把全部失败归因于策略。

论文：https://arxiv.org/pdf/2610.41857

代码：https://github.com/ALEXander22568/simuguard

33适配多种相机的视觉惯性里程计

OpenVIO: A Versatile Open-Source Visual-Inertial Odometry System

OpenVIO 支持针孔、鱼眼及拼接全景相机，并讨论系统各模块的设计取舍。适合需要跨相机部署的团队，比较时仍应对齐数据集与标定条件。

论文：https://arxiv.org/pdf/2610.41671

代码：https://github.com/93won/openvio

34边缘设备上的多视角实时策略

RTPolicy: Real-Time (32.9 Hz) Multi-View Visuomotor Policy for Robotic Manipulation on Edge Devices

RTPolicy 用视角融合与视觉压缩减少动作生成成本，并在双臂异步执行中验证。推理速度与芯片、数值精度相关，不能把设备上的频率视为通用指标。

论文：https://arxiv.org/pdf/2610.41625

项目页：https://d-robotics-ai-lab.github.io/RTPolicy.project

35用接触约束组织多机器人规划

Contact-based optimization for collision-free motion planning for a million robots

顺序插入冲突接触约束，减少直接离散优化整条轨迹的负担。大规模结果建立在规划动力学模型上，不等于同等规模实体机器人已完成部署。

论文：https://arxiv.org/pdf/2610.41548

代码：https://github.com/jankammeth/sci

36实验室长流程需要检查有效性

BioAgent: Self-Directed Skill Acquisition and Capability Growth for Long-Horizon Laboratory Protocols

BioAgent 根据经人工审核的需求复用技能，缺能力时补充教学，并在执行中检查前置条件、结果和实验状态。完整任务成功仍明显低于单技能成功。

论文：https://arxiv.org/pdf/2610.41383

项目页：https://suiuko.github.io/bioagentweb/

37从第一视角示范学习物体操作

HandMimicX: Learning Human-Like Dexterous Manipulation via an Object-Centric Policy from Egocentric Videos

HandMimicX 重建手物交互，再训练以物体目标为中心的仿真策略并转移到实机。核心是空间目标泛化，而非仅复制人的手部运动轨迹。

论文：https://arxiv.org/pdf/2610.41336

项目页：https://anonymous.4open.science/w/handmimicx/index.html

38复用求解分解，也要修正梯度

RapidDOC: Riccati-Preconditioned Implicit Differentiation for Differentiable Optimal Control under Operator Mismatch

RapidDOC 将前向分解缓存用于隐式微分的预条件，检查目标残差后再迭代修正，避免算子不匹配时直接复用产生错误梯度，适合可微最优控制系统。

论文：https://arxiv.org/pdf/2610.41324

代码：https://github.com/opt-man/RapidDOC

39示范迁移要保留接触过程

Contact Imitation Interface:Transferring Dexterous Skills by Showing How Contact Unfolds

CIMI 通过共享接触几何、指尖视图和优先接触对齐的重定向，连接人类示范与灵巧手控制。分层策略分别规划接触表面轨迹和腕指协调。

论文：https://arxiv.org/pdf/2610.41028

代码：https://github.com/ariszxxu/CIMI

项目页：https://nus-lins-lab.github.io/CIMIWeb/

40扑翼仿真要兼顾气动与控制

FlapSim: A Composable MuJoCo Simulation Platform and Reduced-Order Aerodynamic Model Hierarchy for Flapping-Wing Robotics

FlapSim 将机器人描述、翼几何和多层次气动模型组合进 MuJoCo，支持可重复实验。报告的定量气动比较主要使用固定基座，不应等同自由飞行验证。

论文：https://arxiv.org/pdf/2610.40902

代码：https://github.com/MingshanHe/FlapSim

41让机器人按任务自主换工具

SwapBot: Tool-Centric Robot Manipulation via Interchangeable End-Effectors and Tactile Sensing

SwapBot 通过统一连接接口和带触觉的低成本工具适配器支持自主换装，并配合遥操作采集示范。适合切割、刷涂等不同接触需求的家庭任务。

论文：https://arxiv.org/pdf/2610.40646

项目页：https://personalrobotics.github.io/SwapBot

42形变物体控制的基础教程

A Tutorial on Shape Servoing of Deformable Objects-Part I: Fundamentals

以变形雅可比连接机器人微小动作与形状变化，讲解离线估计、在线更新和局部稳定性。双臂操作弹性杆的平面仿真贯穿全文，适合建立方法基础。

论文：https://arxiv.org/pdf/2610.40559

代码：https://github.com/nachocz/tutorial_on_shape_servoing_of_deformable_objects

43用绘画观察语言与动作泛化

BURGER: Benchmarking and Understanding Robotic Generalization via Embodied dRawing

BURGER 用可扩展轨迹生成构建语言条件绘画任务，检验物体组合、空间关系与多解动作。实机还展示人类干预下的响应，适合细分泛化失败来源。

论文：https://arxiv.org/pdf/2610.40532

代码：https://github.com/Qxjyqzw/BURGER

项目页：https://siyingzhao.github.io/BURGER/

44课程收益还是简单起点收益

Is It the Curriculum or Just the Easy Start? Benchmarking Unsupervised Environment Design on Quadrupeds

在统一训练预算下比较多种课程与随机化方法，跨不同四足形态检验随机和保留地形。没有单一方法处处领先，选型应结合机器人与目标地形。

论文：https://arxiv.org/pdf/2610.40364

代码：https://github.com/Solvro/ml-wojtek-ued

45四足导航也能推开挡路物体

LLM-guided Planar Loco-Manipulation for Quadruped Robots

语言规划器联合选择技能、物体和几何目标，再交由低层强化学习执行。真机结果区分技能组合正确与端到端完成，说明规划正确仍不足以保证成功。

论文：https://arxiv.org/pdf/2610.40248

项目页：https://anonymousprojects-dev.github.io/LLM-guided-LocoManipulation-Quadrupeds/

46在人群中按轨迹预测碰撞风险

Don’t Risk It!: Trajectory-Conditioned Distributional Risk for LiDAR-Only Crowd Navigation

DRIP 直接使用激光雷达占据历史，预测候选动作序列对应的最小间距分布，并以碰撞结果约束风险学习，减少对易失效目标跟踪的依赖。

论文：https://arxiv.org/pdf/2610.40050

代码：https://github.com/dont-risk-it-planner/drip

项目页：https://dont-risk-it-planner.github.io/

47双臂衣物操作的统一评测

ClothDojo: Learned Data Generation and Benchmarking for Bimanual Garment Flattening and Folding

官方仓库提供衣物展平与折叠的仿真基准、固定初态和统一成功判据，并支持生成轨迹与随机化图像数据；本次未能读取论文全文，暂不转述性能数字。

论文：https://arxiv.org/pdf/2611.01190

代码：https://github.com/clothdojo/clothdojo.github.io

项目页：https://clothdojo.github.io/

48用编码器补足视觉短时失效

Visual-Inertial RGB-D SLAM with Encoders for a Differential Wheeled Robot

紧耦合视觉、惯性、编码器与深度观测，改善差速轮式机器人的定位鲁棒性。正文注明其已发表于 2022 年，本期作为新增索引中的历史工作补充阅读。

论文：https://arxiv.org/pdf/2610.41768

代码：https://github.com/leavesnight/ORB_SLAM2

49为在线规划增量更新距离场

GPU-accelerated Incremental Euclidean Distance Transform for Online Motion Planning of Mobile Robots

通过 GPU 并行计算局部距离信息，并将新观测融合进全局距离场，服务移动机器人规划。正文标明 2022 年接收，应作为历史工程方法阅读。

论文：https://arxiv.org/pdf/2610.41667

代码：https://github.com/JINXER000/GIE-mapping

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment