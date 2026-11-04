# 机器人做错了，奖励模型为什么还在加分？｜附代码与评测入口

> 原始抓取（2026-11-04，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/Em5CuCvWGpaWc_3R97YnYg

机器人做错了，奖励模型为什么还在加分？｜附代码与评测入口
具身智能小站 具身智能小站
 Nov 4, 2026, 6:00 AM

阅读路线

今天更新 57 篇具身智能新论文，覆盖机器人策略与奖励学习、触觉与灵巧操作、移动操作与导航、数据与评测平台等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进54 篇扫读

先深读D&A的奖励评估流程，再跟进ODE Policy的连续动作建模与uni-robot-agent的工具接口设计；其余论文可按触觉、数据、导航或平台需求回看。

先看这一篇

1别让奖励模型把失败看成进步

D&A: Decoupling Description and Assessment Enables VLMs as Zero-Shot Robotic Reward Models

 

一句话判断：机器人奖励模型容易把动作不断发生误当成任务持续推进。D&A先描述视频中的实际变化，再按明确成功标准评分，让失败识别与部分进度有据可查。对使用VLM筛选演示、构建奖励或生成纠错反馈的团队，这套流程比单次打分更值得检验。

你可以带走的两点

• 先描述，再评估：先识别视频里的对象和实际变化，再依任务成功标准估计进度；消融表明，仅拆分两个阶段仍不够，还要保证描述没有遗漏任务相关对象。

• 看排序相关性，也看使用成本：在RBM-EVAL-OOD同一Gemini 3.7 Flash消融中，Kendall τα由0.486升至0.626，这不是机器人成功率；部署还需考虑VLM时延、费用和语义幻觉。

方法亮点：将任务相关对象落地、客观轨迹描述与显式成功标准串成奖励估计流程，以任务完成条件替代视频内部时间顺序；同一描述还可用于生成纠错反馈。

论文：https://arxiv.org/pdf/2611.03622

代码：https://github.com/memmelma/dna

项目页：https://weirdlabuw.github.io/dna

另外 2 篇值得跟进

2让动作沿连续动力学自然衔接

ODE Policy: Robot Learning via Action Dynamics, Not Actions

 

看点：动作块边界的跳变会妨碍平稳控制。ODE Policy直接学习动作变化的向量场，从上一动作积分得到下一动作；覆盖模仿学习、动作模型和强化学习，适合关注平滑性与推理效率的团队。

结果：平滑性要连同推理条件比较：作者在14项真机任务中测试，并在单RTX 4090的药片操作实验中报告动作块生成10.3 ms，对比FM-DiT的36.3 ms；这不是所有平台通用的加速倍数。

论文：https://arxiv.org/pdf/2611.01729

代码：https://github.com/NTUMARS/ODE-POLICY

项目页：https://ntumars.github.io/ODE/

3用统一工具接口组织机器人智能体

Democratizing Robot-Use Agents: A Unified Harness with Open Models

 

看点：把模型推理、机器人动作与反馈放进统一且可替换的接口，让开放权重模型参与机器人操作研究；重点关注几何信息、工具可靠性和执行调度对能力的影响。

结果：把输入条件说清楚：仿真评测不使用特权输入，仍提供相机标定、机器人描述和观测反馈；共享工具支持推理与执行重叠，真机实验覆盖19项桌面评测，能力边界需要按场景判断。

论文：https://arxiv.org/pdf/2611.01560

代码：https://github.com/Open-RUA/uni-robot-agent

项目页：https://openrua.ai/uni-robot-agent

54 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04一次连续运动，同时校准位姿与时延

SyncHeC: Markerless Spatio-Temporal Hand-Eye Calibration via Differentiable Rendering and Continuous-Time Joint Optimization

以连续时间正运动学和可微渲染，联合估计手眼外参与图像时延；覆盖多臂与深度传感器，进一步检查时间补偿对下游策略成功率的影响。

论文：https://arxiv.org/pdf/2611.02999

代码：https://github.com/hongzhengdong/SyncHeC

05把未来接触经验蒸馏给当前策略

ForceLDM: A Force-aware Latent Dynamics Model for Contact-Rich Manipulation

训练时利用未来力矩和光流构建教师，再将接触动力学表征蒸馏给当前观测策略；在五类真机接触任务中检查新物体、扰动及噪声课程的作用。

论文：https://arxiv.org/pdf/2611.03294

代码：https://github.com/tpy001/ForceLDM-code

项目页：https://tpy001.github.io/ForceLDM/

06验证损失低，不代表后训练潜力高

Beyond Validation Loss: Coverage and Reliability Dynamics in Robot Policy Pretraining

用闭环pass@k区分一次执行可靠性与反复采样的成功覆盖，研究预训练检查点对后训练的影响；提出离线选点和生成增强，适合训练流程设计。

论文：https://arxiv.org/pdf/2611.01565

代码：https://github.com/real-stanford/policy-learning-dynamics

项目页：https://zhanyisun.github.io/robot-policy-learning-dynamics/

07到不同真实场地，重新比较机器人策略

RoboColosseum: Evaluating Physical Intelligence at Scale In-The-Wild

以分布式站点和本地配对试验比较机器人策略，区分直接部署与统一微调；强调跨形态排名和部分任务进度，避免只看单场景成功率来选模型。

论文：https://arxiv.org/pdf/2611.02497

资源：https://github.com/RoboColosseum

项目页：https://robocolosseum.ai/

08把双臂灵巧操作扩展到移动平台

MABEL: An Open Anthropomorphic Mobile Bimanual Platform for Teleoperation and Learning

将移动底盘、升降躯干、双臂和灵巧手接入统一控制接口，贯通遥操作采集与模仿学习；适合评估工作空间、集成成本与整机协同的团队。

论文：https://arxiv.org/pdf/2611.04083

代码：https://github.com/robotmabel/MABEL

项目页：https://robotmabel.github.io/website/

09把图像目标变成物理可行的位姿

VerifLM: Physically Verified Image-Edit-Guided Goal Pose Generation for Robotic Manipulation

将编辑图像里的目标关系转成六自由度物体位姿，再经多世界物理验证筛选；在几何与物理参数不确定时检查候选，避免只看图像觉得摆放合理。

论文：https://arxiv.org/pdf/2611.03268

代码：https://github.com/trustworthyrobotics/VerifLM

项目页：https://veriflm.github.io/VerifLM-webpage/

10让没有动作标注的视频帮助导航

OpenNWM: A Generalist Navigation World Model with Latent Action Pretraining

先从异构导航视频学习潜动作，再用真实物理动作做后训练；结合NavAnywhere评估跨场景预测与导航，适合研究无动作数据的规模和迁移价值。

论文：https://arxiv.org/pdf/2611.01844

代码：https://github.com/OpenSpace-Lab/OpenNWM

项目页：https://openspace-lab.github.io/OpenNWM/

11避开障碍后，回到原来的操作路线

Route-Guided CBF-QP Repair for Safe Vision-Language-Action Models

检测短动作块中的受阻路径，生成绕障后重接原路线的软引导；保留硬安全约束，在SafeLIBERO里同时看避碰、完成与安全成功，避免只追求不碰撞。

论文：https://arxiv.org/pdf/2611.01713

代码：https://github.com/Ckck12/Route-Guied-CBF-QP

项目页：https://ckck12.github.io/Route-Guied-CBF-QP/

12把策略评测失败变成可复用数据

Rollout then Recycle: 3,718 Labelled Real-Robot Episodes from Benchmarking 7 Policies on 12 Tasks

汇集同一批任务上多种策略的演示与执行轨迹，保留成功、次优和失败标签；适合奖励模型与失效分析，使用时需留意低成本平台和任务范围。

论文：https://arxiv.org/pdf/2611.03168

资源：https://huggingface.co/collections/armnet/armnetbench-v01

13在桌面对抗中研究真机多智能体学习

KlaskTron: An Open-Source Platform for Physical Adversarial Multi-Agent RL

用双CoreXY驱动的桌面对抗平台连接硬件、数字孪生与强化学习；通过学习执行器模型支持仿真策略转移，适合研究高速接触和多智能体竞争。

论文：https://arxiv.org/pdf/2611.03138

代码：https://github.com/IDSCETHZurich/klask_hardware

项目页：https://idscethzurich.github.io/klask_hardware/

14在接触对抗中检查模仿学习失效

KlaskTron-IL: A Contact-Rich Adversarial Imitation Learning Benchmark with a Validated Sim-to-Real Bridge

以真人、增强与合成数据分层研究对抗模仿学习，分析行为克隆和判别奖励的局限；复现论文和部署真机采用不同动作限幅，结果不能直接混合。

论文：https://arxiv.org/pdf/2611.03091

代码：https://github.com/IDSCETHZurich/klasktron-il

15让连续控制策略与搜索候选对齐

Iterative Gumbel Planning for Continuous Control

用分维离散策略对齐搜索动作，减少把胜出候选压回高斯分布的误差；结合Gumbel采样与多轮搜索，在连续控制基准中比较性能和样本效率。

论文：https://arxiv.org/pdf/2611.02236

代码：https://github.com/liushaohuai5/EZ-IGP

项目页：https://liushaohuai5.github.io/ez_igp/

16在每个噪声层约束回报分布

Value Forcing: Aligning Return Distributions at All Noise Levels

通过流匹配学习回报分布，并在每个噪声层施加贝尔曼一致性；形成两类离线学习算法，研究有限数据下的价值估计，实验属于长时仿真任务。

论文：https://arxiv.org/pdf/2611.02160

代码：https://github.com/chongyi-zheng/value-forcing

项目页：https://chongyi-zheng.github.io/value-forcing

17微调任务成功，也保住分布外能力

RAFT: Reasoning-Aligned Fine-Tuning Improves Robustness of VLA Post-Training

先用任务推理标注训练适配器，再冻结适配器学习动作，缓解微调后鲁棒性下降；作者在DROID、双臂YAM及仿真中检查视觉与指令变化。

论文：https://arxiv.org/pdf/2611.04368

代码：https://github.com/princeton-prism/RAFT

项目页：https://princeton-prism.github.io/RAFT/

18先理解任务，再寻找可执行目标

Feijoa: Task-Driven Grounding and Execution for Mobile Manipulation

根据当前指令反复查询RGB-D场景，而非依赖固定语义表示；结合可达性检查与执行验证，在Spot真机上展示开放指令下的移动操作与失败恢复。

论文：https://arxiv.org/pdf/2611.04080

项目页：https://williamshen-nz.github.io/feijoa/

19把仿真评测的等待时间用起来

RoboScope Kit: Unified Interfaces for Fast and Scalable Embodied-AI Evaluation

把仿真工作进程与策略副本分池调度，并减少动作块之间无用的渲染；重点是保持原基准评分规则，比较加速时需同时注明硬件与并发配置。

论文：https://arxiv.org/pdf/2611.03850

项目页：https://roboscope.pages.dev/

20先证明动作可行，再移动机器人

TAMPAgent: Agentic Belief-Space Planning for Mobile Manipulation

让语言代理维护物体位置假设，由规划工具确认抓取、可达与放置约束；以信念更新和动作认证处理遮挡与搜索，关注长任务中的感知与执行闭环。

论文：https://arxiv.org/pdf/2611.03584

项目页：https://umishra.me/TAMPAgent/

21为当前状态选择策略，也选择目标

Energy-Based Policy Orchestration

为每个低层策略学习可行性与目标误差相关的能量，运行时联合挑选策略和子目标；支持模块化接入，在人形移动操作中检验切换与目标修正。

论文：https://arxiv.org/pdf/2611.02546

项目页：https://yewr.github.io/epo/

22让人形机器人先缩手，再恢复平衡

ReflexArc: Bottom-Up Integration of Instantaneous Tactile Reflexes into Whole-Body Humanoid Control

把触觉反射通路直接接入电机，再通过策略训练接纳避让动作；在G1上研究突发接触后的撤离与平衡恢复，关键是反应时延和控制兼容性。

论文：https://arxiv.org/pdf/2611.04424

项目页：https://minusmodulo.github.io/reflexarc-website/

23人形换抓时，不能只模仿物体轨迹

ReContact: Reconstruct and Retarget Human Contacts from Video for Contact-Rich Humanoid Manipulation

从视频恢复物体中心的接触计划，再学习锚点、力向和滑移模式；用于遮挡下的换抓与支撑转移，仿真统计和G1真机展示需要分开理解。

论文：https://arxiv.org/pdf/2611.02968

项目页：https://recontact-robot.github.io/

24机器人自己交互，补上仿真的动力学

Closing the Dynamics Gap through Self-Directed Robotic Interaction

把视觉重建后的物理不确定性转成主动交互目标，由代理估计摩擦等参数；在台球设置下减少仿真预测误差，并展示真台球桌实验，尚非通用动力学保证。

论文：https://arxiv.org/pdf/2611.02691

项目页：https://cue-dynamics.github.io/

25用独立内镜视角检验手术三维感知

iMED: A Multi-Endoscope Dataset for Surgical 3D Perception

用同步双内镜数据评估跨视角几何泛化，避免把沿单一路径的图像插值当成三维理解；覆盖不同组织与运动条件，适合手术视觉评测。

论文：https://arxiv.org/pdf/2611.04755

代码：https://github.com/surgical-vision/imed

项目页：https://imed-challenge.github.io/

26让世界模型同时预测几何与触觉

Hybrid Particle-Voxel Dynamics for Multimodal 3D World Simulation

以粒子与体素混合表示建立动作条件世界模型，联合描述物体形变、触觉与多视角观测；面向操作任务，重点关注长时预测的物理一致性。

论文：https://arxiv.org/pdf/2611.04492

代码：https://github.com/jwang771/3d-world-simulator

项目页：https://jwang771.github.io/3dworldsimulator/

27保留因果结构，少量观测适配真机

A Practical Causal Framework for Autonomous Mobile Robots: From Simulation to Real-World Human Environments

将仿真中学到的因果结构迁移到人类共享环境，仅用少量真机观测更新参数；在移动机器人上验证，适合研究社会导航的适配与可解释决策。

论文：https://arxiv.org/pdf/2611.04027

代码：https://github.com/lcastri/causal-sim2real

项目页：https://lcastri.github.io/causal-sim2real/

28预测物体移动后，空间关系会怎样

FlowBench: Counterfactual Imagination Training for Spatial Intelligence

围绕假设动作后的场景变化构建反事实空间推理评测，连接室内场景与桌面操作；同时训练视觉语言模型和光流预测模型，考察仿真到真实迁移。

论文：https://arxiv.org/pdf/2611.03937

资源：https://github.com/liuyifan22/FlowBench

29让机械臂学会推，也学会吹

BeyondContact: A Reinforcement Learning Framework for Airflow-Based Non-Contact Manipulation

为Isaac Lab加入实验辨识的气流力模型，结合机械臂夹爪旁的风机训练推吹协作；通过真机零样本部署探索超出直接接触范围的物体操作。

论文：https://arxiv.org/pdf/2611.03059

代码：https://github.com/iit-DLSLab/BeyondContact

30先判断传感器可靠性，再做模态对齐

SomaAlign: Reliability-Aware Multimodal Representation Learning for Soft Robotic Grasping

用重建误差估计模态可靠性，再自适应调整对比学习并处理语义冲突；结合软抓取多模态数据评估检索与结果预测，预测准确率不等于抓取成功率。

论文：https://arxiv.org/pdf/2611.02794

代码：https://github.com/Hijackoo/SomaAlign

31用人类演示适配机器人策略

Human-to-Robot Transfer via Bridging Representation and Action Space in VLAs

同时对齐几何表征与动作潜空间，让未配对的人类和机器人演示共同适配VLA；分离动作解码器，减少跨形态监督对原有控制先验的干扰。

论文：https://arxiv.org/pdf/2611.04391

项目页：https://junhyeong-park.me/h2rb/

32只采手持数据，学习跨形态操作

AIRoA-0: Long-Horizon, Precise, Cross-Embodiment Manipulation from Handheld Data

用手持夹爪示范贯通预训练与后训练，通过腕部视角和匹配夹爪连接不同机器人；重点看陌生地点、长任务与精细操作，而非任意硬件通用性。

论文：https://arxiv.org/pdf/2611.03815

代码：https://github.com/airoa-org/AIRoA-0

项目页：https://a-0.airoa.io/

33把硬件、数据与学习流程一起设计

Pixel: A Co-Designed Full-Stack Robot System for Learning General-Purpose Mobile Manipulation

围绕全栈协同设计连接轮式人形硬件、手持采集、全身控制和策略训练；展示主动感知与移动双臂协作，适合系统团队检查各层之间的能力错配。

论文：https://arxiv.org/pdf/2611.03648

项目页：https://pixel-robot.com/pixel/

34让人类示范避开机器人的碰撞区

Pixel-UMI: Learning Environment-Aware Whole-Body Manipulation from Sensorized Human Demonstrations

把重建场景几何加入人体到机器人的全身重定向，再处理视角误差与动作时延；通过在线人工纠正补足接触信息，适合受限空间中的双臂任务。

论文：https://arxiv.org/pdf/2611.03647

项目页：https://pixel-robot.com/pixel-umi

35把强化学习专家收进同一个VLA

Beyond Policy Patches: Distilling RL Specialists into Vision-Language-Action Policies

先用专家数据初始化，再采集自身执行状态进行价值加权蒸馏；让专家只提供训练监督，部署回到单一VLA，在精细操作阶段比较成功率与动作平滑性。

论文：https://arxiv.org/pdf/2611.02777

项目页：https://repair-vla.github.io/

36把视频先验蒸馏给实时动作模型

W2A: Distilling World Priors into Action Models for Real-Time Robotic Control

训练时压缩视频扩散中间表征，部署时移除视频模型，降低世界动作模型推理负担；兼顾仿真和真机，速度比较仍需对齐骨干、硬件与推理配置。

论文：https://arxiv.org/pdf/2611.02600

项目页：https://misaka0502.github.io/W2A

37用预期触觉和实时反馈修正动作

TF²-Steering: Action Refinement with Tactile Forecasts and Feedback via Velocity-Field Steering

结合预测触觉和实时触觉，对流匹配动作专家做速度场残差修正；通过分阶段训练减少视觉先验受损，在仿真与真机中检查接触密集任务的收益。

论文：https://arxiv.org/pdf/2611.01483

项目页：https://yiming-cc.github.io/tf2-steering/

38柔性物体变形控制为什么会失效

A Tutorial on Shape Servoing of Deformable Objects—Part II: Advanced Concepts

从变形雅可比的奇异值分析出发，解释可控性、可观测性与机械失稳；配套教程和交互演示，适合梳理双臂柔性物体形状控制的适用边界。

论文：https://arxiv.org/pdf/2611.03825

代码：https://github.com/nachocz/tutorial_on_shape_servoing_of_deformable_objects

项目页：https://nachocz.github.io/tutorial_on_shape_servoing_of_deformable_objects/

39从智能戒指识别，走到机器人执行

OpenZilo: An Open Smart-Ring Platform for Human-Centric Sensing and Interaction with Robots

把语音、惯性手势与生理传感接到分层机器人接口，区分识别准确与任务完成；实验覆盖指令和完整任务，生理事件环节使用模拟通知。

论文：https://arxiv.org/pdf/2611.03410

代码：https://github.com/ziloai/openzilo

项目页：https://openzilo.com/

40把软指形变变成可见的接触线索

Robot-Matched Soft Fingers for UMI Data Collection: Lessons from 100 Hours of Pick and Place

让手持采集器使用与机器人匹配的柔软斜置夹指，通过可见形变传达接触线索；用多物体抓放研究相机配置与中间训练对新物体适配的影响。

论文：https://arxiv.org/pdf/2611.03779

代码：https://github.com/lute-corp/soft-yubi

项目页：https://lute-corp.github.io/soft-yubi-site/

41让触觉同时看到接触位置与剪切

PanoTac: A Dense Three-Axis Tactile Sensor for Robust Touch-Centric Manipulation

每个触觉单元同时提供法向与两轴剪切响应，形成稠密三轴触觉；通过试管准备、舀取和倾倒任务检验接触几何与力线索，区别于只做压力检测。

论文：https://arxiv.org/pdf/2611.03520

项目页：https://panotac.github.io/

42在人群与语义地形之间导航

SNAP: Semantic Navigation Among Pedestrians

以语义鸟瞰图统一表示地形与动态障碍，学习人群避让和语义通行；部署到四足机器人，关注随人流变化调整速度，同时遵守可通行区域约束。

论文：https://arxiv.org/pdf/2611.03498

项目页：https://snap-navigation.github.io/

43不放标定板，校准腕相机与固定相机

AnchorHEC: Targetless Hand-Eye Calibration of Wrist and Fixed Cameras

结合RGB重建和正运动学，先精化腕相机，再用度量场景点定位固定相机；围绕无标靶联合标定处理重建尺度误差，并保留物理机器人验证。

论文：https://arxiv.org/pdf/2611.02884

项目页：https://hwanggb.github.io/AnchorHEC/

44用四维运动引导动作模型关注交互

Gnomon4D: Learning Where and When to Focus for Geometric Motion-Grounded World-Action Modeling

联合预测视频、动作与四维几何运动场，把注意力集中到交互区域和抓放转换；借此连接无动作的人类视频预训练与机器人训练，关注几何表征的作用。

论文：https://arxiv.org/pdf/2611.02474

代码：https://github.com/Droliven/Gnomon4D/blob/main/codes/

项目页：https://droliven.github.io/Gnomon4D/

45把灵巧手长任务拆成可检验阶段

DexArena: A Unified Benchmark for Long-Horizon Dexterous Manipulation

面向双臂灵巧操作提供分阶段任务定义，同时衡量中间进度与整段完成；覆盖家庭和工业场景，有利于区分单技能能力和跨阶段衔接的失败来源。

论文：https://arxiv.org/pdf/2611.02629

资源：https://github.com/SIPAIR-Elastic/DexArena

项目页：https://sipair-elastic.github.io/DexArena-website/

46指令少说几句，机器人还会做吗

VagueBench: Benchmarking the Robustness of Embodied Agents on Ambient Context Illusions

删除指令中非必要的背景提示，同时保持任务语义不变，测试上下文依赖；以成对指令检查规划与执行退化，提醒评测不要只覆盖冗长明确的表达。

论文：https://arxiv.org/pdf/2611.03765

代码：https://github.com/stardreaming/VagueBench

47机器人太重，会增加模仿学习难度吗

When Robots Outweigh Objects: How Inertia Mismatch Shapes Robot Policy Learning

通过受控仿真改变机器人、物体质量与传动反射惯量，发现接触惯量失配会影响行为克隆；也讨论在既有硬件上选择更低惯量的示教姿态。

论文：https://arxiv.org/pdf/2611.03501

项目页：https://younghyopark.me/inertia-mismatch/

48在噪声与动作空间共同微调策略

CANDO: Conservative Action-Noise Directional Optimization for Residual Reinforcement Learning

把噪声方向调整与动作残差结合，保留生成式策略先验并进行在线改进；以保守价值估计和门控执行约束探索，适合残差强化学习研究。

论文：https://arxiv.org/pdf/2611.04543

项目页：https://cando-rl.github.io/

49在连续坐标里做机器人特征匹配

CoReMatch: Recurrent Continuous Coordinate Refinement for Real-Time Robotic Feature Matching

不再只在固定网格上预测匹配置信度，而是循环细化连续坐标；面向定位、建图与导航，在室内外基准和边缘硬件上比较精度与运行效率。

论文：https://arxiv.org/pdf/2611.03932

代码：https://github.com/edge-cv/CoReMatch

50球形平衡机器人的避障不能忘了倾角

Safe-Ballbot: Tilt-Consistent MPC with Control Barrier Functions for Ballbot Navigation

用允许倾角推导规划加速度上界，连接上层MPC和下层平衡控制；结合控制屏障与预测误差膨胀避障，目前证据来自含轮接触和力矩饱和的仿真。

论文：https://arxiv.org/pdf/2611.03307

项目页：https://safe-ballbot.github.io/

51分开更新路由器与动作专家

MAPO: Mixture of Experts with Alternating Policy Optimization for Universal Humanoid Motion Tracking

针对MoE动态路由造成的策略散度估计偏差，交替优化路由器与专家；以更新约束的正确性改进通用人形动作跟踪，重点是训练稳定性与覆盖范围。

论文：https://arxiv.org/pdf/2611.02613

项目页：https://mapo-corl2026.github.io/MAPO/

52为导航实验自动生成可比轨迹

TRAILS: Trajectory Generation for Automated Indoor Locomotion Studies

把室内布局、贝塞尔路径和速度曲线联合生成，控制运动约束与实验变量；适用于虚拟导航和遥临场研究，不应把场景生成能力等同真实导航成功率。

论文：https://arxiv.org/pdf/2611.03254

代码：https://github.com/JucciK/Trails

53局部修补仓储机器人的低效计划

Local Solution Repair for Multi-Agent Pickup and Delivery

检测回退、拥堵停等行为，对多机器人取送计划局部修复；在长期仓储仿真中提高吞吐量，适合已有调度器增量优化，不能直接外推真仓效率。

论文：https://arxiv.org/pdf/2611.02090

代码：https://github.com/Ethan-Schneider/LSR.git

54动态机器人边界下的流场预测

DynaSolver: A Neural Operator for Autoregressive 4D Flow Prediction with Dynamic Boundaries

面向关节运动引起的动态边界，建立机器人四维流场基准；结合物理token与回放增强训练，研究长时自回归误差，主要价值在流体仿真代理。

论文：https://arxiv.org/pdf/2611.03595

项目页：https://made-gpt.github.io/DynaSolver-ProjectPage/

55让开放机器人集群管理成员资格

Self-governed access control in open robot swarms using a blockchain-based DAO

用智能合约规定机器人加入、移除与访问规则，研究开放集群治理；通过仿真和实物概念验证抑制恶意节点渗入，仍需结合通信和计算条件评估。

论文：https://arxiv.org/pdf/2611.03975

代码：https://github.com/rainazakir/dao

56拆开多源估计中的信息贡献

Source-Resolved Collective Cognition in Robotics: Mathematical Foundations for Multimodal and Collective Estimation, and Their Reach across the Cognition Loop

围绕广义奇异值分解梳理多模态与群体估计，把信息归因、子空间几何和统计依赖分开讨论；偏数学基础与数值参考实现，不属于真机策略性能报告。

论文：https://arxiv.org/pdf/2611.02693

资源：https://doi.org/10.5281/zenodo.23862358

57共享空间里的机器人私密交流

Just Between Us? Exploring a Wearable-Mediated Channel for Discreet Human-Agent Interaction in a Shared Game Setting

通过耳内音频与侧倾头部建立定向交流通道，研究共享游戏中的可用性与公平感；实验采用研究员操控的屏幕代理，不能等同自主机器人。

论文：https://arxiv.org/pdf/2611.04692

代码：https://github.com/atikkhannilgar/Discreet_Interaction

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment