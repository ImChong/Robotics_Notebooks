# 一张照片变成仿真场景，机器人真能用吗？｜附代码与项目入口

> 原始抓取（2026-10-31，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/qUVNn3FiT89CDRzqmfN-0Q

一张照片变成仿真场景，机器人真能用吗？｜附代码与项目入口
具身智能小站 具身智能小站
 Oct 31, 2026, 1:48 AM

阅读路线

今天更新 50 篇具身智能新论文，覆盖世界模型与仿真重建、触觉与精细操作、策略学习与可靠性评测、人形交互与移动控制等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进47 篇扫读

先读 SceneRig，看重建如何接受机器人操作检验；再跟进 VLA-Retain 的适配评测口径与 Ropedia W0 的跨形态世界建模。其余按优先级扫读，结合资源入口选择深入方向。

先看这一篇

1让照片重建经得起机器人操作

SceneRig: An Agentic System for Simulation-Ready 3D Scene Reconstruction from Single Images

 

一句话判断：真实场景搬进仿真后，物体看起来相似，却可能一碰就倒、抓取位置错位。SceneRig把几何对齐与物理验证纳入重建循环，并进一步检查能否预测真实策略表现、复现操作轨迹，适合关心仿真评测与数据生产的团队深读。

你可以带走的两点

• 把操作可用性纳入重建评测：在五项任务、两种策略共100次真实回合的对应仿真中，成败一致率为69%，SimFoundry为54%；这不是机器人任务成功率。

• 看清回放分母与输入条件：对25条成功真机示范直接回放，SceneRig完成20/25，SimFoundry为9/25；两者共享控制与物理设置，重建均使用标定深度和相机参数。

方法亮点：把图像引导的位姿修正与物理仿真验证耦合到编辑循环中，同时用真实操作结果检验场景，而不只比较渲染相似度。 当前主要面向桌面刚体场景，尚不恢复物体关节与形变。

论文：https://arxiv.org/pdf/2610.28585

代码：https://github.com/Red-Fairy/SceneRig

项目页：https://red-fairy.github.io/SceneRig/

另外 2 篇值得跟进

2适配一个视角，会不会损失其他能力

VLA-Retain: Benchmarking Cross-Dimension Effects of Vision Tuning in VLAs

 

看点：换相机或降噪后的成功率变高，仍可能牺牲其他场景。VLA-Retain用同一模型调优前后的配对回合，把目标收益和非目标退化分开，适合部署验收与适配方案选择。

结果：五种VLA在七类扰动、每配置10,030回合上比较；十个模型—适配域组合中，8个出现非目标维度平均收益为负。

论文：https://arxiv.org/pdf/2610.27698

资源：https://github.com/BoruiXu/VLA_vision_tuning_benchmark

3用人类经验训练跨形态世界模型

Ropedia W0: Scaling Loco-Manipulation World Models across Embodiments with Human Experience

 

看点：人和机器人身体不同，动作数据难以直接混用。Ropedia W0用相机位姿和投影骨架统一运动条件，把人类视频经验用于跨形态未来预测，值得世界模型与数据团队跟进。

结果：训练数据约为1万小时人类视频与3000小时机器人视频；评估覆盖五个机器人基准，分别报告不适配、少样本和完整数据适配，核心指标包含视频与运动一致性。

论文：https://arxiv.org/pdf/2610.26855

代码：https://github.com/Ropedia/ropedia-w0

项目页：https://ropedia.github.io/ropedia-w0/

47 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04从机群失败中，检索可复用的成功经验

Learning from Fleet Experience: A Collective Learning Paradigm for Robots

CEL用失败情境检索机群成功片段，配合异步系统更新共享策略；真实多任务与机群规模实验支持收益，免新增人工纠错不等于初始策略无需人工数据。

论文：https://arxiv.org/pdf/2610.25736

资源：https://github.com/RLinf/RLinf

项目页：https://cel-robot-learning.github.io

05触觉安全，也要检查传感器物理层

PhantomTac: Hijacking Robotic Tactile Systems with Deploy-and-forget Tags

PhantomTac展示磁触觉测量受外界磁场误导后，抓取闭环可能出现异常；真机实验提示需校验视觉与触觉一致性，不能只相信单一传感通道。

论文：https://arxiv.org/pdf/2610.26174

代码：https://phantomtac.org/

06灵巧手基准，考验接触与技能组合

HandsOn: A Large-Scale Simulation Benchmark for Generalizable Dexterous Manipulation

HandsOn把示范学习任务与无目标任务示范的迁移测试分开，覆盖刚体、关节和形变对象；阶段诊断揭示接触建立、动作协调及成果保持的持续困难。

论文：https://arxiv.org/pdf/2610.26765

代码：https://github.com/TsinghuaC3I/HandsOn

07价值模型会打分，也要会识别倒退

RoboValue: A Fine-Grained Sim-and-Real Benchmark for Unified Evaluation of Robotic Value Models

RoboValue从任务状态、时序进度、失败恢复和价值一致性评测打分模型，覆盖仿真与真机数据；终局判断较强，不代表理解过程或恢复也可靠。

论文：https://arxiv.org/pdf/2610.26663

资源：https://github.com/RoboValue-Benchmark/RoboValue

项目页：https://robovalue-benchmark.github.io/

08用语言连接不同世界模型的动作

LaLa-World: Scaling Latent World Models with Natural Language Actions

LaLa-World用语言统一数值动作与视觉变化，在潜空间预测未来并支持规划、策略微调；匹配实验显示效率优势，不能外推为所有控制场景的加速。

论文：https://arxiv.org/pdf/2610.27851

代码：https://github.com/junha1125/LaLa-World

项目页：https://june-page.github.io/lala-world/

09把子目标留在训练阶段

VaDar: Implicit Goal Conditioning via Value Disaggregation

VaDar用多个子目标价值共同训练单一策略，部署时无需再选择子目标；长任务仿真对比支持评论器分解的价值，目前仍依赖预定义子任务结构。

论文：https://arxiv.org/pdf/2610.26629

代码：https://github.com/simchowitzlabpublic/vadar-public

10出现视觉扰动，先判断该修哪一种

What Would Removing This Corruption Change? Failure Attribution for Vision-Language-Action Models

将场景与扰动表征分离，以潜空间干预估计移除扰动的收益；仿真和Franka实验支持风险归因，当前仍依赖训练时已知扰动及其标签。

论文：https://arxiv.org/pdf/2610.27277

项目页：https://vla-attribute.github.io

11大模型会推理，精细动作仍难执行

Large Multimodal Models for Direct Robot Control: GPT-6 as a Beginning

GPT-GEAR系统测试通用模型直接控制，涵盖多个仿真基准、几何约束新任务及真机；揭示语义理解与精确运动之间的差距，推理延迟也是瓶颈。

论文：https://arxiv.org/pdf/2610.28259

代码：https://github.com/6UOOON9/GPT-GEAR

项目页：https://6uooon9.github.io/GPT-GEAR/

12不调参数，也能读懂触觉示范

Embodied Task Agents Can Perform Fine Manipulation through Tactile In-Context Learning

Tactile-ETA把动作与视觉、触觉片段对齐，作为冻结智能体的上下文经验；仿真和真机比较支持触觉信息增益，尚不是运动段内的高速触觉闭环。

论文：https://arxiv.org/pdf/2610.27613

代码：https://github.com/No-518/Tactile-ETA

项目页：https://no-518.github.io/tactile-eta/

13长任务评测，同时检查规则与物理

Embodied Playground

Embodied Playground以统一工具权限和预算评测多阶段操作、谜题与家庭任务，分别检查目标、规则及物理有效性；规则披露不足也会影响失败归因。

论文：https://arxiv.org/pdf/2610.27039

资源：https://github.com/Winniechen2002/Embodied-Playground

项目页：https://winniechen2002.github.io/Agent-bench-website/

14一台头戴相机兼做观测和动作标注

A Head Camera Is (Almost) All You Need: An Electronics-Free UMI Glove Labelled and Observed by One Moving Camera

头戴相机同时读取手套标记与工作区定位标记，减少采集硬件及同步链路；单任务真机对照呈现潜力，但小样本差异尚未达到常用显著性阈值。

论文：https://arxiv.org/pdf/2610.27622

代码：https://github.com/YosubShin/head-cam-umi

15别让合理的未来走错任务阶段

Plausible Future, Wrong Choice: Steering World-Action Models toward Intended Future

TaskWAM以任务要求和执行状态两类令牌约束未来生成，并用StageBench测阶段混淆；仿真、真机和多形态实验关注长任务的顺序一致性。

论文：https://arxiv.org/pdf/2610.28431

项目页：https://isee-laboratory.github.io/TaskWAM/

16确认完成之后，再更新任务记忆

EventSync: Event-Driven Memory Updates for History-Dependent Manipulation

EventSync结合图像和末端轨迹判断子目标是否完成，再写入历史并触发下一步；在RoboMME和四项真机任务验证，避免过早记忆误导后续动作。

论文：https://arxiv.org/pdf/2610.27917

项目页：https://zjjjacinthe.github.io/eventsync-project/

17多一个视角，补足视频动作歧义

Multi-View Latent Action Pretraining for Cross-Embodiment Robot Learning from Videos

同步多视角通过交叉注意力编码潜动作，减少遮挡和出平面运动歧义；跨形态与人手到真机实验支持这一路线，但精密操作仍需要机器人动作监督。

论文：https://arxiv.org/pdf/2610.27794

项目页：https://learnsyslab.github.io/multiview_latent_action_pretraining/

18重定向人手动作时，把物体一起考虑

DexOAK: Object-Aware Kinematic Retargeting

DexOAK用手—物交互图保留相对几何，再学习可快速推理的重定向网络；结合动力学修正提升双手操作，并完成真机开环回放，不能只看手姿态相似。

论文：https://arxiv.org/pdf/2610.26762

项目页：https://fspinola.github.io/dexoak/

19抓运动目标，先预测它将去哪里

TrajWAM: An Efficient Trajectory-Guided World Action Model for Dynamic Robotic Manipulation

TrajWAM用目标掩码轨迹引导未来表示，再分步蒸馏实现单步推理；动态仿真与传送带、转盘抓取验证，但主要处理目标独立运动的场景。

论文：https://arxiv.org/pdf/2610.25763

项目页：https://kevingeng3.github.io/TrajWAM/

20柔性机械臂换轨迹，也能复用学习

Reinforced Iterative Learning Control for elastic joint arms: Guaranteed cross-trajectory generalization under model uncertainties

RILC耦合迭代学习控制与强化学习，让轨迹经验参与新轨迹初始化；仿真及弹性关节机械臂验证误差改善，稳定性保证仍需满足论文假设。

论文：https://arxiv.org/pdf/2610.28329

代码：https://github.com/yursds/RILC_training

21让组合扩散策略真的形成分工

Learning to Be Different: Component Specialization for Compositional Diffusion Policies

对各组件的噪声预测加入去相关损失，减少多个组件重复学习同一动作；多基准和真机检验支持分工效果，但惩罚过强会损害任务成功率。

论文：https://arxiv.org/pdf/2610.27194

代码：https://github.com/Al3cL1/decorrelation_release

项目页：https://al3cl1.github.io/robotics-score-decorrelation/

22多机器人共享预测，也共享不确定性

CoopUQ: Robust Multi-Robot Social Navigation via Uncertainty-Aware Cooperative Forecasting

CoopUQ融合多机观测预测占据与不确定性，将高风险区域纳入约束强化学习；仿真分布外与真机导航均评估，目前聚焦同类二维激光机器人。

论文：https://arxiv.org/pdf/2610.26852

代码：https://github.com/tasl-lab/CoopUQ

项目页：https://coopuq.github.io/

23多机器人巡检，联合规划覆盖与避碰

A GPU-Resident Pipeline for Multi-Robot 3D Inspection with Limited Perception

将视点生成、可见性、集合覆盖、车辆路径与时空避碰串成GPU流水线；多网格实验核查覆盖和轨迹平滑后的碰撞，尚不是水下现场部署验证。

论文：https://arxiv.org/pdf/2610.26106

代码：https://github.com/krystofpsenicka/3D-Inspection

24预测真正影响动作的状态变化

Vision-Language-Delta-Action: Learning Compact Deltas as Actionable Transition Interfaces for Robotic Manipulation

VLDA以紧凑Delta表示任务相关变化，结合局部过滤、全局锚定与动作去噪；多仿真基准和双臂真机测试支持这条路线，减少完整未来预测的负担。

论文：https://arxiv.org/pdf/2610.25874

资源：https://github.com/baidu-baige/LoongForge/tree/master/cookbook/loongforge-vlda

项目页：https://baidu-baige.github.io/LoongForge/projects/vlda.html

25从人手视频学触觉表征

Touch-JEPA: Tactile Representation Learning from Hand-Object Video with Limited Tactile Supervision

Touch-JEPA在手部几何上预测被遮蔽的视觉特征，用少量触觉标注校准表示；覆盖三个人手基准与机器人迁移，触觉幅值不等于跨传感器力标定。

论文：https://arxiv.org/pdf/2610.28519

项目页：https://touch-jepa.github.io

26把交互世界重建成可修改程序

NeoWorld-3: From Pixels to Code-Defined Physical Worlds

NeoWorld-3将几何、关节和物理检查封装为工具，由代码代理迭代重建；物体、场景及双臂迁移实验展示价值，隐含质量与摩擦仍难从图像确定。

论文：https://arxiv.org/pdf/2610.27690

项目页：https://neoworldproject.github.io/Studio/

27把未来物理变化组织成动作词

Physical Words: A Vocabulary for Structuring Latent Reasoning in Robot Manipulation

PhysWord用原子动作词锚定未来视觉与触觉潜表示，再为动作生成提供上下文；仿真和双臂、灵巧手真机实验支持阶段推理，词表尚未覆盖全身控制。

论文：https://arxiv.org/pdf/2610.26854

项目页：https://nvidia-isaac.github.io/video_to_data/physicalword/

28会拒绝危险指令，未必会在行动前停下

Refuses When Asked, Complies When Acting: A Runtime Safety Check for Robot Control Loops

研究对比机器人动作循环与启动前安全询问，发现同一模型判断不同；小规模真机记录支持增设检查，但检查基于起始帧离线评估，未验证全程防护。

论文：https://arxiv.org/pdf/2610.27350

代码：https://github.com/Arnav-Gr0ver/Refuses-When-Asked-Complies-When-Acting

29把触觉图像直接交给通用模型

GPT-6 Astra Is a Native Visuo-Tactile Policy for Contact-Rich Manipulation

TacTalent以视觉接口输入触觉RGB图像，用少量示范驱动闭环操作；两套仿真基准显示接触信息价值，采样规模、推理成本与真机验证需一起看。

论文：https://arxiv.org/pdf/2610.26846

代码：https://github.com/ccyydd/TacTalent

项目页：https://tactalent.github.io/

30先快速重定向，再筛动力学风险

Fast Motion Retargeting Pipeline for Scalable Humanoid Robot Learning

FRDV把约束运动学重定向与可选动力学验证分开，支持人形动作数据批处理和G1迁移；论文的高吞吐结果不包含后续动力学验证成本。

论文：https://arxiv.org/pdf/2610.28470

项目页：https://daroslab.github.io/frdv/

31轮子打滑时，里程计该信多少

Learning Wheel Motion Uncertainty for Degeneracy-Aware LiDAR-Inertial Odometry on Quadrupedal Wheel-Legged Robots

学习每个轮子、每个方向的运动不确定性，再与激光和惯性约束紧耦合；真实退化走廊等场景验证定位改善，网络由仿真速度监督训练后迁移。

论文：https://arxiv.org/pdf/2610.26550

项目页：https://weihairuo.github.io/wheel-motion-uncertainty/

32让仿真教师在真实世界教视觉策略

ReDOA: Real-World Distillation via Observation Alignment for Image-Based Manipulation Policy Learning

ReDOA先把状态教师部署到真机采集轨迹，再对齐观测、蒸馏图像策略；三项非抓取操作验证效率与成功，前提是教师本身能跨过仿真到现实鸿沟。

论文：https://arxiv.org/pdf/2610.25691

项目页：https://isri-aist.github.io/redoa-project-page/

33力反馈何时该影响动作

FR-VLA: Implicit Force Reasoning VLA for Contact-Rich Manipulation

FR-VLA先由视觉语言上下文估计力反馈的重要性，再调节动作侧的力令牌通路；两平台接触任务验证阶段相关门控，细微力控修正仍是难点。

论文：https://arxiv.org/pdf/2610.25981

项目页：https://fr-vla.pages.dev/

34遮挡之下，人形机器人如何保持接触

VLM-Guided Contact Retargeting and Generative Partner Modeling for Humanoid-Human Interaction

REACT用语义接触重定向和伙伴状态预测，应对近身交互中的遮挡；在仿真与G1真机验证五类互动，部署依赖机载深度观测。

论文：https://arxiv.org/pdf/2610.28523

项目页：https://yutoshibata07.github.io/REACT/

35双人形互动，保留接触也保留运动能力

OmniInteract: Towards Unified Whole-Body Interaction Control for Dual Humanoids

OmniInteract用表面接触约束联合重定向双机动作，再在冻结运动先验上学习伙伴相关残差；仿真及真机交互展示协调，关注技能扩展时的能力保留。

论文：https://arxiv.org/pdf/2610.26043

项目页：https://calvinyhli.github.io/omniinteract.github.io/

36机器人是否真的需要一张认知地图

ForagerSim: Evaluating Embodied Map-Based Reasoning in Large-Scale Dynamic Worlds

ForagerSim独立调节感知、尺度、拓扑、动态和任务难度，对比学习策略、建图方案与人类；复杂大地图暴露推理瓶颈，人类参照样本也需谨慎解释。

论文：https://arxiv.org/pdf/2610.26395

代码：https://github.com/MrTomzor/foragersim

37一个操作者，同时管腿与手

LocoPilot: Single-Operator Bilateral Teleoperation for Loco-Manipulation Policy Learning

LocoPilot将四足底盘、机械臂和夹爪纳入统一遥操作接口，并提供估计力反馈；同一动作格式贯通采集与部署，真机策略验证规模仍有限。

论文：https://arxiv.org/pdf/2610.27998

代码：https://github.com/iit-DLSLab/LocoPilot

38遥操作延迟，也要换算成空间偏差

K-AoI: A Spatio-Temporal Synchronization Metric for Digital Twin-based Teleoperation in Dynamic Network Environments

K-AoI用瞬时笛卡尔速度加权信息年龄，让数字孪生同步评测反映运动后果；机械臂与Unity测试床覆盖不同网络条件，是对时间指标的补充。

论文：https://arxiv.org/pdf/2610.26201

代码：https://github.com/luziyaotian/kaoi-testbed

39统一接口后，再比较模型控机器人

Frontier LLMs Enter the World of Robotics: A Cross-Model Evaluation on RoboColiseum

Agent2Act在RoboColiseum统一观测、动作和反馈接口，对比指令、空间、操作与鲁棒性；结果显示模型各有长处，控制框架本身也影响表现。

论文：https://arxiv.org/pdf/2610.27865

项目页：https://robocoliseum.ai/leaderboard?metric=overall

40遥操作界面会改变示范质量

What Novice Operators See Changes What the Policy Learns: A User Study of Keypoint Hand Visual Representation for Teleoperated Demonstration

PointHand让操作者与策略共享稀疏手部关键点，研究视觉强调如何影响采集和学习；新手实验显示两者需联合设计，证据限于单一仿真装杯任务。

论文：https://arxiv.org/pdf/2610.27907

项目页：https://omron-sinicx.github.io/point-hand/

41冻结动作解码器，也能偏向好示范

AWLS: Advantage-Weighted Latent Steering for One-Step Generative Robotic Manipulation Policies

AWLS只训练潜噪声修正器，以优势值给示范动作加权，避免直接沿评论器梯度改策略；仿真与真机均验证，但不能凭空获得数据中缺失的技能。

论文：https://arxiv.org/pdf/2610.27736

项目页：https://awls-corl2026.github.io/

42探索时建图，结束后再精修

VEOcc++: Learning Persistent Voxel Feature Maps for Embodied Online Semantic Occupancy Mapping and Offline Refinement

VEOcc++用持久稀疏体素特征贯通在线语义占据建图与离线精修；基准和移动机器人验证迁移，仍依赖外部相机位姿，存储随场景增长。

论文：https://arxiv.org/pdf/2610.27704

项目页：https://wryzju.github.io/VEOccpp/

43让四足机器人自己分配蹬地与滑行

SkateNav: Learning Energy-Efficient Navigation through Adaptive Skating on Passive-Wheel Quadrupedal Robots

SkateNav用限时到达目标替代恒速跟踪，联合学习转向与推滑节奏；仿真对照和被动轮Mini Cheetah真机迁移显示潜力，仍需外部航点和时间预算。

论文：https://arxiv.org/pdf/2610.25962

项目页：https://ktw1404.github.io/SkateNav/

44侧身、弯腰，让全身一起避障

SWAP: Test-Time Steerable Humanoid Whole-Body Collision Avoidance

SWAP让高层扩散策略提出多种避障动作，低层强化学习维持平衡；用事后布障构造监督，仿真与G1真机验证，测试时可按速度或姿态偏好引导。

论文：https://arxiv.org/pdf/2610.25711

项目页：https://iscoyizj.github.io/SWAP-steerable-humanoid-oa-website/

45让四足机器人把CAD图纸变成砖墙

Towards Autonomous Bricklaying Using Boston Dynamics Spot

Spot在供料区与搭建区间移动，依靠标记定位和CAD目标位姿堆砖；实验采用木砖与重复小墙搭建，展示施工流程原型，尚非实际砌筑系统。

论文：https://arxiv.org/pdf/2610.27472

代码：https://github.com/soerensenkarl/MARA-bricklaying

46先分清选错目标，还是执行失败

FIT-MAC: A Closed-Loop Multi-Agent Framework for Robust Vision-Language-Action Models

FIT-MAC在冻结VLA外增加规划、定位、核验与纠错层，按失败类型介入；LIBERO-PRO仿真验证扰动恢复，额外干预开销仍待压低。

论文：https://arxiv.org/pdf/2610.28109

代码：https://anonymous.4open.science/r/research-artifact-B431

项目页：https://fitmac-research.pages.dev/

47用机器皮肤检验触觉传感器布局

Robotic Physical Comparison of Biologically Motivated and Counterfactual Sensory Arrangements using Biomimetic Skin

在柔性材料内布置不同传感器，比较仿生与反事实排列的弱接触、两点辨别和纹理感知；结果为布局设计提供依据，范围限于所测试的被动触觉任务。

论文：https://arxiv.org/pdf/2610.26680

项目页：https://poyotamu000.github.io/biomimetic-skin-sensory-arrangements/

48用示范经验指导触觉纠错

GPT-6-Astra Understands Touch and Outperform Vision–Tactile–Language–Action Models

将专家示范整理成任务指导与触觉参照，再由冻结多模态模型闭环出动作；在五项UniVTAC仿真任务比较策略，效果依赖示范和观测接口。

论文：https://arxiv.org/pdf/2610.28457

项目页：https://gpt-as-tactile-policy.github.io/

49把操作评测做成可复建的密室

HATCH: Hardware-Attested Task Chains for Verifiable Manipulation

HATCH提出由传感器确认任务完成、串联线索的实体密室评测，配套任务与搭建规范；目前主要是基准设计，自动评分和审计工具仍在路线图中。

论文：https://arxiv.org/pdf/2610.27485

资源：https://github.com/shizacharania/hatch-benchmark

50小样本模仿，先提取任务相关特征

Data-Efficient Learning from Demonstration for Humanoid Robots via Feature-Level Perception and Prompt-Based Object Detection

将人体关节、物体位置和深度提为紧凑特征，再预测NAO关节轨迹，降低从原始图像学习的负担；定量结果主要来自仿真，真机展示可行性。

论文：https://arxiv.org/pdf/2610.26170

代码：https://github.com/TaarLab/NaoLfD

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment