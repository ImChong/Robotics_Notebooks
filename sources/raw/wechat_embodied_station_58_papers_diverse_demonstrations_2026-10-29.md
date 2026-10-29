# 机器人缺的，是更多示范，还是更多成功解法？｜附代码与项目入口

> 原始抓取（2026-10-29，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/s23oOBojr5YbchuFmQhcwA

机器人缺的，是更多示范，还是更多成功解法？｜附代码与项目入口
具身智能小站 具身智能小站
 Oct 29, 2026, 6:00 AM

阅读路线

今天更新 58 篇具身智能新论文，覆盖操作数据与策略学习、世界模型与具身记忆、导航与人形控制、触觉与安全评测等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进55 篇扫读

先深读DiscoDemo的成功示范多样性，再跟进OpenTac的触觉工具链和Watch Your Step的落脚感知；其余按优先级扫读，可从各篇资源区进入原文。

先看这一篇

1让强化学习生成多种成功示范

DiscoDemo: Discovering Efficient and Diverse Robot Demonstrations for Imitation Learning

 

一句话判断：模仿学习缺的往往不是更多相似轨迹，而是多种可成功的解法。DiscoDemo把强化学习用于数据生成，以逆向课程和多样性目标扩展覆盖，再训练同一个视觉策略；四项仿真和真机任务提供了较清楚的对照。

你可以带走的两点

• 把RL当作示范生成器：并行逆向课程先获得成功行为，再以成功门控的多样性目标发现不同解法，经安全筛选导出静态数据集；下游策略无需调用生成器。

• 比较时固定数据量和策略：各生成方法每任务提供3,000条成功示范，微调同一DROID预训练π0.5；真机每任务20次、固定第一训练种子，四任务平均成功率为77.5% 对 38.75%（P-RFCL）。

方法亮点：新增机制是让成功门控的解法多样性参与生成器训练，再把收益落实到下游视觉模仿，而非仅优化RL控制器自身回报；过强多样性也会损害任务能力。

论文：https://arxiv.org/pdf/2610.32093

代码：https://github.com/DAVIAN-Robotics/DiscoDemo

项目页：https://davian-robotics.github.io/DiscoDemo/

另外 2 篇值得跟进

2把三维触觉皮肤接入设计、仿真和控制

OpenTac: An Open Ecosystem for Customizable, Scalable, and Transferable 3D Tactile Sensing

 

看点：机器人触觉落地常卡在传感器形状、制造与软件接口之间。OpenTac把柔性皮肤设计、读出、仿真和跨传感器转换连成工具链，并在多种机器人上验证法向与剪切反馈。

结果：同一传感器描述文件连接制造、采集、仿真和跨皮肤转换；论文验证多种机器人上的接触控制与学习，跨皮肤实验测的是接触形状迁移。

论文：https://arxiv.org/pdf/2610.31661

代码：https://github.com/opentac/opentac

项目页：https://opentac.github.io/

3只看下一步落脚附近的地形

Watch Your Step: Anticipated Perception for Humanoid Locomotion

 

看点：感知行走不一定需要把整张高程图都交给策略。该工作先预测未来落脚点，再读取邻域小块地形；训练更省，也在G1上展示迁移，但可能漏掉侧方接近的障碍。

结果：用本体状态和速度指令预测未来两个落脚点，只读取相应高程图小块；模型与行走策略交替更新，避免让策略从全图中自行寻找相关位置。

论文：https://arxiv.org/pdf/2610.29612

代码：https://github.com/leggedrobotics/foothold-prediction-model

项目页：https://leggedrobotics.github.io/watch-your-step

55 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04用真实未来蒸馏一步可读的世界表征

Self-Distilling Future Video Representations for World Action Models

FRED以真实未来的教师表征监督带噪预测，并加入历史上下文；真机长程、环境泛化与人到机器人迁移实验提示，动作解码器需要可靠的未来表征。

论文：https://arxiv.org/pdf/2610.30320

代码：https://github.com/suhyeok-jang/FRED/

项目页：https://huiwon-jang.github.io/FRED/

05先显式预测三维物理状态，再生成视频

DeXplicit: Making Physical State Explicit in Dexterous Manipulation World Models

DeXplicit先预测物体运动和形变，再以几何结果引导视频合成；多数据集对照支持这种拆分，生成加速来自特定配置比较，不宜当成统一硬件上的通用倍数。

论文：https://arxiv.org/pdf/2610.30781

代码：https://github.com/Abecid/DeXplicit

项目页：https://dexplicit.github.io

06把后训练用到机器人规划器本身

AgenticRoboRL: Bringing Coding-Agent Post-Training to Agentic Robotics

AgenticRoboRL固定执行组件，通过监督微调和多轮强化学习改进规划器；在LIBERO系列和不同执行框架上验证，主要证据来自仿真工具调用与代码生成。

论文：https://arxiv.org/pdf/2610.28589

代码：https://github.com/xinxinliu-ai/AgenticRoboRL

07让技能代码、未来视频和动作一起生成

Agentic World Action Models

AW-0将程序、视频与控制轨迹接入统一专家架构，并自动整理可复用技能；仿真及双臂真机实验检验规划和执行的连接，值得关注代码如何成为预测的一部分。

论文：https://arxiv.org/pdf/2610.30136

代码：https://github.com/ZhuoyangLiu2005/agentic-wam

项目页：https://agentic-wam.github.io/

08从带噪的人类动作中学习可靠监督

EgoForesight: Effectively Scaling Policy Learning with Noisy Egocentric Actions

EgoForesight按数据源可靠性分配动作噪声与损失权重，并用真实未来蒸馏表征；扩展实验支持收益，但源级加权仍不能消除系统性重定向偏差。

论文：https://arxiv.org/pdf/2610.31845

代码：https://github.com/LinShan-Bin/EgoForesight

项目页：https://lin-shan.com/EgoForesight

09为机器人数据统一生成分段、描述与奖励

FLORA: An Open Model Suite for Scaling Robotics Data Engines

FLORA把时间分段、动作描述和过程及结果奖励接入同一数据引擎；评测覆盖描述与奖励基准，部署规模和成本比较需结合论文的硬件与计价假设。

论文：https://arxiv.org/pdf/2610.31779

代码：https://github.com/NVIDIA/Flora

项目页：https://research.nvidia.com/labs/gear/flora/

10先用感知运动对齐筛选候选策略

Sensorimotor Alignment: A Platonic Proxy for Embodied Policy Success

Sensorimotor Alignment比较感知表征与末端轨迹的几何结构，为昂贵闭环评测提供离线筛选信号；跨模型相关性值得参考，但不能替代真实任务成功率。

论文：https://arxiv.org/pdf/2610.31670

代码：https://github.com/yunfeixie233/sensorimotor-alignment

项目页：https://yunfeixie233.github.io/s2/

11把三维理解内化成可持续更新的记忆

GeoMem-VLA: Internalizing 3D Scene Understanding into Temporal Memory for Vision-Language-Action Models

GeoMem-VLA以几何教师监督专用词元，并随时间检索与整合关键观察；经过各基准下游适配后在仿真和真机验证，推理阶段无需调用三维教师或重建完整场景。

论文：https://arxiv.org/pdf/2610.29866

代码：https://github.com/GeoMem-VLA/GeoMem-VLA

项目页：https://geomem-vla.github.io/

12只用一个VLA估计认知不确定性

Post-hoc Epistemic Uncertainty for Vision–Language–Action Policies

事后核方差头从单策略嵌入估计知识缺口，无需梯度更新；SmolVLA与LIBERO-10支持筛选示范和转交监督，但任务间覆盖差异不能替代任务内风险排序。

论文：https://arxiv.org/pdf/2610.29730

代码：https://github.com/post-hoc-vla-uncertainty/post-hoc-vla-uncertainty

13连续时间内检查约束，而非只看离散点

Continuous-Time Motion Planning and Steering with Global Convergence

GlobMP用分段多项式与梯度投影联合处理规划及策略轨迹修正；实验覆盖多机器人和双臂操作，全局收敛结论依赖细网格起点、分段增长等理论条件。

论文：https://arxiv.org/pdf/2610.30607

代码：https://github.com/GlobalMotionPlanning/GlobMP

项目页：https://globalmotionplanning.github.io

14把理解、视频预演和控制接到同一底座

RynnUnify: Towards Unified Embodied Brain

RynnUnify以推理器和视频生成器交替规划，并扩展到机器人动作；实验覆盖具身理解、长程视觉规划与真机，但各能力的评测范围需分别看待。

论文：https://arxiv.org/pdf/2610.32965

代码：https://github.com/alibaba-damo-academy/RynnUnify

项目页：https://alibaba-damo-academy.github.io/RynnUnify.github.io/

15给示范重新配速，而非整体加速播放

Retiming Robot Demonstrations with a Learned Speed Prior

学习式速度先验为不同轨迹分配局部节奏，再以约束控制器执行动作块；仿真与真机支持效率收益，比较时需同时观察成功率和运动平滑度。

论文：https://arxiv.org/pdf/2610.32676

项目页：https://mllab.snu.ac.kr/projects/learned_speed_prior/

16导航策略自己选择该看的表征

See What I Need, Dream What I Want: A Navigation Foundation Model on a Self-Constructed Representation Space

RepresentNav跨视觉模型层级重组特征，并在行动前预测潜在未来；语言导航、跟踪和真机实验均有验证，核心启发是让导航目标决定视觉接口。

论文：https://arxiv.org/pdf/2610.31762

项目页：https://wang-yinxi.github.io/representnav/

17目标到来前看到的线索，也要记住

Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation

WholeBodyMem按物体事件选择历史关键帧，供全身动作专家读取；HuMeGo包含仿真及G1真机任务，结果支持保留关键时刻，而非简单增加均匀采样帧。

论文：https://arxiv.org/pdf/2610.31166

项目页：https://memory-before-goal.github.io

18只靠外部相机识别设备异常

Adaptive World-Models for Anomaly Recognition (AWARE): A Framework for Robot Dynamics Prediction and Introspection

AWARE从相机轨迹估计隐含动力学，再结合预测误差识别异常；在真实起重机上验证碰撞与电机退化，正常工况变化仍可能触发误报。

论文：https://arxiv.org/pdf/2610.32742

代码：https://github.com/adaptive-intelligent-robotics/aware

项目页：https://adaptive-intelligent-robotics.github.io/aware/

19把运营规则写进Nav2代价地图

Natural-Language to Nav2: Operational Policies for Costmap-Based Navigation

将自然语言转为经确定性检查的导航策略，再投影成禁行区与代价层；语言解析仍有误接收，导航收益是在使用正确任务策略的条件下测得。

论文：https://arxiv.org/pdf/2610.32633

代码：https://github.com/a56239637-dot/Natural-Language-to-Nav2-Operational-Policies-for-Costmap-Based-Navigation

项目页：https://a56239637-dot.github.io/Natural-Language-to-Nav2-Project-Page/

20大规模流匹配策略，一步采样可能已够用

When One Step Is Enough: An Empirical Study of Euler Integration in Flow-Matching Policies

对多种VLA的实证分析发现，充分训练且容量足够时，一步欧拉采样常可接近多步表现；无需重训的部署机会值得试，但仍应结合执行时域逐任务验证。

论文：https://arxiv.org/pdf/2610.30735

项目页：https://onestepflow.github.io/

21积累执行经验后，减少逐动作审核

FastHarness: Experience-Driven Review Scheduling for Vision-Language-Action Policies

FastHarness冻结底层策略，根据阶段记忆和审核结果调节大模型监督频率，失败时切入恢复流程；四项仿真任务支持效率收益，尚不等于真机部署验证。

论文：https://arxiv.org/pdf/2610.31427

代码：https://github.com/FastHarness/fast-harness

22把自由漂浮抓取纳入空间操作评测

AstroBench: Benchmarking Free-Floating Capture and Space Robotic Manipulation

AstroBench区分物体运动条件、捕获终点与空间工位任务；通用策略直接迁移和专用ACT来自不同实验配置，应分别解释，不能将两者当成严格同条件排名。

论文：https://arxiv.org/pdf/2610.31064

代码：https://github.com/AstroUpLab/AstroBench

项目页：https://astroup.com/astrobench/

23把任务变化与人类协作放进长期评测

TARSphere: A Robot-Oriented Full-Stack Framework for Long-Horizon Embodied Agents in Evolving Physical and Social Environments

TARSphere将物理执行、角色对话与事件驱动任务状态统一进仿真框架；配对条件检验扰动、任务更新及意图理解，提醒成功率之外还要统计碰撞和跌倒。

论文：https://arxiv.org/pdf/2610.30043

代码：https://github.com/xuyicheng-zju/TARSphere

项目页：https://xuyicheng-zju.github.io/TARSphere/

24探索地图时，也考虑语义信息的新颖性

AOV-GS: Active Open-Vocabulary Semantic Mapping with 3D Gaussian Splatting

AOV-GS以冻结的SAM与CLIP为候选视角排序，再拟合开放词汇语言场；室内仿真显示语义地图收益，但实验使用真实相机位姿，尚未覆盖SLAM定位噪声。

论文：https://arxiv.org/pdf/2610.29374

代码：https://github.com/OptimusFaber/AOV-GS

25把人形移动操作的策略和控制器分开比较

LocoManiLab: A Unified Testbed for Data, Policies, Controllers, and Deployment in Humanoid Loco-Manipulation

LocoManiLab打通采集、训练、控制器与部署，提供家庭场景的原子和组合任务；统一对照揭示未见任务、视觉扰动及控制器选择对性能的不同影响。

论文：https://arxiv.org/pdf/2610.30876

项目页：https://locomanilab.github.io/

26跨人形机器人复用同一套奖励结构

MPBRS: Multi-Potential-Based Reward Shaping for Unified Reward Configuration in Humanoid Locomotion

MPBRS以多个势函数差分构造运动奖励，在不同人形机器人上复用结构和权重；仍需各机器人的关节映射及默认姿态，复用奖励不等于零适配迁移策略。

论文：https://arxiv.org/pdf/2610.30334

代码：https://github.com/aralab-unr/MPBRS

27手持灵巧手采集，保留真正的动作指令

DexPort: A Hand-Held Skill Interface for Dexterous Hand Data Collection

DexPort复用采集与部署的手部硬件，用少量参数化技能协调手指；四项真机任务显示，接触时应记录指令而非仅记录关节状态。

论文：https://arxiv.org/pdf/2610.32969

项目页：https://amberxie88.github.io/dexport/

28VLA采样步数要和执行时域一起调

Coupling Execution Horizon and Sampling Depth in Flow-Matching VLA Policies

研究发现流匹配策略的采样深度与开环执行长度存在耦合；仿真及Franka实验提示，减少步数的效果随重规划频率变化，不能只看单次推理耗时。

论文：https://arxiv.org/pdf/2610.32102

项目页：https://jiabingyang01.github.io/Horizon-Depth-VLA/

29让编程智能体读懂力反馈再改程序

FORCA: A Force-Aware Coding Agent for Contact-Rich Robot Manipulation

FORCA把腕部力矩记录与视觉结果一起用于诊断和修改控制代码；五项真机任务覆盖探触、擦拭等接触操作，无需任务示范仍依赖受控执行接口。

论文：https://arxiv.org/pdf/2610.31841

项目页：https://tanoom.github.io/forca/

30风险高或证据不足时，先交给恢复策略

SEVI: Safe Evidential Real-World Reinforcement Learning

SEVI用证据式安全评论器同时估计约束风险与认知不确定性，再决定是否覆盖动作；验证覆盖仿真和真机USB插入，其安全表现仍受训练分布约束。

论文：https://arxiv.org/pdf/2610.31829

项目页：https://sevi-safe.github.io/

31从RGB人类视频同时学手部动作和空间背景

Reconstruction-Augmented Visuomotor Pre-Training of 3D-Grounded Bimanual Dexterity from RGB-Only Human Videos

SpatialDex重建场景几何并估计手部运动，以门控融合和未来预测支持双手预训练；仿真及真机收益是在预训练后用机器人示范微调取得，并非直接零样本操作。

论文：https://arxiv.org/pdf/2610.29716

项目页：https://spatial-dex.github.io/

32把未来要执行的子任务提前写进记忆

Plan First, Act Later: Streaming Reasoning for Executable Memory in Robot Control

PAL将视觉流整理成事件和可执行子任务清单，执行时选择相关项交给VLA；RoboMME和真机实验支持提前构建记忆，训练还需处理滚动执行中的状态分布。

论文：https://arxiv.org/pdf/2610.29329

项目页：https://qinghangliu.github.io/pal-webpage/

33部署中吸收人工干预，在线更新策略

Training in Deployment: Human-in-the-loop Online Post-Training for Robot Manipulation

TIDE区分人工干预与策略生成数据的来源，以监督学习在线后训练；双臂真机实验支持短时适配，但仍依赖人工确认，长期遗忘和跨平台泛化尚未验证。

论文：https://arxiv.org/pdf/2610.29264

项目页：https://logosintelligencelab.github.io/TIDE/

34视觉慢更新，触觉快速修正未执行动作

T-WLA: Bridging Slow Vision and Fast Touch

T-WLA缓存视觉记忆，供轻量触觉编辑器修正动作块，无需每次重跑视觉分支；手部预测与真机操作实验支持多速率设计，未来图像和压力仅作训练监督。

论文：https://arxiv.org/pdf/2610.28814

项目页：https://anonymous-submission-20.github.io/T-WLA.github.io/

35世界模型评策略，不能只追求画面逼真

Physical Canvas: A Vision Tactile World Model for Robot Policy Evaluation and Learning

PhysCanvas将视觉、触觉和任务状态放入统一画布，以一致性监督约束生成；四项真机任务检验模型评估与策略后训练，关注预测成功率和真实执行的相关性。

论文：https://arxiv.org/pdf/2610.28613

项目页：https://zebinx.github.io/PhysCanvas_HomePage/

36奖励函数搜索也需要分工与历史记忆

R2E: A Role-driven Reward Evolutionary Framework for Automated Reward Function Design

R2E让探索、保守改进和重组三种角色协作，并用精英池保存有效奖励；在仿真运动与双手任务中检验，启发是管理搜索过程而非只增加采样。

论文：https://arxiv.org/pdf/2610.32439

代码：https://github.com/ChangYuandao/R2E

37让生成的语言成为下一步动作的记忆

UniWALM: Unified World Action Language Model for Embodied Robot Control

UniWALM在同一骨干中联合去噪视频、动作与语言，并将文本记忆送回下一轮；记忆任务和真机展示收益，但完成时机判断、抓取精度与推理速度仍有限制。

论文：https://arxiv.org/pdf/2610.28626

项目页：https://rahulsajnani.github.io/projects/uniwalm/

38从真实示范重建可交互的仿真世界

WorldMirror: Agentic Real-to-Sim Generation of Interactive Worlds from In-the-Wild Robot Demonstrations

WorldMirror把真实到仿真建模写成可反复修订的世界程序，结合生成工具和验证反馈；DROID实验同时检查视觉、空间关系与轨迹回放的物理一致性。

论文：https://arxiv.org/pdf/2610.31834

项目页：https://worldmirror-page.github.io/

39双手追踪人类示范，要明确各自接触角色

DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations

DONIC结合强化学习与成功轨迹回收，按每只手的接触角色设计物体中心奖励；BiHOI支持跨参考轨迹的通用双手追踪，结果仍需区分训练物体与留出物体。

论文：https://arxiv.org/pdf/2610.29997

项目页：https://ropedia.github.io/DONIC/

40先学物理可执行的语义，再流式重定向

SMR: Learning streaming motion retargeting for humanoids from human semantic

SMR由物理专家直接追踪人体语义目标，筛选可执行轨迹训练因果Transformer；G1实验比较语义保真和闭环表现，流式输入仅用当前及过去观测。

论文：https://arxiv.org/pdf/2610.29169

项目页：https://logosintelligencelab.github.io/SMR/

41低成本柔性阵列也能测剪切力

VecTile: A Low-Cost Flexible Tactile Array for 3-DoF Force Sensing

VecTile以硅胶凸点将三轴接触转为可区分的压力分布，配合压阻层完成读出；单点力标定、弯曲测试和插头插入实验支持其应用，误差需对应测试载荷。

论文：https://arxiv.org/pdf/2610.31472

项目页：https://jiannnh.github.io/VecTile/

42把三维场景流作为动作模型的第三模态

UniFlowWAM: A Unified World–Action Model with Visual Semantics and 4D Geometry

UniFlowWAM让视觉特征、三维流与动作在专家间联合交互，多种目标赋予各模态不同角色；仿真及双臂真机验证收益，部署时无需额外输入场景流。

论文：https://arxiv.org/pdf/2610.30991

项目页：https://kksham111.github.io/UniFlowWAM_ProjectPage/

43摔倒后如何起身，也能在推理时选择

Test-Time Motion Steering for Perceptive Humanoid Fall Recovery

RISE学习恢复动作词汇，再以视觉扩散规划器预测并引导身体运动；仿真支持朝向、速度和路径调整，G1实机演示补充了感知适应与起身方式选择。

论文：https://arxiv.org/pdf/2610.30061

项目页：https://osheraz.github.io/RISE/

44一段人类RGB示范，扩展到同类物体实例

Real2Sem2Real: 3D Semantic Augmentations for Robot Learning from a Single Human RGB Demonstration

Real2Sem2Real通过三维追踪、语义对齐与可行抓取扩增机器人示范；两台真机验证迁移，但仍需物体尺度测量，刚体与接触建模假设限制适用范围。

论文：https://arxiv.org/pdf/2610.28880

项目页：https://real2sem2real.github.io/

45从状态教师蒸馏腹腔镜视觉策略

Revisiting Distribution Matching for State-to-Vision Policy Distillation in Laparoscopic Robotic Manipulation

以可访问特权状态的策略作为教师，训练读取RGB与本体信息的学生，并分析分布匹配目标；证据来自手术操作仿真，不能直接外推临床或真实手术安全性。

论文：https://arxiv.org/pdf/2610.31010

代码：https://github.com/MingyuShengSMY/SurgiPD

46前沿模型直接操控LIBERO，要算成功也算成本

From Reasoning to Action: Benchmarking Frontier AI Models on LIBERO-10 Robot Manipulation Tasks

LLM-LIBERO让通用模型通过工具执行闭环操作，并比较推理强度、调用开销和任务表现；评测属于仿真工具接口设定，可用于定位失败而非代表所有机器人能力。

论文：https://arxiv.org/pdf/2610.30051

代码：https://github.com/IRVLUTD/LLM_LIBERO

项目页：https://irvlutd.github.io/LLM-LIBERO/

47用广义根轨迹同时生成多机器人路径

Extending Harmonic Function Methods via Root Locus for Coordinated Path Generation

通过保角映射与广义根轨迹，为平面可互换机器人构造互不相交的路径及目标分配；安全证书针对实际计算的势函数成立，适用条件和离线假设很关键。

论文：https://arxiv.org/pdf/2610.31625

代码：https://github.com/mfrascarelli/grl-certified-planning

项目页：https://mfrascarelli.github.io/grl-certified-planning/

48故障评测要拆开物理失效与接口错误

Alpha: Systematic Fault Injection and Robustness Evaluation for LLM-Driven Embodied Manipulation

Alpha对感知、执行、运动学与规划接口注入故障，以配对种子比较语言规划器和脚本；该单任务研究的总差异由输出损坏驱动，不支持笼统归因于推理能力。

论文：https://arxiv.org/pdf/2610.30197

代码：https://github.com/jonajoy142/alpha-embodied-robustness

49用轻量激光雷达表征应对拥挤导航

RangeNav: Task-Aligned LiDAR Range-Image Pretraining for Robot Navigation

RangeNav在合成距离图上联合学习重建、语义和俯视预测，再冻结编码器训练导航策略；仿真及四足真机展示环视感知价值，比较需对应传感器视场条件。

论文：https://arxiv.org/pdf/2610.29845

项目页：https://rangenav.github.io/

50用运动学词元压缩示范分段成本

KiRoSe: Using Kinematic Data for Efficient Segmentation of Robot Demonstrations

KiRoSe将重复运动模式编码为稀疏词元，再交给语言模型识别子任务；跨机器人与人类动作实验展示迁移潜力，也让进度分析不必总读多路视频。

论文：https://arxiv.org/pdf/2610.31801

项目页：https://berkeleyautomation.github.io/kirose/

51让规划失败反过来修正场景图数据

SCOPE: Scene-graph Construction Optimized by Plan Error-correction for Robot Long-term Planning Datasets

SCOPE结合场景多样性度量与规划反馈纠错，清理自动生成数据中的逻辑矛盾；仿真实验显示下游规划收益，重点是可执行性而非只追求文本丰富度。

论文：https://arxiv.org/pdf/2610.31518

项目页：https://scope-page.github.io/

52用骨架监督拆开运动与场景变化

Skeleton-Grounded Latent Action Model for Robot World Models

SkeLAM把潜在动作分成骨架约束的运动分量和视觉残差，再学习动作适配；DROID实验支持更好的运动预测，人到机器人的零样本转移主要以生成结果展示。

论文：https://arxiv.org/pdf/2610.31474

项目页：https://aopolin-lv.github.io/SkeLAM/

53先对齐边界，再衔接两段人形动作

Robot Motion Transition with Human Motion Prior

STAMP把机器人动作转入人体空间，学习选择位置、朝向与过渡时长，再用人体先验补间并映回机器人；跨形态仿真和真机演示检验动作串联。

论文：https://arxiv.org/pdf/2610.30817

项目页：https://childtoy.github.io/STAMP-project/

54语言纠正不只加特征，也要泛化与屏蔽

Flexible Language-Guided Representation Editing for Adaptive Robot Behavior

FLEX用语言反馈添加缺失特征、泛化过窄特征并屏蔽无关项；二维导航、Franka操作及真实用户研究检验交互修正，目前主要面向准静态刚体任务。

论文：https://arxiv.org/pdf/2610.28893

项目页：https://mit-clear-lab.github.io/FLEX/

55多机器人共享地图时保留估计一致性

Consistent Distributed Invariant EKF SLAM with Point and Object Features

Joint-CI在接收机器人的误差坐标中构造位姿与公共特征联合估计，再做部分状态协方差交集；仿真及真实数据派生实验检验精度与不确定性一致性。

论文：https://arxiv.org/pdf/2610.31221

代码：https://anonymous.4open.science/r/Joint-CI-for-Distributed-RI-EKF-SLAM-E14B/

56用RGB与深度建立可预测的共享状态

DepState-JEPA: Learning Depth-Grounded State Representation for Visual Perception and Manipulation

DepState-JEPA把视觉、深度和动作条件预测连接到共同潜在状态，并加入运动学一致性监督；表征及操作实验支持几何作用，应分别看预测质量与控制收益。

论文：https://arxiv.org/pdf/2610.29246

资源：https://github.com/ShawnLeung-SII/depstate-jepa

项目页：https://shawnleung-sii.github.io/depstate-jepa/

57同一策略切换动作风格，先看目标串扰

One Policy, Many Styles: Where to Aggregate Preferences for Steerable Humanoid Control?

比较奖励、优势、损失和梯度层面的偏好聚合，并用响应矩阵衡量风格选择；两种人形控制框架没有统一赢家，响应幅度与无关行为变化之间存在取舍。

论文：https://arxiv.org/pdf/2610.28925

代码：https://anonymous.4open.science/r/Pref-Align-RL-214C/

58合成图文未必补得上未明说的意图

CEMIR vs. FISER: a comparative study of reasoning-based and enrichment-based approaches to pragmatic intent recognition

CEMIR与FISER在HandMeThat v2上对比图文扩充与结构化推理；多重比较校正后未见显著差异，提示文本派生图像未必带来新的意图信息。

论文：https://arxiv.org/pdf/2610.32022

代码：https://github.com/CEMIR-Project/CEMIR

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment