# 机器人做错了，奖励为什么还在增加？｜附代码与项目入口

> 原始抓取（2026-10-30，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/dxeEQnbdviUi4D4LuTfEhw

机器人做错了，奖励为什么还在增加？｜附代码与项目入口
具身智能小站 具身智能小站
 Oct 30, 2026, 6:00 AM

阅读路线

今天更新 28 篇具身智能新论文，覆盖操作学习与数据、世界模型与迁移、运动控制与感知、多机器人协作等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进25 篇扫读

先读 Event2Reward 的进度监督与真机训练口径，再跟进 TACOS 的接触监督和 Kairos 的规划验证；其余按优先级快速浏览，结合资源入口选择深读方向。

先看这一篇

1让奖励分清进步、停滞与倒退

Event2Reward: Learning Robotic Rewards from Progress Relations

 

一句话判断：机器人探索时会停滞甚至倒退，把示范时间当作进度容易奖励错误动作。Event2Reward 用成功示范锚定进度尺度，再用失败与次优轨迹中的局部关系校正变化方向，为强化学习提供更可信的稠密奖励。对需要少量现场数据适配的操作任务，这套监督设计值得细读。

你可以带走的两点

• 监督不只看时间：用停滞、倒退与恢复事件提供局部进度关系，再与绝对进度锚点共同训练；奖励来自进度势函数的变化。

• 看同预算真机比较：三项精密插入任务中，30 分钟训练后的平均成功率为 60.0%，Robometer 为 37.8%；每任务、每条件三次训练，每个检查点评估二十回合。

方法亮点：新增的是对停滞、倒退和恢复事件的局部进度关系监督：无需为次优轨迹逐帧指定绝对进度，也能约束奖励变化方向。

论文：https://arxiv.org/pdf/2610.39212

代码：https://github.com/niejunnan25/Event2Reward

项目页：https://niejunnan25.github.io/Event2Reward/

另外 2 篇值得跟进

2先判断脚下会不会滑，再决定落脚点

TACOS: Terrain-Aware COntact Supervision for Visual Footstep Planning

 

看点：几何平整不代表地面可靠。TACOS 从状态估计因子图的接触残差构造不稳定性监督，再投影到此前图像，学习视觉落脚预测；真机森林路径展示了它对泥地、植被等非几何风险的处理。

结果：监督来源可比较：硬评分利用后验信息，软评分可因果获得；两者都由接触残差生成，而非凭直觉选择标签。

论文：https://arxiv.org/pdf/2610.37806

代码：https://github.com/norlab-ulaval/TACOS

3让规划器按任务语义选择搜索引擎

Kairos: Semantics-Aware Orchestration of Planning Engines across Classical, Numeric, Temporal and Hybrid PDDL

 

看点：运行时规划既要覆盖时序、数值与连续变化，又要解释失败意味着什么。Kairos 按 PDDL 语义调度异构引擎，并独立验证候选计划，为机器人系统提供更明确的规划接口。

结果：失败语义进入接口：区分有证明的不可解与预算内未找到计划，避免上层把搜索超时误判成任务无解。

论文：https://arxiv.org/pdf/2610.39454

代码：https://github.com/PlanSys2/kairos

25 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04把探索预算用在影响任务的模型误差上

Explore Where It Matters: Targeted Exploration for Efficient Rover Adaptation

TERRA 用奖励加权误差适配世界模型，再预测哪些探索能暴露任务相关缺陷。真机松软坡面试验展示少量在线尝试的适配价值，结论限于所测车辆与地形。

论文：https://arxiv.org/pdf/2610.38657

代码：https://github.com/CLeARoboticsLab/terra

项目页：https://terra-explore.github.io/

05封闭空间里，用机载信号估计连续体形状

Controlling Continuum Robot Shape in Enclosed Space with Embedded Sensing using IMU and Actuator Feedback

融合 IMU 与执行器反馈，用神经网络重建连续体机器人形状并闭环控制。研究系统比较表征、网络、目标与传感输入，为无法外部观测的空间提供设计依据。

论文：https://arxiv.org/pdf/2610.39055

代码：https://github.com/sunroboticslab/Embedded_Shape_Sensing_Tmech2027

06按任务表现分配 VLA 的量化精度

Quantize by Doing: Task-Guided Mixed-Precision Quantization of Vision-Language-Action Models

QbD-VLA 用短程闭环成功率选择混合精度，考虑模块之间的相互影响。仿真配置迁移到真机时，X-VLA 与 π₀.₅ 表现不同，说明量化收益需要按模型验证。

论文：https://arxiv.org/pdf/2610.38661

代码：https://github.com/jiuyixu25/QbD-VLA

07筛掉示范之前，先检验随机基线

CuraEval: Evaluating Demonstration Curation on Mixed-Skill Robot Data

CuraEval 在混合技能数据上控制子集大小与训练预算，比较筛选、随机抽样和全量训练。调整评测协议会改变收益方向，提醒数据清洗结论必须附带统计口径。

论文：https://arxiv.org/pdf/2610.38221

代码：https://github.com/jiuyixu25/CuraEval

08通信受限时，让多机器人持续合图

SKiD-SLAM: Resource-Aware Distributed Multi-Robot LiDAR SLAM for Planetary Missions with Field Deployments at Analog Sites

SKiD-SLAM 按行星任务的通信与计算预算组合地点识别、配准和分布式融合模块，并在熔岩洞与模拟行星地表部署；现场证据来自各自同构的机器人团队。

论文：https://arxiv.org/pdf/2610.38141

代码：https://github.com/sparolab/SKiD-SLAM

项目页：https://sparolab.github.io/research/skid_slam/

09实时预测障碍环境中的机械臂可达空间

HERON: Real-Time Obstacle-Aware Reachability Maps for Mobile Manipulators

HERON 用三维网络预测含障碍的可达性地图，并在嵌入式硬件上量化部署。它学习的是逆运动学生成的地图近似，边界区域误差和量化损失仍需结合用途评估。

论文：https://arxiv.org/pdf/2610.38233

代码：https://github.com/Lab-CORO/Heron

10长任务出错后，只修复失效的依赖

LOGIV: Graph-Verified Execution with Causal Local Repair for Long-Horizon Robot Tasks

把动作完成与条件仍然有效分开记录，沿任务图定位需要恢复的支持条件。LOGIV 在已有策略之上加入验证和局部修复；真机提升仍受底层接触精度约束。

论文：https://arxiv.org/pdf/2610.40016

代码：https://github.com/Enqiao-Lu/LOGIV

项目页：https://enqiao-lu.github.io/logiv-project/

11机器人团队需要交流多少信息才够用

Minimum Sufficient Communication for Independent Behaviour Tree Robot Teams

在固定独立行为树与声明的策略族中，比较消息内容、发送频率及成功率容差。研究区分充分上界与真正最小值，其低通信结论依赖具体负载和候选策略范围。

论文：https://arxiv.org/pdf/2610.38176

代码：https://github.com/sherryjiang-hub/Multi-robot-collaboration-

12生成多臂示范时，把碰撞与时序一起安排

ScheduleGen: Demonstration Generation via Skill Scheduling for Multi-Arm Imitation Learning

ScheduleGen 联合技能迁移、连接运动规划和多臂调度，减少忽略协同约束带来的无效示范。研究覆盖仿真与真机迁移，数据生成成功率与策略成功率需要分开看。

论文：https://arxiv.org/pdf/2610.39429

项目页：https://schedulegen.github.io/

13控制器换了，策略也要读懂动作的新含义

Cross-Controller Transfer via In-Context Behavior Adaptation

以动作指令及实际运动组成交互上下文，让统一策略适配不同控制模式与增益。跨控制器迁移不仅是数值换算，也可能需要调整接触任务中的操作策略。

论文：https://arxiv.org/pdf/2610.39280

资源：https://github.com/JonathanYang0127/icl_cross_controller

项目页：https://controller-transfer.github.io/

14把语言模型的脱困经验教给决策树

Teaching a Tree to Recover: Online Distillation of an LLM Supervisor with Counterfactual Replay

利用反事实回放判断恢复动作是否必要或有害，在线更新决策树。测试时无需调用语言模型即可处理熟悉类型的异常；当前主要证据来自预设导航布局的仿真。

论文：https://arxiv.org/pdf/2610.39775

代码：https://github.com/kristianceder/Counterfactual-Tree-Teaching

15选择动作模型时，把预算和证据一起算

RoboSeed: Multidimensional Evaluation of Vision-Language-Action and World-Action Models: Capability, Resource Affordability, and Research Evidence

RoboSeed 将能力、训练和部署负担、研究价值及复现证据分维度评估。对协议不同或测量缺失的模型保留空缺，不输出缺乏可比性的总排名，适合选型时参考。

论文：https://arxiv.org/pdf/2610.40026

代码：https://github.com/PrimitiveX/RoboSeed

16把双手示范采集接到训练数据流水线

UMI-Live for In-The-Wild Bimanual Robotics Data Collection

UMI-Live 集成相机、激光惯性定位和异步重建，连接手持采集、数据处理与策略训练。双手搬运验证每个模型仅十次测试，应把它作为数据管线证据理解。

论文：https://arxiv.org/pdf/2610.39385

代码：https://github.com/AfterQuery-Research/umi-live

项目页：https://umi-live.github.io/

17让语言导航模型真正运行在无人机上

OpenFlyV2: An onboard vision-language-action model for aerial perception and navigation

OpenFlyV2 共享视觉语言表征，用运动记忆与动作先验降低机载推理负担。论文报告仿真到真机导航，四地汇总结果包含训练重建场景和未见地点，须区分两者。

论文：https://arxiv.org/pdf/2610.39287

代码：https://github.com/Ideal-ljl/AeroBrain

18让灵巧手接住形状和弹性不同的飞来物

Catch It All!: Generalizable Dynamic Catching with a Dexterous Hand

Catch It All! 用抓握奖励和安全课程训练动态接物，再零样本部署到 G1。单个自视角相机处理未见物体，重点在视觉噪声、接触变化与硬件约束的共同设计。

论文：https://arxiv.org/pdf/2610.39044

资源：https://github.com/arth-shukla/catch-it-all

项目页：https://arth-shukla.github.io/catch-it-all/

19同时预测未来画面与显式三维运动

WorldFlow: Joint Video and 3D Motion Prediction via Large-Scale Pretraining

WorldFlow 把三维位移与视频联合建模，让视觉演化约束几何动态。评测关注轨迹误差与跨数据来源泛化，不能直接把预测精度改善解释为闭环操作成功率提升。

论文：https://arxiv.org/pdf/2610.39278

资源：https://github.com/worldflow-rbs/worldflow

项目页：https://worldflow-rbs.github.io/

20按信息收益重分配多机器人探索任务

EARS-Info: Information-Aware Event-Driven Reallocation for Multi-Robot Exploration

EARS-Info 在城市地图栅格上结合边际感知收益、可行迁移和返程约束。大规模仿真比较有参考价值，但开发中使用过评测城市，不能据此宣称未知城市泛化。

论文：https://arxiv.org/pdf/2610.38040

代码：https://github.com/vtavakkoli/SafeSwarm-CityTwin

21单卡训练小型人形机器人的动作跟踪器

SONIC-K1: Porting a Universal Humanoid Motion Tracker to a Small Humanoid on a Single GPU

SONIC-K1 将全身跟踪迁移到 Booster K1，提供策略与评测线索。仿真覆盖较广，但真机踏步和转身仍不稳定，不能把动作片段通过率等同硬件可靠性。

论文：https://arxiv.org/pdf/2610.39327

代码：https://github.com/Kyk0/sonic-k1

项目页：https://kyry.me/sonic-k1

22用几何骨架减少抓取标签的人力成本

Skeleton-Guided Geometry-Aware Auto-Labeling for Robotic Grasping: From 4-DoF Learning to 6-DoF Execution

通过直骨架产生两指抓取标签，训练轻量抓取网络，并扩展到点云上的六自由度标注。数据集准确率、抓取精度与真机成功率衡量不同能力，阅读时需分别比较。

论文：https://arxiv.org/pdf/2610.38443

代码：https://github.com/Farbod82/SkelGGCNN

23为不同机械臂和灵巧手提供统一实验接口

DexSuite: An All-in-One Framework for Dexterous Manipulation Across Arms, Hands, and Tasks

DexSuite 基于 Genesis 整合任务、机器人组合、遥操作与模拟触觉，并提供示范和基准。它强调仿真实验流程复用，工具使用和长序列操作仍是现有策略难点。

论文：https://arxiv.org/pdf/2610.37799

项目页：https://dexsuiteorg.github.io/

24把未来生成与稠密感知放进同一模型

Seeing and Imagining as One: Unified Video Generation and 4D Perception for Embodied World Modeling

GenSight 统一视频、深度、机器人分割及相对光流表示，支持未来生成和观测视频感知。它提供跨模态一致性的建模思路，价值主要体现在世界模型与感知层面。

论文：https://arxiv.org/pdf/2610.39232

项目页：https://gensight-rbs.github.io/

25反复失败后，为四足机械臂补齐控制能力

ELEVATE: Agentic Controller Acquisition from Task Failure for Quadrupedal Manipulation

ELEVATE 将持续任务失败作为寻找缺失全身控制能力的信号，再获取新控制器并更新任务程序。方法面向超出初始工作空间的操作，真机迁移证据限定在所测任务和平台。

论文：https://arxiv.org/pdf/2610.38388

项目页：https://agentic-quad-elevate.github.io/public/

26一句“没关系”，也需要正确的动作时机

When "No Worries" Went Wrong: HRI Lessons for Foundation-Model Robots

从长期人机交互工程案例提炼注意、知识、承诺、具身时序与动作权限等设计经验。TRACE 通过改变交互前提追踪决策边界，适合审查系统架构中的责任断点。

论文：https://arxiv.org/pdf/2610.38400

项目页：https://elikh.github.io/social-hri-framework/

27VLA 的推理技巧能否跨家务任务成立

What Actually Matters? A Paired Evaluation of Inference Settings for π₀.₅ on BEHAVIOR-1K

在 BEHAVIOR-1K 全部百项任务上配对比较动作执行长度、播放速度和去噪步数。部分常用技巧不能稳定迁移，观察结果时要把策略状态与时间预算一起考虑。

论文：https://arxiv.org/pdf/2610.37917

项目页：https://rubenhammele.github.io/behavior2026

28把模仿学习带入泳池与真实海域

Embodied at Sea: Imitation Learning for Multi-Task Underwater Manipulation

研究比较水下操作中的观测设计和策略架构，并报告海域作业案例。泳池最优模型的总体成功口径包含部分成功，不能与真实海域的任务成功率直接合并。

论文：https://arxiv.org/pdf/2610.38199

项目页：https://lair-research-nyu.github.io/Embodied-at-Sea/

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment