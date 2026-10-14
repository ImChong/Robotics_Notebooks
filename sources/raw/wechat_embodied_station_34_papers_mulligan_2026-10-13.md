# 机器人越熟练，采集数据越要盯住失败｜附代码与项目入口

> 原始抓取（2026-10-14，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/mGe8FNbW8HXnr5TUb7PHGw

机器人越熟练，采集数据越要盯住失败｜附代码与项目入口
Original 具身智能小站 具身智能小站
 Oct 13, 2026, 6:00 AM

阅读路线

今天更新 34 篇具身智能新论文，覆盖强化学习与策略后训练、世界模型与仿真、灵巧操作、人形机器人与移动操作等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进31 篇扫读

先深读 Mulligan 的失败导向采集，再跟进 OpenWAM 的模块组合与 EigenDEXplore 的结构化探索；快速扫读按本期评分排序，其中 QUEEN 是跨领域方法参考，其实验对象为国际象棋。

先看这一篇

1把真机采集预算花在失败起点上

Mulligan: Performance-Guided Data Collection for Efficient On-Robot Learning

 

一句话判断：真机策略越熟练，均匀采集越容易重复已掌握的状态。Mulligan 优先重试失败起点，再补足未探索状态，并让价值函数利用失败轨迹。它把有限人工监督预算转向薄弱区域，适合关注部署后持续学习和数据成本的读者。

你可以带走的两点

• 先评测再选择采集起点：重试失败并补齐覆盖。最终轮插笔任务中，HiL-IDQL+Mulligan 为 38/50 成功，均匀 HG-DAgger 为 21/50；该比较包含采集与价值重排序的组合变化。

• 失败数据也有价值：策略用成功的人类控制片段训练，价值函数利用全部交互；作者在三项真机任务中进行了 2,550 次留出盲测，这一数字是全实验评测总量。

方法亮点：把从哪里开始采集作为部署后学习的可优化变量，并与策略训练、价值重排序分开检验。复现时同时控制采集预算、初始状态范围和人工介入条件。

论文：https://arxiv.org/pdf/2610.05882

代码：https://github.com/ankile/mulligan

项目页：https://mulligan.page/

另外 2 篇值得跟进

2在统一框架中比较世界与动作生成

OpenWAM: An Open Framework for Composable World-Action Models

 

看点：世界动作模型的比较常被骨干和训练数据差异混淆。OpenWAM 固定共同基础，比较四种生成顺序，并检验反事实数据能否支持动力学模块复用。

结果：作者在 LIBERO-Long 的 VTA 设置中，将成功率从 68.4% 提升至 97.8%；比较同时改变了机器人视频预训练与因果适配，不能把收益单独归因于生成顺序。

论文：https://arxiv.org/pdf/2610.07922

代码：https://github.com/OpenWAM/OpenWAM

项目页：https://openwam.stanford.edu

3用人类手部先验引导探索，保留关节控制

EigenDEXplore: Structured Exploration for Dexterous Manipulation with Human Priors

 

看点：低维动作空间容易限制灵巧手的表达。EigenDEXplore 将人类协调先验用于探索噪声，保留原生关节动作；跨手型实验有助于区分先验与降维的收益。

结果：四项真机工具操作的平均任务进度为 82% 对 33%；两种策略均训练 600 亿帧，每个物体轨迹组合测试 10 次，指标是到达的路标比例。

论文：https://arxiv.org/pdf/2610.07681

代码：https://github.com/hgupt3/eigendexplore

项目页：https://eigendexplore.github.io/

31 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04让编码智能体围绕瓶颈优化世界动作模型

WAMJET: A Harness for World Action Model Acceleration

WAMJET 用测量、修改和验证工具组织推理加速，跨模型、智能体与显卡测试；阅读时区分保持输出质量的工程优化与引入近似后的延迟收益。

论文：https://arxiv.org/pdf/2610.03797

代码：https://github.com/liulixinkerry/WAMJET

项目页：https://liulixinkerry.github.io/WAMJET/index.html

05从照片检索可操作的仿真资产

AffordCraft: Scalable Construction of Task-Ready Simulation Assets from Single Images

AffordCraft 根据图像与任务检索已有铰接资产，适配外观时保留部件和关节；大规模照片物理测试支持检索路线，能力依赖资产库的覆盖范围。

论文：https://arxiv.org/pdf/2610.06643

代码：https://github.com/AffordCraft/AffordCraft

项目页：https://affordcraft.github.io

06把物理决策留给智能体，执行交给后端

RobotUse: Allocating Computation, Context, and Decisions

RobotUse 让智能体视觉选择目标与位姿，后端完成几何规划和控制；子目标上下文与持续更新的操作手册支持反复修正，评测包含真机经验迁移。

论文：https://arxiv.org/pdf/2610.04929

代码：https://github.com/robotuse-team/RobotUse

项目页：https://robotuse-team.github.io/

07看视频写仿真，评测必须检查是否能执行

Video2World: Benchmarking Coding Agents for Interactive World Modeling from Embodied Videos

Video2World 将视频到仿真视为可执行软件任务，同时检查几何、动力学与任务完成；编码智能体评测提醒：视觉还原更好，不保证交互功能更可靠。

论文：https://arxiv.org/pdf/2610.04432

代码：https://github.com/AetherLabsAI/Video2World

项目页：https://aetherlabsai.github.io/Video2World

08不重训策略，按任务阶段调整速度

FreeSpeed: Training-Free Speed Control for Generative Robot Policies

FreeSpeed 对冻结策略的动作块重采样，并用方向不一致性限制步长缩放；仿真与真机展示在线调速，实际达到的倍率应与请求倍率分开看。

论文：https://arxiv.org/pdf/2610.05734

项目页：https://yuxuanhu9.github.io/FreeSpeed/

09主动转动视线，完成精细双臂操作

EyeRobot 2.0: Active Gaze for Precise Manipulation without Wrist Cameras

EyeRobot 2.0 联合主动注视、中心高分辨率视觉与注视坐标系动作表示，用单套双目视觉做精细操作；多任务真机测试比较了被动双目与腕部相机。

论文：https://arxiv.org/pdf/2610.03710

项目页：https://eyerobot2.github.io/

10移动操作，执行反思与离线学习双循环

MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation

MobiAgent 内循环动态组合技能并视觉反思，外循环整理部署轨迹与更新技能库；仿真和实体平台展示恢复与迭代收益，模型下载仍有访问条件。

论文：https://arxiv.org/pdf/2610.03476

代码：https://github.com/kaiknower/MobiAgent

项目页：https://kaiknower.github.io/mobiagent

11随策略能力变化，动态分配辅助控制

UniIntervene++: An Adaptive Intervention Agent for Efficient Real-World Reinforcement Learning

UniIntervene++ 将自主策略、轨迹纠正和代码辅助纳入统一选项框架，并周期探测自主能力；五项真机任务检验控制分配，人工介入率有专门统计口径。

论文：https://arxiv.org/pdf/2610.03620

代码：https://github.com/dannyyudong/An-Adaptive-Intervention-Agent-for-Efficient-Real-World-Reinforcement-Learning

12VLA 后训练，增强应该加在哪个模块

How (and How Not) to Use Data Augmentation in VLA Post-Training

在所测 PPO 后训练 设置中，只增强价值模块输入更有利于视觉泛化，增强策略输入反而破坏训练；结论来自两种 VLA 与仿真基准，需按条件复用。

论文：https://arxiv.org/pdf/2610.05994

项目页：https://bramgrooten.nl/vla-augm/

13让触觉失败诊断参与策略适配

PEARS: Physical-Prior-Guided Efficient Adaptation via Failure Reasoning and Diffusion Steering for Tactile Manipulation

PEARS 将视觉语言模型的失败诊断转为接触力边界，再用触觉条件噪声引导冻结策略；仿真和两项真机实验检验了在线适配效果。

论文：https://arxiv.org/pdf/2610.08784

项目页：https://song-kun.github.io/pears

14越野导航，在接管之前发出危险警报

CUSP: CUSUM-Governed Survival Hazard Alarms at the Perception Onset for Off-Road Navigation

CUSP 将人类感知危险的时刻与实际接管时刻分开标注，再累积生存风险信号；五个未见场地的评测统一误报率，便于比较报警有效性。

论文：https://arxiv.org/pdf/2610.07882

项目页：https://in-uk.github.io/cusp-project/

15流策略强化学习，减少采样器反向传播

QF3: Fast Flow RL with Filtered Q-Gradients

QF3 用单步流终点预测传递价值梯度，并在回放动作附近过滤更新；实验覆盖人形运动与仿真操作微调，人形策略还进行了真机零样本部署。

论文：https://arxiv.org/pdf/2610.08789

项目页：https://qf3-rl.github.io/

16用语言动作意图连接人类与机器人

EgoLAP: Learning from Egocentric Human Data through Language-Action Reasoning

EgoLAP 将第一视角人类轨迹转换为语言动作与运动推理，与机器人数据共同预训练；真机指标报告任务进度，阅读时应与完整任务成功率区分。

论文：https://arxiv.org/pdf/2610.08726

项目页：https://ego-lap.github.io/

17重型 VLA 做粗规划，轻量模型做修正

StairVLA: Stage-Aware Hierarchical Action Generation for Vision-Language-Action Models

StairVLA 用部分去噪轨迹衔接高层长时域生成与高频局部细化，摊薄昂贵推理；仿真与真机验证支持闭环修正，延迟指标是动作块摊销值。

论文：https://arxiv.org/pdf/2610.07756

项目页：https://dicomsky.github.io/projects/stairvla

18把稳定接触做成跨手型复用的底层能力

GOTT: Object-centric Dexterous Manipulation with a Reusable Cross-Embodiment Primitive

GOTT 将操作拆成靠近、建立接触与移动，以共享闭环原语连接物体轨迹和灵巧手；仿真、真机及未见手型实验检验稳定接触，仍需要可行的靠近规格。

论文：https://arxiv.org/pdf/2610.03861

项目页：https://dex-gott.github.io/

19把仿真运动先验蒸馏进真实视频模型

SimForcing: Distilling Simulation Motion Priors into Real-Domain Robot World Models

SimForcing 对齐潜空间的时间变化，将仿真运动先验传给真实域模型，并控制仿真条件的引导强度；实验区分视频预测质量与下游策略收益。

论文：https://arxiv.org/pdf/2610.06598

代码：https://github.com/Wang-Xiaodong1899/SimForcing

20异步重规划，延续同一条随机轨迹

Execution-Aligned Progressive Noise for Consistent Asynchronous Replanning in Generative Robot Policies

EAPN 按实际执行位移对齐跨动作块噪声，并建模块内时间相关性；多项仿真与真机操作检验行为连续性，高低延迟实验采用各自训练协议。

论文：https://arxiv.org/pdf/2610.06090

项目页：https://embodied.magiclab.top/works/eapn/index.html

21通过采样观测器补偿模型与现实差异

Tackling Sim-to-Real Mismatch Through Sampling-Based Disturbance Observers: From Analytical Models to Learned World Models

SDOB 通过状态预测或代价查询估计扰动，使传统观测器适用于仿真器和学习模型；仿真与真机验证补偿收益，前提包括可观测性及缓变扰动。

论文：https://arxiv.org/pdf/2610.04896

项目页：https://sampling-based-dob.github.io/

22让人形机器人连续搬运多个物体

Humanoid Horizon: Extending Task Horizon in Whole-Body Loco-Manipulation via Parallel Training, Dynamic Starting, and Reward Gating

Humanoid Horizon 通过并行阶段训练、动态初始状态和奖励门控缓解长序列遗忘；仿真搬运验证阶段衔接，但增加物体后性能仍会下降。

论文：https://arxiv.org/pdf/2610.08320

项目页：https://haozhuo-zhang.github.io/Humanoid-Horizon-project-page/

23让动作模仿与可执行数据相互改进

InterMimicGen: Scaling Humanoid Loco-Manipulation through Self-Evolving Motion Imitation

InterMimicGen 汇集人类物体交互并重定向到人形机器人；跟踪器筛选小幅改动后仍可完成任务的动作，逐轮扩充数据，实验包含仿真与真机迁移。

论文：https://arxiv.org/pdf/2610.06850

项目页：https://sirui-xu.github.io/InterMimicGen

24小块视觉扰动，暴露共享编码器脆弱性

TAPDreamer: Transferable Adversarial Patches for World Action Models

TAPDreamer 仅利用公开视觉编码器构造固定局部扰动，在所测世界动作模型间迁移；闭环仿真揭示编码器风险，不能直接外推为实体攻击结论。

论文：https://arxiv.org/pdf/2610.06814

项目页：https://tapdreamer.github.io

25用流反演学习条件噪声分布

Learning Conditional Source Distribution via Flow Reversal for Temporal Flow Matching

CNP-Flow 通过流反演监督条件噪声预测器，再微调原有流模型；视频生成与七自由度机械臂规划实验表明，改进噪声起点也能降低采样开销。

论文：https://arxiv.org/pdf/2610.05349

项目页：https://embodiedai-ntu.github.io/cnpflow/index.html

26从双人舞视频学习顺应接触的人形动作

CoDance: Learning Reactive and Compliant Human-Humanoid Interaction from Video

CoDance 将双人舞重定向并加入多连杆顺应增强，让机器人同时跟随伙伴和响应接触力；仿真消融与实体共舞展示了持续双手接触能力。

论文：https://arxiv.org/pdf/2610.05324

项目页：https://generalroboticslab.com/CoDance

27长任务记忆，只保存当前上下文解释不了的部分

Test-Time Training as Residual Memory for Robot Policies

TTT-RM 用历史重建残差监督快速权重，为有限记忆库补充难以恢复的信息；记忆任务与真机收纳检验长期执行，关注点是互补记忆而非无限历史。

论文：https://arxiv.org/pdf/2610.04701

项目页：https://hatchetproject.github.io/tttrm/

28世界模型同时预测颜色与深度

DepthWorld: 3D World Model for Robot Manipulation

DepthWorld 先联合校准机器人运动学与多视角外参，再以潜变量空间拼接预测颜色和深度；同训练预算下优于仅颜色基线，重点是几何一致性。

论文：https://arxiv.org/pdf/2610.08780

项目页：https://www.jaibardhan.com/depthworld

29策略变强，验证器也要一起更新

VeriFine: Scaling Verification for Self-Improvement in Embodied Reasoning

VeriFine 让策略、课程与评分器共同演进；优化停滞时针对信息量高的失败征求人类指导，校准评判规则，研究重点是具身推理的训练反馈。

论文：https://arxiv.org/pdf/2610.08761

项目页：https://veri-fine.github.io/

30同一策略，让多机器人形成互补分工

CoRE: Learning Collaboration-Role Experts for Decentralized Collaborative Manipulation with One Policy

CoRE 用局部视觉和本体状态路由协作专家，无需身份标签或机器人间消息；仿真与实体协作实验关注互补行为，以及伙伴延迟下的稳健性。

论文：https://arxiv.org/pdf/2610.07752

项目页：https://aus.bot/research/core/

31让监督更关注动作引起的局部变化

KineWorld: Action-Induced Transport Fields for Embodied World Modeling

KineWorld 将机器人运动学映射为相机对齐的输运场，再对未来视频监督加权；双臂仿真数据评测强调动作后果，均匀外观损失是关键比较对象。

论文：https://arxiv.org/pdf/2610.06349

项目页：https://modaxiansheng.github.io/KineWorld/

32让一次完整技能成为世界模型的一步

Flow Policies as Actions of Skill-Level World Models: Learned and Symbolic Abstractions for Long-Horizon Planning

这项研究比较流策略噪声、学习编码和符号标签四种技能接口；积木仿真发现，长序列性能还受搜索限制，符号方案的优势部分来自动作适用性规则。

论文：https://arxiv.org/pdf/2610.04767

项目页：https://andreumatoses.github.io/research/flow-skill-wm

33先学人手与物体关系，再适配机器人

PatternDex: Learning Interaction Patterns to Guide Reinforcement Learning of Bimanual Dexterous Manipulation of Articulated Objects

PatternDex 从人类示范学习交互模式，再预测目标手型的腕部运动和接触点以指导强化学习；多手型仿真与开微波炉真机验证支持模式复用。

论文：https://arxiv.org/pdf/2610.04765

项目页：https://patterndex.github.io/PatternDex/

34让棋类专家既会下棋，也能解释决策

Language Models that Play Chess and Explain Their Moves

QUEEN 将棋类专家编码器连接语言模型，并迭代蒸馏落子解释。这是国际象棋研究，提供专家能力与语言表达结合的参考；论文未验证机器人部署。

论文：https://arxiv.org/pdf/2610.03695

代码：https://github.com/queen-project/queen

项目页：https://queen-project.github.io/

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment