# 机器人动作为什么总在换块时卡顿？｜附代码与项目入口

> 原始抓取（2026-10-14，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/pxBq6lFyzH05LGiBkkR9mQ

机器人动作为什么总在换块时卡顿？｜附代码与项目入口
Original 具身智能小站 具身智能小站
 Oct 14, 2026, 6:00 AM

阅读路线

今天更新 41 篇具身智能新论文，覆盖VLA 与动作生成、世界模型与记忆、导航及运动控制、多机器人协作与评测等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进38 篇扫读

先读 StreamingWAM 的流式动作更新，再跟进 FreeBar 的异步边界对齐与 RoboCombine 的技能交接评测；其余论文按研究方向扫读。本稿依据可读取的官方摘要和作者仓库整理，全文实验与排序仍待复核。

先看这一篇

1让世界动作模型持续更新下一步

StreamingWAM: Streaming World Action Modeling via Chunk-Wise Staircase Flow Matching

 

一句话判断：世界动作模型部署时，单次预测快还不够，相邻动作块接不上仍会让机器人顿挫。StreamingWAM 将持续更新的动作缓冲、阶梯噪声与历史记忆结合，提供同时审视响应延迟和动作连续性的设计思路，适合研究闭环控制的团队优先阅读。

你可以带走的两点

• 把整块生成改为滚动缓冲与逐次细化，让新观察到达后继续修正未来动作；复现时区分首次暖启动与后续稳态更新，不能把两者延迟混为一谈。

• 评估同时检查跨块连续性、任务成功与重规划延迟。作者报告覆盖 LIBERO、LIBERO-Plus、RoboTwin 与真机；当前可读取的仓库实现范围主要是 LIBERO-Plus。

方法亮点：新增之处是把分块噪声层级、前缀延续和历史条件放进同一流式更新流程，同时处理响应延迟与动作衔接。作者报告了仿真及真机评估，本稿尚未完成实验表格核验，因此不引用具体性能数字。

论文：https://arxiv.org/pdf/2610.15585

代码：https://github.com/renshaojie233/StreamingWAM

项目页：https://renshaojie233.github.io/StreamingWAM/

另外 2 篇值得跟进

2异步 VLA 的动作块也要接得上

Barrier-Constrained Guidance For Real-time Asynchronous VLA Execution

 

看点：异步执行时，新动作块需要接上机器人正在执行的旧动作。FreeBar 用推理期一致性约束对齐边界，保留远期调整空间，适合已有流匹配策略的部署团队评估。

结果：可带走的是分区对齐：已承诺前缀严格约束，近未来柔性衔接，远未来保留自由度。作者在仿真与真机报告改善；复现入口需另备检查点和场景资产。

论文：https://arxiv.org/pdf/2610.14997

代码：https://github.com/NamelessCrew/FreeBar

3单个技能会做，串起来为什么失败

Exploring the Limits of Combinatorial Generalization in Embodied Manipulation

 

看点：技能单独成功，并不意味着串联后仍能成功。RoboCombine 把失败定位到子任务交接状态，为长程操作提供可诊断的评测方式，适合做组合泛化与策略评估的读者。

结果：将有序成功率与子任务交接单独检查，区分理想触发和预测触发。作者提出扩展原子起始状态的 PRTI；当前仓库主要提供评测运行时，需自备策略和仿真资产。

论文：https://arxiv.org/pdf/2610.15425

资源：https://github.com/AutoLab-SAI-SJTU/RoboCombine

38 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04机器人状态空间的近邻搜索要尊重几何

nanoflann: A Header-Only KD-Tree Library for Exact Nearest-Neighbor Search on Compile-Time Product Manifolds

nanoflann 将旋转、位姿等乘积流形的距离与剪枝规则纳入精确近邻搜索，避免对原始坐标直接使用欧氏距离。另提供动态索引，乘积流形接口需要 C++17。

论文：https://arxiv.org/pdf/2610.15215

代码：https://github.com/jlblancoc/nanoflann

05验证器有误差，采样目标还能保持吗

Imperfect Verifiers, Exact Targets: Inference-Time Scaling for Continuous Generation

CS-VGB 通过配对去噪与再加噪进行回溯搜索，在给定正则条件下保持目标奖励倾斜分布。除图像生成外还测试多机器人规划；理论保证不能脱离其转移与条件假设。

论文：https://arxiv.org/pdf/2610.15281

代码：https://github.com/KraitGit/CS_VGB

06长时视频预测要保留什么记忆

Remembering What Matters for Long-Horizon Robotic World Models

ReCAP 冻结世界模型，只保留初始场景锚点与近期完整视觉动作块，移除陈旧生成历史。它改善的是给定动作下的视频预测，记忆冲突解释仍可能与上下文长度效应混合。

论文：https://arxiv.org/pdf/2610.14583

资源：https://huggingface.co/easylearning/recap-robot-models

项目页：https://alexander-wu.github.io/recap-project-page/

07把三维动态知识留在训练阶段

FlowVLA: Enabling World Dynamics Understanding through Pluggable 3D Flow Learning

FlowVLA 只在训练时加入点级与末端执行器级三维流监督，将场景运动与控制信号对齐；推理移除辅助头，适合关注二维 VLA 动态理解与部署成本的读者。

论文：https://arxiv.org/pdf/2610.16160

项目页：https://mmai-laboratory.github.io/flowvla/

08从动作历史出发，缩短生成路径

Δ Flow Matching for Robot Foundation Models

Δ-flow 用带轻微噪声的近期历史替代固定高斯起点，让动作与视觉生成转为残差预测。论文在 VLA 与 WAM 上研究单步推理，免重训替换仅限所述预训练 WAM 设置。

论文：https://arxiv.org/pdf/2610.15794

项目页：https://zuo-kuangji.github.io/delta-flow/

09移动操作示范也要教机器人看哪里

Look Before You Move: Agent-Interface-Aligned Supervision for Open-World Mobile Manipulation

AIDA 让教师通过部署时相同的技能接口采集示范，显式覆盖保持目标可见与丢失后恢复。仿真训练后进行真机迁移，提示移动操作的数据质量也取决于主动视觉示范。

论文：https://arxiv.org/pdf/2610.15572

项目页：https://connerqiu.github.io/look-before-you-move/

10把监督分配给精密操作的关键时刻

PRISM: Precision Forcing for Generalist Vision-Language-Action Policies

PRISM 针对流匹配监督被平均分摊的问题，按精度需求重加权并把需求写入条件表示。实验涉及连接器与拧螺丝；启发是让训练更关注容差敏感时刻，而非仅扩大模型。

论文：https://arxiv.org/pdf/2610.14704

项目页：https://seen-e.github.io/PRISM/

11让示范中的柔顺性可以被表达和迁移

Residual Compliant Policy: Learning Compliant Policies from Few Demonstrations

方法在遥操作中显式选择柔顺模式，结合轨迹重定向与物理优化，再由残差强化学习调整导纳控制参考。实验集中于 FR3 的翻转与推动任务，少示范结论需保留任务范围。

论文：https://arxiv.org/pdf/2610.14035

项目页：https://residual-compliant-policy.github.io/

12用序贯检验减少策略排名误判

AnyRank: Reliable Anytime Policy Ranking via Sequential Testing

AnyRank 将真机试验优先分配给难区分的策略，并在任意停止时刻控制错误排名概率。可利用仿真或世界模型给出的不完美先验，适合审视小样本成功率排名的可信度。

论文：https://arxiv.org/pdf/2610.14603

代码：https://github.com/irom-princeton/multi_hypothesis_test

项目页：https://abadithela.github.io/anyrank-page/

13多机器人遥操作如何共享态势

Design and Evaluation of a Collaborative Teleoperation Interface for Multi-Operator Multi-Robot Systems in Challenging Environments

通过核电站数字孪生中的双机器人任务，比较共处共享、远程共享和共处独立界面。研究把任务效率、认知负荷与态势感知一起衡量；用户实验不能替代真实灾害现场验证。

论文：https://arxiv.org/pdf/2610.15001

资源：https://github.com/kenanalperen/NPP-Digital-Twin

14从小队到十一人足球协作

B-Human '26 - Scalable Team Play up to 11 vs. 11

B-Human 总结从标准平台向更大型人形机器人的迁移，并介绍支持不同队伍规模的行为架构。十一人对抗来自展示赛经验，阅读时应与常规比赛和受控基准成绩区分。

论文：https://arxiv.org/pdf/2610.14733

代码：https://github.com/bhuman/MachineLearning/tree/master/IsaacGymRL

15视频看懂了，物理代码能写对吗

PhySimCode: A Benchmark and Evaluation Method for Physics Video to Code Generation

PhySimCode 要求模型从原始视频恢复物理规律、参数并生成可运行仿真代码。评测分开考察规律、仿真等价性、可执行性与参数恢复，避免将会描述物理等同于会重建动力学。

论文：https://arxiv.org/pdf/2610.15722

代码：https://github.com/SourajaKundu/PhySimCode

项目页：https://sourajakundu.github.io/PhySimCode/

16让视频专家只传递动作所需信息

BridgeWAM: Bridging Video and Action Experts via Latent Bridge Queries

BridgeWAM 用潜在桥接查询汇聚视觉、语言和本体信息，动作专家按需读取，避免逐层密集耦合。实验包含泛化与延迟分析；轻量动作头不代表整个模型同样轻量。

论文：https://arxiv.org/pdf/2610.14746

代码：https://github.com/jxfang04/BridgeWAM-code

17把沿途观察积累成导航证据

GG-Nav: Goal-oriented Grounding Chain-of-Thought Elicits Reasoning in Navigation Foundation Models

GG-Nav 通过结构化推理先把中间帧转为目标证据，再结合当前观察决定行动或追问。实验覆盖目标导航基准与 Go2 真机，适合关注漏看目标及停止位置错误的团队。

论文：https://arxiv.org/pdf/2610.14541

代码：https://github.com/DuangZhu/GG_Nav

项目页：https://duangzhu.github.io/GG-Nav-Project-Page/

18把多路历史压缩成增量策略记忆

Memory Without Curation: Learning Multi-Stream Latent Memory for Robot Policies

IMPRINT 联合记忆观察、动作与本体信息，先训练历史重建和进度估计，再与策略联合微调。推理增量更新潜在记忆，针对长历史延迟与视觉捷径学习同时设计。

论文：https://arxiv.org/pdf/2610.14406

项目页：https://yilin-wu98.github.io/imprint_website/

19接触操作的计算预算该花在哪里

Do My Samples Help? Compute Budget Allocation for Real-World Sampling-Based MPC in Contact-Rich Manipulation

BAMPC 在同一控制栈下比较动作探索、动力学与状态不确定性、重规划频率及时间对齐。接触操作实验提示预算分配取决于主要仿真迁移误差，动力学随机化并非总最优。

论文：https://arxiv.org/pdf/2610.14064

项目页：https://bampc-framework.github.io/

20把技能绑定到目标区域，再跨任务组合

RAIN: Robotic Task Generalization with Region-Aware Interaction Networks

RAIN 用动作类型与目标区域掩码表示子任务，结合跨视角编码、轨迹预测和完成判断。参考轨迹重定向用于合成新目标交互，实验覆盖 LIBERO-Analogy 与真机桌面任务。

论文：https://arxiv.org/pdf/2610.14039

项目页：https://seung-hun-lee.github.io/projects/RAIN/

21用潜在预测循环生成动作块

Vision-Predictor-Action: Towards Efficient Embodied AI via Neuro-Symbolic JEPA Predictors

VPA 组合离散技能选择、JEPA 潜在预测与流匹配动作块生成，尝试降低串行推理成本。当前证据是 LIBERO-10 的单次训练概念验证，速度与成功率来自不同相机配置。

论文：https://arxiv.org/pdf/2610.15556

代码：https://github.com/ylee-sling/Vision-Predictor-Action

22分开探索与利用，再组合生成策略

CFG-Policy: Unified Diffusion Policy Learning Framework via Classifier-free Guidance

CFG-Policy 将探索与利用拆成两条生成分支，在采样时组合分数或速度场，以减少强化学习微调中的行为多样性损失。在线与离线转在线设置冻结的分支不同。

论文：https://arxiv.org/pdf/2610.14936

项目页：https://zhy-liu.github.io/cfg-policy-page

23用测试时优化缩小人形命令执行差距

Test-Time Compute for Humanoid Behavior Foundation Models

TTCO 在闭环仿真中搜索更合适的输入命令，让既有行为基础模型更接近期望运动；还可生成模型专属动作标签。它无需重训基础模型，但依赖模拟与采样计算。

论文：https://arxiv.org/pdf/2610.14896

项目页：https://bfmttc.github.io/

24四足视觉记忆随身体状态变化

Beyond Fixed Visual History: Learning Motion-Adaptive Temporal Receptive Fields for Complex-Terrain Quadruped Locomotion

TempoLoco 从长深度历史中学习互补时间视图，再根据本体状态与地形动态融合。实验涵盖复杂地形仿真及带固定机械臂的 B2 真机，重点是让视觉提前量适应运动能力。

论文：https://arxiv.org/pdf/2610.14753

项目页：https://capability-aware.github.io/

25四足负载导航也要估计物体外形

Load-Aware Quadruped Navigation with Online Contact-based Geometry Estimation

LOADNav 学习随负载几何调整路径与间距，CoGE 则从偶发接触在线推断未知负载形状。方法针对固定足迹膨胀过保守的问题，几何估计依赖接触信息逐步更新。

论文：https://arxiv.org/pdf/2610.14596

项目页：https://qiii04.github.io/LOADNav/

26多机器人行动前先预测队友响应

Before We Act: Learning Action-Conditioned Team-Response Beliefs for Decentralized Multi-Robot Manipulation

CARE 从局部历史学习团队响应信念，将候选动作的直接收益与队友适应带来的收益分开建模。它选择性修正名义策略，重点是识别队友响应会改变最佳动作的决策。

论文：https://arxiv.org/pdf/2610.14551

项目页：https://jeong-zju.github.io/care/

27用一个模型连接不同传感模态

TartanMatch: Towards Universal Dense Correspondence Across Modalities

TartanMatch 在共享网络中适配 RGB、事件、热成像、深度与投影激光雷达之间的稠密匹配。混合真实与合成数据训练，研究跨模态联合学习对零样本对应估计的帮助。

论文：https://arxiv.org/pdf/2610.14513

项目页：https://tartanmatch.github.io/

28用视觉子目标引导长程世界模型规划

PILOT: Guiding Long-Horizon World-Model Planning for Robot Manipulation with Visual Subgoals and Policy Priors

PILOT 先生成中间视觉子目标，再用策略初始化 CEM 搜索，通过动作条件世界模型滚动优化。UR5e 与仿真结果支持子目标和策略先验配合，阅读时需计入在线规划预算。

论文：https://arxiv.org/pdf/2610.14335

项目页：https://pilot-planner.github.io/

29只为必要的机器人交互分配计算

Akule: Fast and Scalable Multi-Robot Motion Planning via Sparse Interaction Diffusion

Akule 冻结单机器人扩散先验，学习成对修正、交互选择与组合权重，降低大队伍联合计算。生成轨迹仍交由搜索规划器验证和修复，因此应比较完整规划流程。

论文：https://arxiv.org/pdf/2610.14248

项目页：https://akule-diff.github.io/

30奖励模型应追踪任务变化而非画面变化

Disentangling Task-Relevant Latent Dynamics for Reward Modeling

DiTaR 先从多帧视频解耦任务相关潜在动态，再以历史和语言条件流模型构造方向性稠密奖励。实验覆盖仿真及真机离线重标注，关注视觉干扰对奖励可靠性的影响。

论文：https://arxiv.org/pdf/2610.14159

项目页：https://ditarr.github.io/

31把语言规划与几何执行分工明确

Embodied Zero-Shot Task Planning via Hardware-Aware LLM-VLM Integration for Grounded Robotic Manipulation

框架让语言模型输出符号任务，视觉模型确认语义，再由确定性模块处理坐标、抓取、碰撞与可达性。验证在 Isaac Sim 的 FR3 数字孪生中完成，属于仿真操作方案。

论文：https://arxiv.org/pdf/2610.14106

代码：https://github.com/Ltankersley15/SK-Zero-Shot-Research

32预测固体运动引发的液体响应

Interactive 4D Volumetric Liquid Forecasting under Moving-Solid Interaction

TIDE/TIDES 根据已知的未来固体运动预测稠密液体速度场，引入界面残差与粒子网格反馈。注意其条件输入已给定固体轨迹，并非双向闭环流固仿真。

论文：https://arxiv.org/pdf/2610.16167

代码：https://github.com/jongwonryu/TIDE-TIDES

33导航失去进展后，学会恢复与回看

LERR-Nav: Learning to Coordinate Exploration, Recovery, and Reminiscing for Object Navigation

LERR-Nav 用高层视觉语言策略协调探索、停滞恢复和记忆回访，并结合文字执行摘要与地图历史。其启发是将脱困决策纳入策略学习，实验依据来自 HM3D 和 MP3D。

论文：https://arxiv.org/pdf/2610.15362

代码：https://github.com/kim-os/LERR-Nav

34让视觉语言模型规划双机器人路径

CoPlanVLM: Coordinated Path Planning for a Robot Team using Vision Language Models

CoPlanVLM 结合自由空间叠图、任务分类和专用推理提示，将语言指令转为双机器人协同路径。验证包含 Gazebo 与 Turtlebot 4，路径长度比较仅针对双方均成功的试验。

论文：https://arxiv.org/pdf/2610.14341

代码：https://github.com/AICPS/CoPlanVLM

35用器械骨架约束手术视频预测

SurgCast: Action-Conditioned Future Skeletons for Controllable Surgical World Models

SurgCast 先由机器人状态与未来动作预测器械骨架，再把几何和运动描述注入视频生成器。实验覆盖两套手术数据，研究重点是预测结构一致性，不能直接解释为临床操作能力。

论文：https://arxiv.org/pdf/2610.15645

项目页：https://surgcast.netlify.app/

36让不同视角共享同一个动态世界

Demiurge: Forging Navigable 4D Worlds via Autoregressive Video Rerendering

Demiurge 用动态四维点云缓存串联视频重渲染，结合深度补全、双流变形与几何细化维持跨轨迹一致性。研究聚焦可导航视觉世界生成，尚不能等同于机器人闭环控制。

论文：https://arxiv.org/pdf/2610.14685

项目页：https://hanxinzhu-lab.github.io/demiurge_project_page/

37在 MuJoCo 中研究空中柔性机械臂

An Open MuJoCo Simulation Framework for a Quadrotor Carrying a Tendon-Driven Soft Continuum Arm

框架将腱驱动连续体臂近似为柔性铰链连接的分段刚体链，与四旋翼控制联合仿真。轨迹、增益、工作空间和扰动实验均属仿真，适合研究飞行平台与软臂耦合。

论文：https://arxiv.org/pdf/2610.15308

代码：https://github.com/elhussieny/quad-soft-continuum

38分类正确不代表时序公式满足

Class-Conditional Conformal Prediction for Signal Temporal Logic Inference with Many Classes

研究时序逻辑分类的准确率与公式满足度错位，固定结构后调整谓词阈值，并逐类校准预测集合。有限样本覆盖依赖类内可交换性，适合关注可解释轨迹分类的读者。

论文：https://arxiv.org/pdf/2610.14352

项目页：https://sites.google.com/ucr.edu/multiclasswithconditionalcp?usp=sharing

39把多个人形机器人的动力学组合起来

Composable Dynamics for Cooperative Humanoids: A RoboCompiler Extension

RoboCompiler 扩展以成员级装配与两级约简组合人形动力学、手部连接和地面支撑。数值参考校验与 MuJoCo 仿真验证其一致性，不能将仿真跟踪误差直接外推到真机。

论文：https://arxiv.org/pdf/2610.14237

代码：https://github.com/Mehdi-Heydari-Shahna/RoboCompiler-Humanoids

40海上扰动下分配异构机器人任务

Reinforcement Learning Based Heterogeneous Task Allocation Under Spatio-Temporal Environmental Uncertainty for Multi-Robot Teams

研究不同能力无人艇在时空随机扰动与动态任务下的分配问题，用去中心化强化学习应对观测偏差。实验改变任务构成和队伍规模，结论适用于论文中的仿真设定。

论文：https://arxiv.org/pdf/2610.15237

代码：https://github.com/RobustFieldAutonomyLab/Spatio_Temporal_MRTA

41让本地策略学会按需请求推理帮助

Make π0.5 Learn to Call Astra for Reliable Robot Manipulation

工作冻结 π0.5，并用复用策略特征的 CALL 头判断何时请求外部模型介入恢复。证据来自 LIBERO 仿真，阅读应同时关注任务成功、外部调用频率与每回合成本。

论文：https://arxiv.org/pdf/2610.14117

项目页：https://kykgeek.github.io/Pi05_CALL_Astra/

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment