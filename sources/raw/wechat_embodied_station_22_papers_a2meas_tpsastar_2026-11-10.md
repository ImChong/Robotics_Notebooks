从失败诊断到动作纠正，机器人还缺怎样的接口？
具身智能小站 具身智能小站
 Nov 10, 2026, 6:00 AM

阅读路线

今天更新 22 篇具身智能新论文，覆盖长程操作与失败恢复、导航规划与安全控制、跨形态学习与人形运动、触觉、工具使用与人机交互等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进19 篇扫读

先读 A²-MEAS 的失败恢复接口，再跟进 TPS-A* 的碰撞检查与被动变形四旋翼的窄缝穿越；其余按阅读优先级排列，可结合自己的研究方向快速回看。

先看这一篇

1抓空之后，让纠错真正改变下一步动作

Toward the Era of Agentic Embodied Intelligence

 

一句话判断：长流程操作里，一次抓空就可能让后续步骤失去前提。A²-MEAS 把语言诊断、空间线索和执行历史转成动作模型可用的纠错表示，让机器人有依据地调整下一段动作。值得读的是推理与控制之间的接口，以及恢复后能否继续完成任务的评测方式。

你可以带走的两点

• 看完整任务成功率。作者在 BEHAVIOR 2026 公开集的 100 项任务、每项 10 次记录中，完整系统成功 454/1000，高于 A²-WLA 多模型配置的 406/1000；45.40% 是完整成功率，不能与部分目标完成度 Q-score 混用。

• 把失败历史接入动作生成。真机咖啡任务中，抓取失败并回退后，历史条件 Adapter 在“取滤杯并放稳”步骤完成 94/100，基础策略重试为 14/100。这是特定恢复步骤的结果，部署时权重固定。

方法亮点：将语言纠错与空间、时序证据编码成可学习的动作调制信号，使诊断能够影响连续控制；复现时应区分完整纠错接口与真机历史条件 Adapter，并留意诊断暂停带来的响应延迟。

论文：https://arxiv.org/pdf/2611.09124

代码：https://github.com/AutoLab-SAI-SJTU/A2MEAS

项目页：https://a2-meas.github.io/

另外 2 篇值得跟进

2异形机器人过窄门，按真实轮廓检查碰撞

TPS-A*: Path Planning for Any-Shape Vehicles with Certified Collision Checking in Trajectory Parameter Space

 

看点：长条形或非凸机器人过窄门时，圆形近似容易误判通行空间。TPS-A* 在轨迹参数空间预计算碰撞信息，让搜索复用真实轮廓的安全边界，适合关注移动机器人规划的人。

结果：作者在 BARN 的 300 个场景中，对共享目标位姿求解，全部 277 个目标位姿无碰撞的查询 均得到经独立核验的无碰撞路径；成功查询规划中位耗时 2.9 ms，SBPL 为 169 ms。计时仅含规划，安全边界针对给定障碍模型。

论文：https://arxiv.org/pdf/2611.09210

代码：https://github.com/MRPT/mrpt_path_planning

3让四旋翼接触折臂，穿过比机身窄的缝

Task-Driven Closed-Loop Design Optimization of Passively Morphing Quadrotors for Contact-Mediated Narrow Gap Traversal

 

看点：让无人机变窄，需要机身几何、关节柔顺性与控制共同配合。论文以闭环穿缝成功率联合优化这些设计，使机臂在接触时被动折叠，同时检查正常飞行的跟踪与能耗代价。

结果：在动作捕捉辅助的真机实验中，机身从 31.3 cm 折至 14.1 cm，按 2 m/s 参考速度 穿过 15×20 cm 开口；相对刚性机身，正常飞行位置误差仅多 2 mm。穿缝后仍有明显姿态和高度恢复过程。

论文：https://arxiv.org/pdf/2611.09272

代码：https://github.com/ntnu-arl/soft-quadrotor-optimization

19 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04给导航模型的自信分数加一道校准

CPNAV: Calibrated Vision–Language Navigation for Object-Goal Search in Unseen Environments

用保形预测校准视觉语言模型的动作分数，再配合碰撞过滤与树搜索回退。仿真和公寓真机测试均有改善；理论保证依赖校准与部署数据的可交换性。

论文：https://arxiv.org/pdf/2611.10366

代码：https://github.com/practice-lab-ucla/CPnav

项目页：https://practice-lab-ucla.github.io/papers/cpnav/

05从第一视角视频补出接触与相对压力

TouchAnnotator: Multi-Source Tactile Supervision with Privileged-View Distillation

把手套压力、人体接触和机器人指尖形变映射到统一手部表示，再用多视角教师训练第一视角标注器。输出可补充触觉标签，但不等同于跨设备校准的真实力。

论文：https://arxiv.org/pdf/2611.09129

代码：https://github.com/RammusLeo/TouchAnnotator

06把安全示范与高收益示范分开学

DexDICE: Reconciling Safety and Performance via Dual-Expert Offline Imitation Learning

用安全示范估计允许的状态动作区域，再从高收益示范学习表现，减少手写代价函数。论文覆盖离线基准与移动机器人；理想策略的安全证明不等于实际策略零违规。

论文：https://arxiv.org/pdf/2611.09433

代码：https://github.com/tzs930/dexdice

07给物体目标，让双手自己学会交接

SimDuoReal: Object-Centric Bimanual Policy for Emergent Dexterity

用统一的物体位姿目标训练双手专家，再蒸馏成单一策略，组合抓取、手内调整与交接。真机四项任务完成61/80次；仍需外部目标轨迹和准确的物体位姿跟踪。

论文：https://arxiv.org/pdf/2611.10410

项目页：https://simduoreal.github.io/

08自由握笔写字，同时学会稳笔与控力

Grip, Trace, Write: Force-Aware Imitation and Reinforcement Learning for High-DoF Robotic Handwriting

将握持与书写分成两个策略分支，先模仿逆运动学示范，再按墨迹、接触力和耗时强化学习。结果来自仿真；复现时还需区分论文八任务表与当前七任务评测。

论文：https://arxiv.org/pdf/2611.10451

代码：https://github.com/WillChan9/DexWriting

项目页：https://dexwriting.github.io/

09感知更准，也可能错过整场任务

EmbodiedIntentBench: Benchmarking Intent-Driven Runtime Adaptation for Mobile Robotic Missions

把覆盖、时限、证据和电量写成任务约束，评测计算决策如何影响无人机任务完成。基于固定仿真轨迹回放，揭示高召回率模型也可能因耗时错过关键观察。

论文：https://arxiv.org/pdf/2611.10498

代码：https://github.com/rajatbh21/EmbodiedIntentBench-artifact

10跟随指定行人，先保住目标身份

DobotTrack: Target-Grounded Vision-Language-Action Learning for Embodied Person Tracking

结合掩码与框监督，再按目标身份压缩视觉历史，让语言指定的行人在遮挡和干扰下更容易被持续锁定。主要量化结果来自EVT-Bench，真机部分为定性展示。

论文：https://arxiv.org/pdf/2611.09041

代码：https://github.com/Dobot-Team/dobot_track

项目页：https://dobot-team.github.io/dobottrack-web/

11学人类示范，先对齐动作发生的阶段

BATCH: Behavior Alignment for Cross-Embodiment Learning from Humans

用时间循环一致性和动态时间规整对齐人机行为阶段，再做联合训练与人到机器人的风格迁移。两项真机任务显示收益，但仍依赖机器人示范及相近的行为结构。

论文：https://arxiv.org/pdf/2611.10148

项目页：https://batch-embodiment.github.io/

12先筛候选动作，再想象执行后果

Tiresias: Towards Language-Conditioned World Models

把导航、操作、游戏与驾驶纳入统一评测，先从候选轨迹中筛选，再让世界模型想象后果并重排。可借鉴推理预算分配，但论文尚未验证真实机器人部署。

论文：https://arxiv.org/pdf/2611.10392

代码：https://github.com/UWMILab/Tiresias

13动作跟踪交给动力学，平衡交给残差

DKWBC: Decoupling Motion Tracking and Balancing for Humanoid Whole-Body Control

用逆动力学前馈和关节反馈承担动作跟踪，让学习策略专注平衡残差。实验覆盖跨动作集、负载和外扰；值得关注显式动力学如何减轻全身控制的学习负担。

论文：https://arxiv.org/pdf/2611.09605

项目页：https://robbyant-research.github.io/DKWBC/

14用可复用交互模板训练人形搬运

SUGARv2: Learning Diverse Visual Humanoid Loco-Manipulation Skills from Synthesized Interaction References

从可复用模板合成机器人与物体的交互参考，再用教师蒸馏与强化学习训练视觉策略。五项任务都有真机演示和计数，但各任务分别训练，抓取失败后的恢复仍有限。

论文：https://arxiv.org/pdf/2611.09266

项目页：https://tianshuwu.github.io/sugarv2-humanoid/

15没有标准工具时，记录人如何借物完成任务

TOOLDEX: An Ego-Exo Human Demonstration Dataset for Creative Physical Tool Use

同步采集第一视角与外部多视角，标注手—工具—目标物体交互，覆盖替代工具和可变形物体。评测串起工具选择、三维重建与策略学习，展示接触迁移的难点。

论文：https://arxiv.org/pdf/2611.09036

项目页：https://tooluse-dataset.github.io/

16把人体运动知识迁移到人形惯性估计

A Unified Inertial Encoder for Human and Humanoid Motion

通过共享惯性接口把人体动作与重定向人形动作放到同一输入空间，同时学习速度估计和动作语言对齐。可零样本用于G1，但姿态漂移仍会累积成轨迹误差。

论文：https://arxiv.org/pdf/2611.10137

项目页：https://unified-inertial-encoder.github.io/

17互动投入度评测，把儿童与成人一起纳入

Engagement Estimation in Human-Robot and Human-Human Interactions across Domain, Age and Culture: The MultiMediate’26 Challenge

把儿童与机器人玩耍加入跨文化互动评测，区分连续投入度和分类式任务、社交投入度。不同年龄与标签体系需要不同评估口径，不能只拿成人对话分数判断泛化。

论文：https://arxiv.org/pdf/2611.09913

资源：https://github.com/hcmlab/MultiMediate_26

项目页：https://multimediate-challenge.org/Dataset/

18在路径跟踪外加一层轻量避障过滤

LACART: Lookahead-Augmented Collision Avoidance and Reference Tracking Strategies for Safety-Critical Robots

用前视点控制屏障函数与二次规划修正车辆指令，将障碍、赛道边界和路径跟踪放进同一层。仿真计算开销较小，但车身安全仍依赖几何、感知及执行条件。

论文：https://arxiv.org/pdf/2611.09640

代码：https://github.com/Kanestrene/autonomous-racing-cbf

19搬椅子前，先选有利于最终摆放的抓点

Grasp-TAGS: Task-Aware Grasp Selection via Critic Distillation for Prehensile Loco-Manipulation

从强化学习评论家蒸馏抓点效用网络，按最终搬运目标选择初始抓姿，而非只看能否抓稳。结果来自八种物体的仿真重排，候选抓点仍由人工指定。

论文：https://arxiv.org/pdf/2611.09864

项目页：https://rpm-lab-umn.github.io/Grasp-TAGS-web-page/

20用四部手机记录柔性物体的受控运动

A four-view smartphone video dataset of a robot-held flexible strip, with camera poses and a 3D LaGSplat baseline

同步四路手机视频、相机位姿与机械臂输入，用拉格朗日动力学和高斯表示重建柔性条带。可用于形变与动力学建模，但移动相机位姿有缺失，标定也存在误差。

论文：https://arxiv.org/pdf/2611.09959

代码：https://github.com/LouenPottier/lagsplat-mpov

项目页：https://louenpottier.github.io/xrdays.html

21世界模型上机运行，先看数据搬运的代价

The World Model Planning Accelerator (WMPA): Hardware-Algorithm Co-Design for Embedded AI in Robotics

从理论能耗下界和GPU分析出发，提出近存计算与可变精度的世界模型规划架构。文章适合看硬件瓶颈；40倍降功耗是设计目标，并非已经流片验证的收益。

论文：https://arxiv.org/pdf/2611.10316

代码：https://github.com/Prk10/world-models-theoretical-min

22让社交机器人按情境说好第一句话

How to start a conversation proactively?

把用户状态、环境与触发事件组织成结构化语境，用CCO数据微调小语言模型生成主动开场白。评测以文本指标和模型裁判为主，尚不能代表长期真人互动效果。

论文：https://arxiv.org/pdf/2611.09906

资源：https://github.com/mohit-nathani1/conversation-opener-dataset

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment