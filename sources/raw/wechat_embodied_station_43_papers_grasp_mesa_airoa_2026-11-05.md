# 人能稳稳提起的罐子，换一双机器手怎么提？

> 原始抓取（2026-11-05，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/VS8xbDtQbQTJM3f_WuB9Og

人能稳稳提起的罐子，换一双机器手怎么提？
具身智能小站 具身智能小站
 Nov 5, 2026, 7:00 AM

阅读路线

今天更新 43 篇具身智能新论文，覆盖灵巧操作与示范学习、策略泛化与评测、世界模型与控制、移动导航与人机交互等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进40 篇扫读

先读 GRASP，看人手示范怎样变成机器手能执行的动作。接着看 MESA 怎样分项测泛化，再看 AIRoA-0 怎样用手持夹爪采集双臂操作示范。

先看这一篇

1换一双机器手，怎样复现提罐和倾倒？

GRASP: Gradient-Guided Sampling for Contact-Rich Physics-Based Motion Retargeting

 

一句话判断：人的手指能扣住提手，换一双指长和关节都不同的机器手，同样的手势却可能让容器滑落。GRASP 在物理仿真里同时跟踪手与物体，利用试过的控制轨迹来调整手指的落点和握持姿势，让机器手也能提起容器、稳稳倾倒。

你可以带走的两点

• 先看物体有没有跟上。 判定成功需要整条轨迹走完，物体平均位置误差低于 10 厘米、平均旋转误差低于 0.5 弧度。按此口径，双手仿真通过了 1,254/1,352 条（92.8%），同表最强基线 Do-as-I-Do 为 81.0%。

• 复现可从一次提拿开始。 作者仓库给了 Sharpa 双手的 OakInk2 配置。先准备物体网格和参考轨迹，再比较运动学初解与物理优化后的物体运动，就能看出接触在哪一步失效。

方法亮点：要把较长的人手示范转成机器手能执行的动作，GRASP 先从已跑过的仿真轨迹估计局部代价的梯度，用它引导下一轮采样，再把上一段的计划接到下一个时间窗。作者用同一跟踪目标处理灵巧手和人形机器人动作，优化后的轨迹还能用于训练转笔策略。真机灵巧手展示了开环提拿和倾倒，整批成功率来自仿真。

论文：https://arxiv.org/pdf/2611.05754

代码：https://github.com/ComputationalRobotics/GRASP_code

项目页：https://computationalrobotics.seas.harvard.edu/GRASP/

另外 2 篇值得跟进

2换个杯子、换个位置，分别测泛化

MESA: An Evaluation Framework for Compositional, Semantic, and Spatial Generalization in Robotics

 

看点：机器人换个位置还能抓杯子，换个没见过的杯子却可能失手。MESA 在单臂和双臂仿真中分别改变摆位、物体实例、类别和子任务组合，让人看清是哪一种变化拖低了成绩。

结果：π0.5 在单臂、双臂各五套测试中的平均成功率分别为 52.0% 和 48.7%。只看单臂，同分布为 62.8%，换到训练中未见过的物体类别降至 26.4%。该模型每项任务测试 50 次。

论文：https://arxiv.org/pdf/2611.05567

资源：https://github.com/pairlab/mesa-env

项目页：https://www.pair.toronto.edu/MESA/

3只用手持夹爪采示范，双臂学会换拖布

AIRoA-0: Long-Horizon, Precise, Cross-Embodiment Manipulation from Handheld Data

 

看点：人手持夹爪录示范，再把同款夹爪和腕部相机装上机械臂。AIRoA-0 靠两路近景画面判断任务做到哪一步，能按一条指令给扫地机器人更换拖布和滚刷。

结果：作者采集了 16,151 小时手持示范。在新场地的双臂 Franka 上，10 项任务各测 15 次，部分完成也计分，平均为 78%。用同样的任务示范做后训练，四个对照中，最高的 π0.5 为 27%。其他三种平台另做了任务演示。

论文：https://arxiv.org/pdf/2611.03815

项目页：https://a-0.airoa.io/

40 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04用另一支内窥镜检验手术场景的三维重建

iMED: A Multi-Endoscope Dataset for Surgical 3D Perception

两支立体内窥镜同步观察同一处组织，训练时用其中一支，测试时切换到另一支。iMED 用这一设置检验新视角合成等任务，让几何重建误差更难被近邻视角掩盖。

论文：https://arxiv.org/pdf/2611.04755

代码：https://github.com/surgical-vision/imed

项目页：https://imed-challenge.github.io/

05把仿真队列和推理队列分开，评测能快多少

RoboScope Kit: Unified Interfaces for Fast and Scalable Embodied-AI Evaluation

RoboScope 让仿真进程共享策略。同卡评测 LIBERO-Long 的 500 回合，耗时从官方评测的 2990 秒降至 671 秒，成功率相当。

论文：https://arxiv.org/pdf/2611.03850

资源：https://github.com/MINT-SJTU/RoboScope-Kit

项目页：https://mint-sjtu.github.io/RoboScope-Kit/

06用一次真机校准生成多视角示范

PyRUA-DataFarm: LLM Agent as a Cost-Efficient Teleoperator for Real-Ready Robot Data

每项任务录一条校准示范，再由智能体搭仿真、渲染多视角视频。五项真机任务换桌布和灯光后，合成数据组成功 87/100 次，等量遥操作示范组为 46/100。

论文：https://arxiv.org/pdf/2611.05320

代码：https://github.com/DAGroup-PKU/PyRUA-Lean/tree/DataFarm

项目页：https://dagroup-pku.github.io/PyRUA-Lean/DataFarm/

07人形机器人搬东西前，先比较动作的后果

Coordinating Humanoid Loco-Manipulation through Action-Conditioned Latent Foresight

先比较哪套走路与手部动作更接近示范的下一阶段，再执行开头几步。七项真机任务换物体配置后，该方法成功 46/70 次，GR00T N1.7 为 34/70。

论文：https://arxiv.org/pdf/2611.04839

资源：https://huggingface.co/datasets/ju6276/Action-Conditioned-Latent-Foresight

项目页：https://ju6276.github.io/Action-Grounded-Foresight/

08从地面够到高处的移动双臂平台

MABEL: An Open Anthropomorphic Mobile Bimanual Platform for Teleoperation and Learning

MABEL 用升降躯干、双臂和全向底盘扩大触及范围，并让遥操作与学习策略共用一套控制接口。作者列出约 8100 美元的零件成本，还实测了无线双目视频回传延迟。

论文：https://arxiv.org/pdf/2611.04083

代码：https://github.com/robotmabel/open_mabel

09学人手动作时，先预测桌上哪些点会动

FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video

FloMo 让模型预测场景中各点的三维位移和机器动作。作者在真实操作中用同一模型比较预测视频、预测运动和两者都预测，发现加入画面预测反而拉低了策略成功率。

论文：https://arxiv.org/pdf/2611.05062

代码：https://github.com/pairlab/flomo

项目页：https://flomo-wam.github.io/

10让模型想象物体移动之后的相对位置

FlowBench: Counterfactual Imagination Training for Spatial Intelligence

FlowBench 给出场景图，再问假设物体移动，空间关系会怎样改变。同一道题还可以给出结果画面作对照，由此区分模型是看不懂空间关系，还是推不出运动后果。

论文：https://arxiv.org/pdf/2611.03937

资源：https://github.com/liuyifan22/FlowBench

11机械臂到位后，还能稳稳停住吗

PolicyIK: High-Precision Inverse Kinematics for Robot Arms with Reinforcement Learning

机械臂够到目标后还可能慢慢漂开。PolicyIK 让策略记住关节指令与实际位置的差，再按剩余误差调整动作幅度。评测取最后一秒的平均误差，把“停得稳”也算进成绩。

论文：https://arxiv.org/pdf/2611.05372

代码：https://github.com/Tinaliu0123/PolicyIK

项目页：https://tinaliu0123.github.io/PolicyIK/

12离线挑策略，先看它从哪一步开始偏离

RACE: Calibrated Cumulative Scoring for Rollout-Free Robot Policy Checkpoint Selection

夹爪靠近物体时，几毫米偏差可能决定成败。RACE 用成功示范估计各状态可容忍的动作偏差，一旦偏得太远，就调低后续步骤的权重，据此给训练检查点排序。

论文：https://arxiv.org/pdf/2611.05838

资源：https://changyeon.site/race/data.json

项目页：https://changyeon.site/race/

13给手持夹爪换上软指，再学抓新物体

Robot-Matched Soft Fingers for UMI Data Collection: Lessons from 100 Hours of Pick and Place

手持采集器与机器人装上相同软指，指尖变形也进入画面。在 R1 Pro 上，先用多物体示范训练，再针对四件新物体微调，平均成功率从仅微调的 35% 升至 57%。

论文：https://arxiv.org/pdf/2611.03779

代码：https://github.com/lute-corp/soft-yubi

项目页：https://lute-corp.github.io/soft-yubi-site/

14指令少说几句，为什么机器人就不动了

VagueBench: Benchmarking the Robustness of Embodied Agents on Ambient Context Illusions

VagueBench 为同一任务配两种说法，只删去不影响意思的辅助信息。在室内仿真中比较六种智能体发现，精简措辞后更容易出现空计划，系统连第一步动作都没给出。

论文：https://arxiv.org/pdf/2611.03765

资源：https://github.com/stardreaming/VagueBench

15把人手操作视频变成机器人示范

RoboTryOn: Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On

保留桌上的物体和动作，把画面中的人手换成目标机器手，再从合成视频反推动作指令。RoboTryOn 用这套流程训练策略，专门测试了缺少该任务真机示范时的双手操作。

论文：https://arxiv.org/pdf/2611.05163

代码：https://github.com/ryan-seungyong-lee/RoboTryOn

项目页：https://ryan-seungyong-lee.github.io/RoboTryOn/

16用同一份三维状态预测画面和触觉

Hybrid Particle-Voxel Dynamics for Multimodal 3D World Simulation

作者把点云压成紧凑的三维状态，按动作预测它如何变化，再从中读出触觉与不同机位的画面。研究还检验了把这个模型用于规划时，对未见物体和机器人配置的适应能力。

论文：https://arxiv.org/pdf/2611.04492

代码：https://github.com/jwang771/3d-world-simulator

项目页：https://jwang771.github.io/3dworldsimulator/

17看见机器人将要抓偏，提前接手

Correct Robots Before They Make Mistakes: Proactive Human-in-the-Loop Intervention via Preference Learning

操作者能在扩展现实设备里看见机器人下一步打算怎么动。PHIL 把同一起点的原计划和人工纠正动作配成对，用于后续训练，并在四项真机任务中比较提前和事后接手。

论文：https://arxiv.org/pdf/2611.05667

项目页：https://intuitive-robots.github.io/PHIL-page/

18在线强化学习应离示范动作多远

GSBR: Geometric Support Behavior Regularization for Online RL

GSBR 用基础策略采样的动作估计可行范围，范围内允许策略挑出表现更好的动作，越过边界才逐渐加重惩罚。作者在四个机器人仿真基准上比较了这种做法与硬性限制。

论文：https://arxiv.org/pdf/2611.05295

代码：https://github.com/charlesxu0124/gsbr

项目页：https://charlesxu0124.github.io/gsbr/

19用疏密两种画面预测长程双臂操作

HiLo-WAM: Decoupling Long-Horizon Task Evolution and Local Dynamics in World Action Models

HiLo-WAM 用稀疏画面跟踪任务进度，用密集画面观察眼前的手物接触，两路信息共同引导动作。作者在仿真里分别改变时间跨度和采样密度，也在真机上验证。

论文：https://arxiv.org/pdf/2611.05023

代码：https://github.com/xywang227/HiLo-WAM

20人和机器手法不同，能否学会同一个动作

Human-to-Robot Transfer via Bridging Representation and Action Space in VLAs

H2RB 让人和机器人的示范共享动作表示，同时各自保留一套解码器，再用几何信息对齐所见场景。真机实验把换物体、换摆位和迁移新技能分开测试，示范无需逐条配对。

论文：https://arxiv.org/pdf/2611.04391

代码：https://github.com/jun-brro/h2rb-code

项目页：https://junhyeong-park.me/h2rb/

21给灵巧手同时指定物体朝向和手指姿势

DexHOT: Learning Hand-Object Configuration Transfer for Dexterous Manipulation

下一步能否接着转动物体，取决于手指落在哪里。DexHOT 把手形和物体姿态一起设为目标，学会短距离调整再串联动作，真机上 200 次姿态转换成功了 123 次。

论文：https://arxiv.org/pdf/2611.05119

项目页：https://dex-hot.github.io/

22省下生成未来视频的时间，让机器人更快出手

Future Representation Distillation for Imagination-Free World Action Models

DreamFree 训练视觉编码器从当前画面预测未来特征，部署时直接据此生成动作。在作者的测试平台上，140 亿参数的视频骨干最高可做 11 赫兹闭环控制。

论文：https://arxiv.org/pdf/2611.04527

项目页：https://harryjo97.github.io/dreamfree/

23碰撞的头几毫秒，让触觉先接管

ReflexArc: Bottom-Up Integration of Instantaneous Tactile Reflexes into Whole-Body Humanoid Control

触觉通道直接把退让姿态送到电机，全身控制器随后维持平衡。在 G1 碰撞实验中，与人为延后 20 毫秒的同一反射相比，即时启动使首次力峰降低 34.1%。

论文：https://arxiv.org/pdf/2611.04424

项目页：https://minusmodulo.github.io/reflexarc-website/

24让海洋预报决定下一次去哪里采样

Closed-Loop Adaptive Sampling on a Generative Ocean Model

多台海洋机器人依据洋流挑选能到达、又有望降低预报不确定性的采样点。在模拟海况中，这种策略通常优于固定巡航。模型最没把握的区域与实际误差最大的区域有时相距很远。

论文：https://arxiv.org/pdf/2611.05776

代码：https://github.com/UMich-RobotExploration/closed-loop-sampling

25让轮足机器人按步行地图穿过街区

WayTrace: A Map-to-Language-to-Trace Hierarchy for Embodied Urban Navigation

地图路线先拆成“向左过街”等短指令，再由车载模型在图像上画出行走轨迹。轮足机器人在苏黎世测试了四条实地路线，最长的一条为 1.37 公里。

论文：https://arxiv.org/pdf/2611.05129

26让小车提前为下一个窄弯留出空间

Barrier-State-Based Safe Navigation in Complex Environments Using Look-Ahead Safety Corridors

作者沿前方路径拼接多个安全区域，把连续弯道和窄通道纳入同一个约束。仿真测试包含移动障碍，差速小车实测则展示了如何通过窄口、绕开原地图上没有的障碍。

论文：https://arxiv.org/pdf/2611.05617

代码：https://github.com/TRL-NJU/planning-guided-safety-corridor

27机械臂怎样兼顾提重物和快速挥动

RAIPAL: A Powerful and Agile Humanoid Arm for Human Tasks

RAIPAL 依据人臂执行任务时的力量、速度和耐久需求设计关节。原型在不同测试中分别达到 20 公斤峰值负载和每秒 20.6 米的末端速度，持续负载另做热分析。

论文：https://arxiv.org/pdf/2611.04334

代码：https://github.com/donghyuk1/RAIPAL_SOEM

28让在人群中行走的机器人适应新场地

A Practical Causal Framework for Autonomous Mobile Robots: From Simulation to Real-World Human Environments

作者先在仿真中学出人机运动之间的因果结构，再用现场采集的速度、位置和行人轨迹更新模型参数。在人与机器人共享空间的导航实验中，比较了这种适配与非因果基线。

论文：https://arxiv.org/pdf/2611.04027

代码：https://github.com/lcastri/causal-sim2real

项目页：https://lcastri.github.io/causal-sim2real/

29下一条遥操作示范该采什么

GUILD: Guided Iterative Data Collection for Imitation Learning

让人反复演示同一类抓取，未必能帮机器人应付部署现场。GUILD 先请视觉语言模型提议场景，再估计每批示范对目标任务的贡献，据此决定下一轮该换什么物体、怎样摆放。

论文：https://arxiv.org/pdf/2611.05368

项目页：https://robin-lab.cs.utexas.edu/guild2026/

30动作预测的误差该传回视觉语言模型吗

Should the Action Expert Train the VLM? Action-Token Supervision and Gradient Routing in Mixture-of-Transformer VLAs

作者分别开关动作标记监督和梯度回传，再比较驾驶与操作策略。一个值得复查的差别是，开环预测里两种训练方式接近时，闭环驾驶中车辆沿路线走出的距离仍可能不同。

论文：https://arxiv.org/pdf/2611.05249

项目页：https://mot-vla-gradient-routing.github.io/

31让机械臂在大模型思考时继续反应

Asynchronous Robot Policies

大模型低频更新任务线索，小模型结合最新画面高频出动作。ARP 在训练时故意错开两者看到画面的时间，再用准静态操作和动态扰动仿真检验它能否跟上现场变化。

论文：https://arxiv.org/pdf/2611.05155

项目页：https://arp.insait.ai/

32让不同视角的模拟相机看见同一次动作

PrimeSim: A Generalist World Model Simulator from Large-Scale Cross-Embodiment Demonstrations

PrimeSim 用跨平台示范对齐动作与视频帧，再让多个视角交换信息。它在世界模型测试中能维持较长的连续预测，但细小位移的局部校准仍落后，使用时值得单独检验。

论文：https://arxiv.org/pdf/2611.04870

项目页：https://primebottech.github.io/PrimeSim-website/

33把抓取进度写成模型能读的几何量

Do Agentic Policies Need to Look at Every Step? Geometric Abstraction for Reasoning and Action

机械臂套环时，每一步都能把环孔与杆的偏差写成文字。作者用标定过的深度相机更新这些几何量，让语言模型按需要再看图，并在仿真里对比逐步看图的操作策略。

论文：https://arxiv.org/pdf/2611.05058

代码：https://github.com/Spectraorder/geometric-abstraction

项目页：https://geometric-abstraction.github.io/

34从无动作标签的视频里学控制

ELA: End-to-End Latent Action Policy Learning from Video Demonstrations

视频里的两帧之间发生了什么，既要让策略猜得准，也要能还原成机器指令。ELA 把这两个要求放进同一训练过程，在仿真和真机任务中用少量带动作标注的示范校准。

论文：https://arxiv.org/pdf/2611.05571

项目页：https://end2end-latent-action.github.io/

35机械臂做完一项任务，谁来重摆桌面

CRONOS: Benchmarking Non-Episodic Reinforcement Learning for Robotic Manipulation

CRONOS 让下一项仿真任务从上一项留下的桌面开始。作者比较了何时需要人工式复位，发现训练前期适当重置、后期重点处理无法恢复的场景，有助于策略学会连续操作。

论文：https://arxiv.org/pdf/2611.05080

项目页：https://embodiedai-ntu.github.io/cronos/

36先调整策略的噪声，再微调输出动作

CANDO: Conservative Action-Noise Directional Optimization for Residual Reinforcement Learning

CANDO 保留基础策略，让强化学习同时调整输入噪声和输出动作。作者用保守的价值估计筛选调整方向，在五项仿真任务中，将联合调整与只改其中一处的方案作了对照。

论文：https://arxiv.org/pdf/2611.04543

项目页：https://cando-rl.github.io/

37机器狗收拾玩具，先弄清要收哪几件

FEIJOA: Task-Driven Grounding and Execution for Mobile Manipulation

FEIJOA 接到指令后才查询场景，动手前检查是否够得着，动手后检查物体是否到位。Spot 实地试验成功 35/50 次，其中 17 次经历过失败后的恢复。

论文：https://arxiv.org/pdf/2611.04080

38同一套集群动作，观众看到了什么

Reality and Perceptual Gaps in Expressive Swarm Robotics

同一套分散控制程序被搬上桌面小车、无人机和人形尺度的移动平台。现场演出的轨迹和节奏随平台而变，观众问卷还把“看懂意图”和“感到安心”区分成了两种反应。

论文：https://arxiv.org/pdf/2611.05244

资源：https://git.initrobots.ca/aimran/buzz_unified

项目页：https://initrobots.ca/dessaim/

39软物体没变形，是拉不动还是看不见

A Tutorial on Shape Servoing of Deformable Objects-Part II: Advanced Concepts

这篇教程用变形雅可比矩阵分析哪些形状变化容易做到，再用数值示例区分三类失效。机器人可能拉不动物体，传感器可能看不到变化，材料本身也可能突然失稳。

论文：https://arxiv.org/pdf/2611.03825

代码：https://github.com/nachocz/tutorial_on_shape_servoing_of_deformable_objects

40把视频里的动作压成四个标记

DreamX: Video Diffusion Model is a Universal Latent Action Learner

同一个动作换个背景或换只手，画面就会大变。DreamX 从视频扩散模型中把每帧的运动变化压成四个动作标记，并分别测试它们能否迁移到新视频和新机器人的控制中。

论文：https://arxiv.org/pdf/2611.05377

项目页：https://cvlab-kaist.github.io/DreamX/

41机器人怎样循着气味找漏点

OdorSpaces: Visual-Olfactory Embodied Navigation in 3D Environments

泄漏的丙烷沿地面扩散，火源上方的气体则会上升。OdorSpaces 把气体输运接入室内导航仿真，让机器人结合相机和沿途浓度读数寻源，并分别评估这两种情形。

论文：https://arxiv.org/pdf/2611.05738

42旁人走近时，机器人怎样提醒学生写错了

For You, Not for Everyone: Bystander-Aware Delivery of Critical Feedback in Robot Tutoring

一项 40 人实验让机器人在旁人出现后转身，用屏幕私下给出预设纠错提示。与当众念出相同反馈相比，参与者更认可它的隐私意识。两组尴尬感的差异仍不明确。

论文：https://arxiv.org/pdf/2611.05001

代码：https://github.com/atikkhannilgar/Bystander_adaptive_robot_tutor_in_shared_setting

43耳机提示配合轻轻侧头，旁人会察觉吗

Just Between Us? Exploring a Wearable-Mediated Channel for Discreet Human-Agent Interaction in a Shared Game Setting

纸牌游戏中，玩家从耳机听取提示，侧头回应研究员操控的屏幕角色。40 个使用该通道的回合有 39 个按计划完成，访谈中其他玩家对这种配合是否公平看法不一。

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