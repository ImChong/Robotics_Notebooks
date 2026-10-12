# 原始抓取：机器人会操作，为什么仍听不懂空间指令？（具身智能小站 · 34 篇）

- **原文：** <https://mp.weixin.qq.com/s/HSsx-QeViQjmnsOTL_FU2Q>
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA），页面纯文本，未做改写
- **抓取日期：** 2026-10-12
- **说明：** 供追溯条目线索；论文技术事实以各详情节点引用的 arXiv 原文为准。

---

机器人会操作，为什么仍听不懂空间指令？
Original 具身智能小站 具身智能小站
 Oct 12, 2026, 6:00 AM

阅读路线

今天更新 34 篇具身智能新论文，覆盖策略部署与时间建模、世界模型与数据合成、灵巧操作与全身控制、导航、记忆与评测等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进31 篇扫读

先深读 Chronos 的物理时间建模，再跟进 Spa-Bench 的真机空间诊断与 LeWAM 的联合世界动作学习，其余按研究方向快速浏览。

先看这一篇

1让机器人策略显式理解物理时间

Chronos Policy: Learning Robot Policies on Physical Time

 

一句话判断：相机降帧、传感器不同步或推理变慢，都会让固定序列位置失去原来的时间含义。Chronos 把观测采集时刻与动作执行时刻写进策略，用时间感知 RoPE配合时序随机化训练，提供统一处理部署时序变化的思路，尤其适合动态操作与多传感器接触任务。

你可以带走的两点

• 把采集时间与执行时间作为条件，让注意力读取物理时间差；仅做时序随机化可能引入跨传感器歧义。

• 仿真 Conveyor 的 0–200 ms 推理延迟条件下，成功率从 DP 的 1.8% 到 76.4%；每方法每条件 450 回合，跨三训练种子汇总。 真机评分含半分部分成功；可变控制频率仍待研究。

方法亮点：新增的是物理时间进入注意力位置编码的机制，而不只是添加延迟扰动。时序随机化与时间条件共同支持异步、多速率数据学习。

论文：https://arxiv.org/pdf/2610.13929

项目页：https://changhaowang.github.io/chronospolicy/

另外 2 篇值得跟进

2会操作，不代表理解空间指令

Spa-Bench: A Real-Robot Benchmark for Spatial Grounding in Vision-Language-Action Models

 

看点：Spa-Bench 用真机配对场景区分操作能力与空间理解。保留对象和动作训练支持，只改变空间组合，再用直接点名指令做对照，帮助定位 VLA 失败原因。

结果：同场景把空间描述换成直接点名，三项完整评测策略成功率提高 28.3–40.0 个百分点，用于区分目标选择与动作执行瓶颈。 六项 SO-101 真机任务；4,092 回合含 1,200 示范和 2,892 评测。五策略中只有三项完成全协议，不宜混作统一排名。

论文：https://arxiv.org/pdf/2610.12696

项目页：https://justintiensmith.github.io/Spa-Bench/

3用同一个潜在模型执行与规划

LeWAM: End-to-End World and Action Modeling with JEPAs

 

看点：LeWAM 联合学习未来潜在状态与动作流匹配，让模型既能直接执行，也能利用动力学筛选动作。十类仿真环境与真机行为克隆评测支持这一设计，但仍按任务和形态分别训练。

结果：潜在动力学与动作联合训练，用高斯约束改善表征和测试时规划；可直接执行，也可搜索候选动作。 区分带目标图像的视觉规划与无目标图像的任务行为克隆。跨物体、视角和形态泛化仍需进一步验证。

论文：https://arxiv.org/pdf/2610.12818

代码：https://github.com/MinghaoFu/lewam

项目页：https://le-wam.github.io/

31 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04用有价值的经验持续改进机器人策略

Evo-RL: Iterative Robot Reinforcement Learning from Selective Experience

Evo-RL 将人类纠正与成功轨迹中正优势片段结合，迭代筛选训练经验。配套跨形态数据与真机精细操作实验，适合关注强化学习落地与经验复用的读者。

论文：https://arxiv.org/pdf/2610.13345

代码：https://github.com/MINT-SJTU/Evo-RL

项目页：https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset

05把世界动作模型压缩到更低推理成本

SmolWAM: A Latent World-Action Model for Robotic Manipulation on Edge Devices

SmolWAM 保留教师预训练主干，裁剪深度、缓存文本并以单步流生成动作和未来潜变量。作者在 LIBERO 上报告速度收益；尚无真机验证，长时域任务受裁剪影响更明显。

论文：https://arxiv.org/pdf/2610.13740

代码：https://github.com/iitimii/SmolWAM

项目页：https://smolwam.vercel.app/

06少量示范微调时保住原有泛化能力

RARA: Robust Policy Fine-Tuning through Retrieval-Anchored Representation Alignment for Robotic Manipulation

RARA 先检索意图与运动相近的预训练示范对齐表征，再限制微调漂移。覆盖两类策略、仿真与真机共十三项任务，提示适配时应同时检查域内表现与分布外泛化。

论文：https://arxiv.org/pdf/2610.13578

项目页：https://shua-kang.github.io/RARA_robot/

07同一份人类示范服务不同灵巧手

Human2Dex: Dexterous Robot Learning from Wrist-Centric Human Demonstrations

Human2Dex 用腕部视角、统一手部关键点与抓握口袋坐标对齐人类和机器人信息。两种灵巧手各自重定向并训练策略，六项真机任务支持示范接口复用。

论文：https://arxiv.org/pdf/2610.13119

项目页：https://human2dex.pages.dev/

08用未来监督动作，部署时省去未来生成

Forward Dynamics Models

FDM 让动作特征预测未来状态，训练梯度回流到策略，推理时跳过未来生成。视觉语言与视频生成两类主干均做实验，为减少世界模型部署成本提供方向。

论文：https://arxiv.org/pdf/2610.13117

项目页：https://logosroboticsgroup.github.io/FDM/

09在同一命令流上比较机器人运动平滑性

JitterVLA: A Replay Benchmark for Motion Quality in VLA-Driven Human-Robot Manipulation

JitterVLA 固定命令和到达时序，成对回放不同控制器，测量冲击与跟踪误差的取舍。通用质点实验和小规模 UR3 检查用于筛选方案，不能直接证明人机交互安全。

论文：https://arxiv.org/pdf/2610.13628

资源：https://huggingface.co/datasets/armteam/JitterVLA-teleop-episodes

项目页：https://faithogunwoye.github.io/JitterVLA/

10用物理验证筛选世界模型合成示范

Grounded-Dreamer: Robot Learning from World Model Synthetic Data with Physical Grounding

Grounded-Dreamer 从少量真实初始图像重建任务场景，以仿真轨迹训练视频世界模型和逆动力学，再回放验证合成动作。真机策略仅使用筛选后的合成数据训练。

论文：https://arxiv.org/pdf/2610.13164

项目页：https://fffstrong.github.io/Grounded-Dreamer/

11用光流监督轻量策略的未来运动预测

Evo-World: A Lightweight Vision-Language-Action Model with Future Motion Prediction via Optical-Flow Supervision

Evo-World 用光流强调场景运动，减少未来图像中静态外观的干扰，并保留预测表征。Meta-World、RoboTwin 与 AgileX 真机评测支持轻量 VLA 的这一训练思路。

论文：https://arxiv.org/pdf/2610.12936

代码：https://github.com/MINT-SJTU/Evo-World

12让视觉语言模型直接看懂动作坐标系

VLMs are Native Action Models: Unleashing the Manipulation Capabilities of VLMs with Explicit Action-Space Grounding

EASG 将工具中心动作坐标系投影到腕部图像，让 VLM 用文本接口输出连续动作，并沿用同一接口微调。跨机械臂实验共享夹爪，跨形态结论仍有适配边界。

论文：https://arxiv.org/pdf/2610.12832

资源：https://github.com/General-Robo/EASG

项目页：https://general-robo.github.io/EASG/

13让飞行操作智能体主动查明失败原因

Fly-by-Code: Embodied Coding Agents for Aerial Manipulation with Active Visual and Physical Feedback

Fly-by-Code 在仿真中生成诊断程序，主动换视角或测试接触假设，再修改任务程序。六类仿真任务与两项小样本真机转移，说明反馈获取也应成为可执行步骤。

论文：https://arxiv.org/pdf/2610.12549

项目页：https://fly-by-code.github.io/

14让人形机器人用身体接触完成操作

PanTact: Teaching Whole-Body Dexterity with a Wearable Contact Interface

PanTact 以可穿戴接触贴片记录人类全身示范，策略预测接触目标，再由控制器选择机器人表面实现。G1 展示支撑、推扫和分担负载，拓宽双手之外的操作方式。

论文：https://arxiv.org/pdf/2610.12558

项目页：https://pantact.github.io/

15减轻长尾任务被高频动作模式压制

Evidence-Guided Counterfactual Learning for Long-Tailed Robotic Policies

证据引导反事实学习定位指令与动作相关视觉信息，通过移除证据构造反事实监督，并按示范频次调整训练。仿真与真机检查尾部任务收益，部署时不增加计算。

论文：https://arxiv.org/pdf/2610.12861

代码：https://anonymous.4open.science/r/EVIDENCE-GUIDED-COUNTERFACTUAL-LEARNING-FOR-LONG-TAILED-ROBOTIC-POLICIES-440F/

项目页：https://krlozhilderman287-creator.github.io/EVIDENCE-GUIDED-COUNTERFACTUAL-LEARNING-FOR-LONG-TAILED-ROBOTIC-POLICIES/

16把实验室动态现象纳入机器人评测

AutoBio+: Benchmarking Dexterous Manipulation with Dynamic Experimental Phenomena in Realistic Laboratories

AutoBio+ 覆盖十四项任务和四种机器人形态，将火焰、蒸汽、气泡与真实实验室重建背景结合。并行仿真支持批量示范和闭环评测，真机检查用于考察迁移差距。

论文：https://arxiv.org/pdf/2610.13643

项目页：https://autobio-plus.github.io/AutoBioPlus

17用因果扩散规划精准推放多个物体

Non-Prehensile Multi-Object Placement with Mobile Robots via Diffusion-Based Trajectory Planning

该方法先去噪机器人轨迹，再条件生成物体轨迹，以推物成本引导采样。包含 Push-T、多箱仿真与 TurtleBot 真机测试，适合研究移动机器人精确非抓取操作。

论文：https://arxiv.org/pdf/2610.12684

项目页：https://anonymous.4open.science/r/Causal_Diffusion_Planner-0181

18扰动后撤回过时结论，再安排修复

REMM: Revisable Embodied Memory for Manipulation

REMM 为冻结的操作策略增加可修订记忆，保留证据历史并维护任务条件的依赖关系。五十一项真机试验含操作者协助，报告的是记录进度，不能等同完全自主成功率。

论文：https://arxiv.org/pdf/2610.12710

项目页：https://revanthsenthil.github.io/remm/

19世界模型也需要学习失败的交互动力学

Beyond Successful Robot Demonstrations: Learning from Human Interactions and Modeling Failure Dynamics

HF-WM 用稳定点流连接人类视频与机器人交互，再以扰动动作和奖励后训练补充失败动力学。评测覆盖动作条件视频生成和策略评估，生成质量与控制成功需分别看待。

论文：https://arxiv.org/pdf/2610.13522

项目页：https://peiyantu.github.io/HF-WM/

20用对象关系图识别操作失败

Robot-WATCHDOG : failure detection through object-centric graph representation

Robot-WATCHDOG 仅从正常示范学习对象时空图，比较几何关系评分与未来状态预测。两类数据共六项任务、多个随机种子评测，重点检查语义和交互层面的错误。

论文：https://arxiv.org/pdf/2610.13403

项目页：https://cea-list.github.io/Robot-WATCHDOG/

21把生成视频转为受约束的机器人轨迹

CASTER: Formalizing Zero-Shot Manipulation from Synthetic Videos as Constrained Optimization

CASTER 将合成视频作为轨迹搜索初值，在数字孪生中施加物理可行性与语义约束。十二项真机任务检查零样本操作，提示视频合理性与轨迹可执行性需要分别验证。

论文：https://arxiv.org/pdf/2610.12567

项目页：https://collab.me.vt.edu/CASTER

22通过时序专家蒸馏适应复杂导航场景

TAME-Nav: Temporal Asymmetric Mixture-of-Experts Distillation for Agile and Versatile Robot Navigation

TAME-Nav 用具备障碍跟踪信息的教师指导受噪观测学生，结合历史编码、软专家路由与域随机化。仿真和四足真机检查脱困及狭窄动态通道，试验含轻微接触。

论文：https://arxiv.org/pdf/2610.12877

项目页：https://tame-nav.github.io/

23没有双目相机，也能扩充双目示范

StereoEngine: Scaling Stereo Bimanual Robot Data without Stereo Cameras

StereoEngine 一路从单目视频估深、变形和补全右视图，另一路将仿真双目联合转成真实外观。双臂操作实验检验合成数据价值；部分场景仍需人工设计提示。

论文：https://arxiv.org/pdf/2610.12734

项目页：https://stereoengine.github.io/

24用优先级保住动作迁移中的关键接触

PriMoRetarget: Humanoid Motion Transfer Through Prioritized Motion Retargeting

PriMoRetarget 用明确层级替代多目标加权，让低优先级更新保留脚落点和交互接触。四类数据集检查运动和跟踪，但源数据抖动与穿透等缺陷仍可能被继承。

论文：https://arxiv.org/pdf/2610.12642

项目页：https://nickioan.github.io/primoretarget-page/

25只用二元奖励学习接触丰富的装配

AlphaNIST: Emergent Assembly Skills from Binary Task Rewards

AlphaNIST 在仿真中从零训练单策略完成 NIST 二十类装配，不使用示范、稠密进度奖励或脚本原语。观察到装配准备与恢复行为，训练成功率不能视为真机部署结果。

论文：https://arxiv.org/pdf/2610.13354

项目页：https://alpha-nist.github.io/

26比较机器人该存文字、图像还是场景图

What Should Embodied Agents Remember? Benchmarking and Evaluating Memory Modality for Embodied Agents

MeMoREval 先积累未知后续用途的经历，再在没有新观测时回答或行动。四个仿真环境、五十三项任务同时衡量成功与记忆成本；两项真机演示验证协议可行性。

论文：https://arxiv.org/pdf/2610.12561

资源：https://github.com/jimwang418/MeMoREval

项目页：https://MeMoREval.github.io/

27让策略预测动作将产生的物理效果

Beyond Actions: Learning Future Operation Targets for Vision-Language-Action Models

AFCE 把动作、机器人响应与稀疏视觉变化压成效果词元，再解码为动作块。DexJoCo 检验单双臂任务；报告波动来自评测种子，同一方法只使用一个训练检查点。

论文：https://arxiv.org/pdf/2610.13212

项目页：https://afceproject.github.io/AFCE-Web/

28根据新旧信念的一致程度修改推车计划

Belief-Overlap Partial Inversion: Object-Centric Flow Replanning for Humanoid Loco-Manipulation

BOPI 将在线物体参数估计、流规划和全身控制闭环连接，按信念差异决定部分反演深度。十八类仿真目标与真机推车展示，强调修订计划时兼顾适应与连续性。

论文：https://arxiv.org/pdf/2610.13087

项目页：https://bopi-loco.github.io/

29把群体强化学习复现变成平台职责

MARGoS: A Configuration-Driven Platform for Reproducible MARL Experiments in ARGoS

MARGoS 将配置哈希、随机种子传递、环境指纹和导出实验包纳入 ARGoS 实验平台。五个维度评估复现与协作流程，价值集中在研究工程基础设施。

论文：https://arxiv.org/pdf/2610.13772

代码：https://github.com/tombackert/margos

30从普通视频学习反应式人形网球

ATHLETE: Learning Reactive Humanoid Tennis via Trajectory-Guided Motion Matching

ATHLETE 将通用行走与视频击球动作做轨迹引导匹配，再经教师学生学习和强化学习形成反应策略。真机打球依赖外部动捕提供球位置与机器人根部姿态。

论文：https://arxiv.org/pdf/2610.13572

项目页：https://anonymous.4open.science/w/ATHLETE2027/

31在固定动作词元预算内安排两条动作链

LeapBot-JAC: Joint Action Chains for World Action Models

LeapBot-JAC 区分稀疏未来参考与密集执行目标，共享动作专家并对齐物理时间。LIBERO、泛化基准和两项真机任务检查收益，主要比较仍是单训练种子。

论文：https://arxiv.org/pdf/2610.12886

项目页：https://github.com/LeapWM/leapbot-jac

32通信何时对群体机器人真正有用

The value of information and the emergence of communication in evolved e-puck robots

两台仿真 e-puck 通过单比特信号寻找食物，演化比较循环网络与内置动作选择架构。有直达目标的罗盘时通信消失，提示通信价值取决于新增信息而非信号存在。

论文：https://arxiv.org/pdf/2610.13118

代码：https://github.com/ferdiex/easagru

项目页：https://doi.org/10.5281/zenodo.23051976

33把多模态手术导航集成到机器人控制台

Multimodal Image-based Robotic Assistance and Guidance Environment (MIRAGE): A Novel Unified Framework for Surgical Navigation and Visualization

MIRAGE 将双目重建、增强现实、导航与术中超声集成到达芬奇控制台。四种平台上的模体实验检查配准与显示误差，结果属于临床前技术验证，尚不能推断临床获益。

论文：https://arxiv.org/pdf/2610.12780

项目页：https://sites.google.com/dartmouth.edu/halter-lab/resources/mirage

34用多帧预训练识别长时段事件动作

Event-based Action Recognition by Pre-Training on Multi-Frame Representations

该工作以多帧事件表示、稀疏感知掩码和词元选择训练视觉 Transformer，改善长且复杂的动作序列识别。两类事件数据集评测提供证据，研究对象是识别而非控制。

论文：https://arxiv.org/pdf/2610.13856

项目页：https://github.com/nrthby/event-action-mae

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment