# 微信公众号原始抓取：机器人出错后，退一步能否做得更好？｜附论文与复现线索

> 原始抓取落盘（正文纯文本，未下载图片）。来源见 [sources/blogs/wechat_embodied_station_robot_31_papers_2026-10-15.md](../blogs/wechat_embodied_station_robot_31_papers_2026-10-15.md)。

- **原始链接：** https://mp.weixin.qq.com/s/ZfE6_UgqmHFePROtf2eE2w
- **抓取方式：** WebFetch 返回「环境异常」验证页；改用浏览器 UA 直接取 HTML 后本地解析 `#js_content`（2026-10-15）

---

TITLE: 机器人出错后，退一步能否做得更好？｜附论文与复现线索
DESC: 跟进 VLA 动态参数适配、执行恢复、人形移动操作与触觉数据，兼顾仿真生成和统一评测。整理方法看点、实验边界及论文、代码与项目入口，方便研发读者快速定位值得深读的工作。
nickname htmlDecode(
ct et(r,G)
user_name gh_a617744e2c14
createTime 2026-10-15 14:00
BODY_LEN 9581
style="visibility: hidden; opacity: 0; ">
阅读路线
今天更新 31 篇具身智能新论文，覆盖VLA 适配与执行恢复、人形控制与移动操作、触觉数据与接触学习、仿真生成与评测等方向。为了方便快速浏览，本期按阅读优先级分成三档。
1 篇深读2 篇跟进28 篇扫读
先深读 MorphAct 的动态参数适配，再跟进 ACRO 的执行恢复和 TacClaw-Infinity 的接触数据管线；其余按优先级扫读，沿论文与资源入口继续查看。
先看这一篇

1同一套 VLA 参数，也能随操作情境调整
MorphAct: Robot Policies Morphing with the Worlds They Act In

[IMG https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaB39Xq4KGHX06RL1bjL9G12YV1xgyqicTVSWjvYYScRTQtgyT8zooxWImwU9WYsqI3RfYSKpJC1AiaGyxiakLVxiamdNqnnGIU1LM/640?wx_fmt=png&from=appmsg]

一句话判断：固定参数要同时承担接近、抓取和放置等不同操作，训练信号可能相互牵制。MorphAct 让视觉语言与动作模块按当前情境生成耦合的低秩更新，值得关注它如何缓解共享参数冲突，以及性能收益对应的推理代价。
你可以带走的两点
• 先改表示，再生成动作更新：视觉语言侧适配后的特征继续条件化动作侧生成器，两者通过原生动作损失联合训练，无需操作标签或专家适配器目标。
• 增益与代价一起看：真机每任务每方法 50 次测试，五金分类从 8% 到 22%，即增加 14 个百分点，绝对成功率仍低；相对共享 LoRA，π0/π0.5 动作块延迟约增加 18–21%。
方法亮点：把情境适配从激活变化推进到有效参数动态调整，并让两侧生成器沿表示到动作的计算依赖耦合。参数每次重规划生成，在动作块内部复用。
论文：https://arxiv.org/pdf/2610.20019
代码：https://github.com/MINT-SJTU/MorphAct
项目页：https://mint-sjtu.github.io/MorphAct.io/
另外 2 篇值得跟进

2让机器人退回更有希望的状态重试
ACRO: Actor-Critic Rollout Orchestration

[IMG https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYdXDOpTvEibEcDCvKU1Uu7yttjtf338bUAbjY6Y0uzTR9WBDvvxiaNEPz6ftV6WUVGwiaJLScWz7IswdJVHzDpGBgXtSa9Q8Ecrg/640?wx_fmt=png&from=appmsg]

看点：同一策略在错误状态下继续执行，未必能靠多试几步恢复。ACRO 学习何时中止、退回何处，再让固定 VLA 从新观测继续执行，并把回退动作计入执行预算。
结果：同预算比较恢复能力：三项真机任务合计 60 次测试，从 26/60 成功增至 39/60；回退计入预算。Critic 需部署策略的执行轨迹训练，不能视为零数据外挂。
论文：https://arxiv.org/pdf/2610.19677
代码：https://github.com/JunHoo-Lee/ACRO
项目页：https://junhoo.me/acro

3把接触过程纳入规模化示范数据
TacClaw-Infinity: A Large-Scale Vision-Tactile-Language-Action Dataset for Dexterous Manipulation

[IMG https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZbGtSicCsFLS6t8EZp6G753stGDRjqicREs0iaxDfNa6XGibT9Z8j7zhZR1rt1p5q0pBnWialFuRrLOdJUmcZfd71ficSAd8f2D2s8M/640?wx_fmt=png&from=appmsg]

看点：接触操作的数据难点不只在采集数量，还在视觉、触觉与动作同步后的质量控制。TacClaw-Infinity 把手持采集、可执行性筛选和结构化标注连成一条数据管线。
结果：作者报告超万小时、百万回合的完整采集规模；模型实验在三种机器人上检验无专属机器人示范的迁移。复现时应区分完整语料与当前提供的数据子集，并先核对坐标系和末端标定。
论文：https://arxiv.org/pdf/2610.20320
资源：https://github.com/daimon-robotics/TacClaw-Infinity
项目页：https://daimon-robotics.github.io/tacclaw-infinity-page/
28 篇一眼扫完
下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。
04用执行反馈校准原子动作推理
Atomic-WAM: Execution-Grounded Atomic-Motion Reasoning for World Action Models
Atomic-WAM 先用视觉语言模型描述局部动作，再交给世界动作模型执行；固定执行模型，以执行反馈优化推理器。主要结果来自 RoboLab 的真实到仿真评测。
论文：https://arxiv.org/pdf/2610.20157
代码：https://github.com/ssj9596/Atomic-WAM
项目页：https://ssj9596.github.io/Atomic-WAM-project/
05力反馈有效，还要看用在什么任务
ForceBench: Benchmarking Force Integration for Robotic Manipulation
ForceBench 在统一骨干和控制设置下比较腕力融合方式，覆盖物性推断、阻力交互与装配。仿真和真机显示，平均分接近的融合设计也可能有相反的任务取舍。
论文：https://arxiv.org/pdf/2610.19604
项目页：https://forcebench-platform.github.io/
06先诊断移动操作在哪一环失败
Cracking Mobile Manipulation: From Failure Diagnosis to Agentic Robot System
固定子任务 VLA，分别补足规划、视觉记忆、底盘可达性与末端修正。两类仿真基准和真机检验系统干预，说明局部操作能力不能直接代表长时程系统可靠性。
论文：https://arxiv.org/pdf/2610.19600
代码：https://github.com/PKU-VaLuE-Lab/Cracking-Mobile-Manipulation
07用球面表示串起长时程规划
Hyperspherical Contrastive Representations Unlock Planning and Exploration
C-Slerp 从对比表示中以球面插值推导中间路标，并随目标难度调整路标数量。在导航与堆叠等实验中验证，八块积木结果属于文中实验环境，不能外推真机。
论文：https://arxiv.org/pdf/2610.20409
资源：https://github.com/jonghaepark/c-slerp
项目页：https://jonghaepark.github.io/c-slerp
08把灵巧手接触技能与臂手协调拆开
HiFun: a Hierarchical Framework for Efficient Functional Dexterous Manipulation Learning
HiFun 先学习接触感知手部技能，再学习机械臂运动和技能触发，结合人工恢复与运动基元增强。六项真机任务、两种灵巧手验证，关注在线训练和接触恢复条件。
论文：https://arxiv.org/pdf/2610.19518
项目页：https://hly-123.github.io/HiFun/
09失败轨迹中也有值得学习的动作
Graph-Reweighted Behavior Cloning with Human Interventions for Precise Robotic Manipulation
GR-BC 在混合质量轨迹上建立目标可达图，反向搜索剩余步数，再按实际进展重加权动作块。四项真机精细操作验证，价值在于复用失败与人工恢复片段。
论文：https://arxiv.org/pdf/2610.20504
项目页：https://gr-bc.github.io/
10动作只在需要时多算几步
Refine Actions If Needed: Adaptive Action Chunk Refinement for Robot Manipulation
RAIN 先预测不同精修深度的动作误差，再决定是否调用精修器。八项仿真与四项真机任务检验成功率和计算分配，区分策略延迟与成功任务完成时间。
论文：https://arxiv.org/pdf/2610.19765
代码：https://github.com/rain-policy/RAIN
项目页：https://rain-policy.github.io/
11主动选择更有用的仿真训练数据
Active Generative Simulation: Meta-Learning What to Simulate for Sim-and-Real Policy Co-Training
MetaGenSim 用元梯度学习仿真数据分布的评分器，在用户给定参数空间内选择数据。比较均匀生成与检索，并检验代理策略到 VLA 的迁移及任务不完全对齐的情形。
论文：https://arxiv.org/pdf/2610.19479
项目页：https://meta-gensim.github.io/
12离散动作的瓶颈可能不是精度
When Do Discrete Actions Make Good Continuous Controllers?
比较连续与离散动作在单任务、多任务 VLA、仿真和真机中的表现，指出跨时间与动作维度的因子化误差。离散流与引导机制提供了不同的修正路径。
论文：https://arxiv.org/pdf/2610.20429
项目页：https://iscoyizj.github.io/discrete-action-website/
13移动底盘与机械臂，分层学习修正
Hierarchical Residual Policy Learning for Real-World Mobile Manipulation with Sparse Human Guidance
HiRe-MoMa 冻结基础策略，将残差拆成底盘与机械臂两路，并用稀疏人工干预对齐价值函数。在 TIAGo++ 真机上验证在线微调，关注两类基础策略的适配。
论文：https://arxiv.org/pdf/2610.20607
项目页：https://hire-moma.github.io/
14按动作扰动稳定性调整执行长度
Accelerating VLA Inference with Perturbation Informed Execution
PICES 在一次策略调用内比较名义动作与局部噪声扰动样本，联合选择动作块和执行长度。无需额外训练；端侧收益对应特定硬件及固定长度基线。
论文：https://arxiv.org/pdf/2610.19656
代码：https://anonymous.4open.science/r/PICES_VLA-B1BF
15让三维组合经历物理交互
PhySGA: A Physics-Space Grounded Agentic Framework for Generalizable and Interactive 3D Composition
PhySGA 让组装智能体负责粗规划，评判智能体利用接触反馈做细修正，再将经验写入技能库。在 Isaac Sim 中比较组合质量，适合关注物理场景构建的读者。
论文：https://arxiv.org/pdf/2610.20560
资源：https://github.com/AgenticEI/PhySGA
项目页：https://agenticei.github.io/PhySGA-page/
16机器人团队同时寻找多模态目标
HM3D-M3: A Benchmark for Multi-Agent, Multi-Modal, Multi-Object Goal Navigation
HM3D-M3 将多机器人、多模态和多目标放进同一导航回合，提供目标级成功率及团队效率指标。两类基线与异构真机团队演示，为协作导航提供统一参照。
论文：https://arxiv.org/pdf/2610.20121
代码：https://github.com/NJU-R-L-Group-Embodied-Lab/hm3d-m3-code
项目页：https://robo-lavira.github.io/hm3d-m3/
17会完成动作，却追错目标
What Happens to the Goal? Distinct Failures of Policy Alignment in Goal Misgeneralization
区分目标替代、策略捷径和线索竞争三种目标泛化失效，追踪目标信息如何影响决策。覆盖游戏环境与机器人任务，恢复措施的有效性取决于失效发生的层次。
论文：https://arxiv.org/pdf/2610.19731
代码：https://github.com/Julbarth/what-happens-to-the-goal
18检索经验时也要匹配动作与阶段
Robota-Embed: Unified VLA Embeddings for Robot In-Context Learning and Evolution
Robota-Embed 联合编码视觉、语言和动作，检索当前阶段可用的后续经验。除检索指标外，还评估策略上下文与局部记忆演化，须分清仿真成功率和真机任务进展。
论文：https://arxiv.org/pdf/2610.19495
项目页：https://robota-ai.github.io/robota-embed/
19为探索算法生成可复现的世界
Dungeon and Visibility World Generator
环境生成器可控制房间、通道和障碍，输出连续几何与离散占据表示，并用 Rust 引擎查询视线。适合探索、覆盖和多智能体实验，注意网格分辨率改变可见性。
论文：https://arxiv.org/pdf/2610.19523
代码：https://github.com/EdwinMeriaux/Dungeon_and_Visibility_World_Generator
20从策略模型走向机器人智能体系统
Towards Agentic Robots: Model, Data, Environment, and Harness
这篇综述按模型、数据、环境和系统编排四个领域梳理机器人智能体，将相关工作组织为十七类。适合作为技术地图阅读，配套仓库属于文献资源索引。
论文：https://arxiv.org/pdf/2610.20442
资源：https://github.com/chonghanqin/Awesome-Agentic-Robots
21双打乒乓球中的轮换与避碰
PingpongDuo: Agile Humanoid Table Tennis Doubles with Safe Team Coordination
PingpongDuo 用团队风险规划协调站位，将步法、击球和球拍残差分层训练。双机器人真机展示交替击球；回球增益与厘米级定位结果须区分仿真条件。
论文：https://arxiv.org/pdf/2610.20579
项目页：https://www.archon.tech/research/PingpongDuo
22从双视角视频还原人形操作参考
KinoGen: Customizable Humanoid Loco-Manipulation References via Kinematically Grounded Video Generation
KinoGen 以机器人和物体的标定渲染约束视频生成，再直接拟合机器人配置空间中的运动参考。覆盖八类移动操作并在 G1 部署跟踪策略，仍需重建与策略训练。
论文：https://arxiv.org/pdf/2610.20552
项目页：https://kinogen-humanoid.github.io/
23让 VLA 跟踪物体身份与状态
ObjDream: Guiding Vision-Language-Action Models with Object-Level Understanding and Future State Prediction
ObjDream 将物体身份和动态状态分开编码，以身份路由引导注意力，训练时预测未来物体状态。仿真及真机检验布局、杂乱和背景变化，推理时无需未来预测。
论文：https://arxiv.org/pdf/2610.20223
项目页：https://huailiangma.github.io/objdream/
24让示范采集者直接感受到接触力
FOMI: A Force-Feedback Manipulation Interface for Teaching Contact-Rich Manipulation
FOMI 用剪刀式机械结构将交互力传到操作者手部，支持脱离机器人采集及遥操作。四项接触任务比较采集效率，并用所得示范训练真机策略。
论文：https://arxiv.org/pdf/2610.19839
项目页：https://robbyant-research.github.io/FOMI/
25用一段人类视频引导新任务
One-shot Human Video-Guided Adaptation for Vision-Language-Action Models
CapPro 以阶段描述监督视频表征，将精选视觉 token 和任务查询接入 VLA。新任务适配无需目标任务机器人示范或微调，但依赖预先训练的视频条件模块。
论文：https://arxiv.org/pdf/2610.19746
项目页：https://chenqi20020226.github.io/cappro-project/
26直接生成可执行的人形策略权重
Modeling and Understanding Neural Weights of Humanoid Policies with Diffusion
用线性自编码器压缩策略参数，再由扩散 Transformer 生成权重；在 G1 上展示行走、舞蹈和旋踢。各动作分别建模，不能据此推断任意新技能生成。
论文：https://arxiv.org/pdf/2610.20606
项目页：https://humanoid-weight-diffusion.github.io/
27跨身体示范先对齐物体运动
Human Demonstrations as Post-Training Data for Any Robot Body: Retargeting the Object, Not the Hand
以物体位姿和接触区间建立不依赖人体结构的示范记录，再逐机器人检查可执行性。当前证据是仿真运动学筛选，论文明确尚未用这些记录训练策略。
论文：https://arxiv.org/pdf/2610.20351
项目页：https://playground.dxtr.co/
28触觉如何进入表示、策略与基础模型
Tactile-Guided Robot Learning: From Representation Learning to Foundation Models
这篇综述按触觉在学习和控制流程中的功能组织文献，串联表示学习、策略、预测控制与基础模型。重点讨论跨物体、传感器和形态泛化，适合建立研究地图。
论文：https://arxiv.org/pdf/2610.20134
项目页：https://sites.google.com/view/touch-in-robot-learning-review/home
29连续体机器人的驱动平台可复用
Design and Characterization of a Unified Continuum Robot Actuation Platform
提出解耦机器人本体与驱动软硬件的模块化平台，支持多自由度及串联机械臂组合。通过模块精度表征和两类手术场景演示验证，证据属于研究平台实验。
论文：https://arxiv.org/pdf/2610.19798
项目页：https://ucsdmorimotolab.github.io/unified-actuation/
30把看哪里与如何转头分开学习
Eye, Robot: Learning Expressive Gaze Dynamics via a VLM-Guided Transformer-VAE
异步视觉语言模型选择注视对象，反应式感知维持注视，再以 Transformer-VAE 生成眼头运动。当前评测支持运动学拟合，感知自然度仍待受控实验。
论文：https://arxiv.org/pdf/2610.19635
代码：https://anonymous.4open.science/r/eye_robot_pub-CF64/
31在永久占用约束下平滑分支管道
Topology-Preserving Multi-Usage Pipe Centerline Refinement under Permanent Tubular Occupancy
保持布管拓扑和端口不变，用共享主干与局部分叉控制调整中心线，再执行硬间隙验收。实验只优化合成布局的几何质量，尚不包含流体压降优化。
论文：https://arxiv.org/pdf/2610.19403
代码：https://github.com/bladesaber/MAPF_Pipeline
关注本号，持续跟进具身智能开源论文前沿动态。
整理不易，欢迎点赞、转发，留言交流。
