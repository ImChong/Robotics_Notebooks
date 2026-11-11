机器人动作块太长，什么时候该重新规划？｜附代码与项目入口
阅读路线
今天更新 53 篇具身智能新论文，覆盖操作策略与在线适配、触觉与世界模型、人形控制与导航、具身评测与协作安全等方向。为了方便快速浏览，本期按阅读优先级分成三档。
1 篇深读2 篇跟进50 篇扫读
先读 ExRL，关注长动作块的重规划时机；再看 LingBot-Touch 的跨传感器表征与 Embodied First Exam 的评测口径。其余论文按阅读优先级排列，可按自己的研究方向回看。
先看这一篇
1长动作块，执行多久再重规划
ExRL: Sample-Efficient Online Adaptation of Long-Chunk Policies via Execution Length Control
一句话判断：长动作块能摊薄推理成本，却容易错过中途纠偏；整套策略在线强化学习又代价高。ExRL 冻结基础策略，只学习执行长度，把重规划时机变成一个低维控制问题。它适合已有技能的部署适配，效果仍受基础策略能力约束。
你可以带走的两点
• 把执行长度作为离散动作，用含持续时间的回报与折扣训练适配器；先检查基础策略能否提出有效动作，长度控制无法凭空补齐技能。
• 作者在 Piper 上冻结 π0.5：叠块由固定执行 10 步的 9/20 提高到 17/20，插接由 10/25 提高到 22/25；均为适配后的终测结果。
方法亮点：把高维动作策略的在线优化压缩为何时重规划的决策，并在回报和折扣中显式处理不同执行时长；真机实验保持基础策略不变，检验这一控制接口本身的收益。
论文：https://arxiv.org/pdf/2611.13482
代码：https://github.com/Frankluox/ExRL
项目页：https://exrl-project.github.io/
另外 2 篇值得跟进
2换一种触觉传感器，表征能否复用
Learning Heterogeneous Tactile Representation for General Dexterous Manipulation
看点：不同触觉传感器的信号格式与接触几何差异很大。LingBot-Touch 用统一几何接口和共享编码器学习表征，为下游策略复用触觉经验提供路径，仍需任务适配。
结果：作者项目页报告覆盖 26 项真机任务、每项 20 次试验，并测试多类传感器；下游策略仍需后训练，冻结的是触觉编码器，不能理解为换传感器后直接零训练部署。
论文：https://arxiv.org/pdf/2611.12806
代码：https://github.com/Robbyant/lingbot-touch
项目页：https://technology.robbyant.com/lingbot-touch
3会写代码之后，还要通过机器人实操题
Embodied First Exam
看点：把编程智能体放进机器人任务环境，分开测量传感限制、实时性与特权信息的影响。它为具身代码提供可比较的考试框架，但题库规模与核心评测集需要分别理解。
结果：完整题库含 1156 项任务，主要实验使用 456 项核心任务、901 项测试；核心评测均在仿真中进行，每个智能体每项测试单次作答，成绩需连同信息权限和实时约束一起解读。
论文：https://arxiv.org/pdf/2611.12177
代码：https://github.com/mll-lab-nu/embodied-first-exam
项目页：https://www.embodied-first-exam.ai
50 篇一眼扫完
下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。
04从动作生成先验，走向人形闭环操作
Kimodo-Policy: Learning Kinematic Action Models for Humanoid Loco-Manipulation
冻结预训练动作生成器，再加入视觉与手部控制模块，把运动先验转成闭环运动学动作模型。包含仿真和真机；当前落地依赖指定全身跟踪器，跨控制器迁移仍需验证。
论文：https://arxiv.org/pdf/2611.11554
代码：https://github.com/kimodo-policy/kimodo-policy
项目页：https://kimodo-policy.github.io/
05把试错经验存进记忆，而非模型权重
World In-Context Learning for Agentic Robot Manipulation
用单次示范启动任务，在仿真排练中积累任务记忆，冻结模型也能改进操作。真实部署仍需接口与适配；“一次示范”包含后续交互经验，不能理解为零试错。
论文：https://arxiv.org/pdf/2611.12885
代码：https://github.com/ArcheOne-AI/RoboScript
项目页：https://ArcheOne-AI.github.io/RoboScript
06一句指令被误解，能否直接修正策略权重
Distilling Instructions into Weights: Closed-Form Edits Fix a Robot Policy’s Misreading
从有效改写中提取激活，以闭式权重编辑修补特定语言误读，无需梯度训练。仿真中改善明显；编辑针对的是可由改写修正的错误，不能概括为普遍修复操作失败。
论文：https://arxiv.org/pdf/2611.11894
代码：https://github.com/adeeb10abbas/weightsmith
项目页：https://ali-adeeb.com/weightsmith/
07人在旁边动起来，操作策略还可靠吗
HARP-Bench: Benchmarking Robot Manipulation Around Active Humans
把人的提示、协助、打断与共处加入操作环境，评测主动人类影响。基准以仿真人物为主，配有有限真机验证；任务成功率与人机交互安全仍需分别报告。
论文：https://arxiv.org/pdf/2611.11123
资源：https://github.com/UMass-Embodied-AGI/harp
项目页：https://harp-bench.vercel.app/
08只用单目视频，导航如何记住走过的路
Compass 1: Towards Open-World and Persistent Embodied Navigation from Monocular RGB Videos
通过历史地图与稀疏关键帧，把长时记忆用于持续导航和历史问答。跨形态仿真、真机展示值得关注；效果仍依赖记忆组织与任务设置，不能等同于任意环境自主运行。
论文：https://arxiv.org/pdf/2611.12087
资源：https://github.com/OpenDriveLab/Compass1
项目页：https://www.archon.tech/research/compass1
09多位任务专家，怎样合成一套 VLA
VLA-MOPD: On-Policy Distillation for Multi-Task VLA Policy Consolidation
让学生沿自身访问的状态向专家学习，用在线分布蒸馏合并技能，并约束原有能力退化。覆盖仿真与真机；复现成本还包括专家策略获取，不能只计算学生训练。
论文：https://arxiv.org/pdf/2611.12286
代码：https://github.com/yxuan1234/VLA-MOPD
10世界动作模型扩量，先守住跨模态对齐
SWAM: Scaling World Action Models with Cross-Modal Alignment
混合通用视频、第一视角与机器人数据，研究如何保留语言—视频对齐并迁移到控制。仿真和真机均有评测，复现时要同时关注语言标注质量、数据配比和来源许可。
论文：https://arxiv.org/pdf/2611.12870
代码：https://github.com/shaaaaron/SWAM
项目页：https://char-lotte-xia.github.io/swam-official/
11把世界模型的空间理解用于动作学习
ForeAct: Geometry-Guided Action Learning from World Representations
一次提取视频骨干表征，用未来相对深度作训练辅助，推理时移除该辅助头。仿真与机械臂实验支持几何信息价值；动作头仍用流匹配，不能把整套策略称为单步推理。
论文：https://arxiv.org/pdf/2611.11500
代码：https://github.com/Bulrush666/ForeAct
12可通行区域分割，需要记忆也需要几何
SAM-VTP: Video Memory and Measured Geometry for Continuous Traversability Segmentation
结合视频记忆与深度测得的坡度、粗糙度等信息，持续预测可通行区域。合成数据评测之外有实地导航；仍依赖深度质量、机器人通行尺度，不能仅凭语义判断能否通过。
论文：https://arxiv.org/pdf/2611.12099
代码：https://github.com/lucashoefs/sam_vtp
13检索旧示范，在不改权重时引导新动作
Zero-Shot VLA Adaptation with Retrieved Guidance
检索相近示范，用语义与几何对应关系对齐，再在生成过程中加入动作引导。仿真和真机展示免权重更新的适配；效果仍依赖相关示范库与跨场景匹配质量。
论文：https://arxiv.org/pdf/2611.10894
资源：https://huggingface.co/datasets/raguvla/real-world-evaluation
项目页：https://raguvla.github.io/
14让系统辨识留下可检查的推理证据
ASPIRE: An Agentic System-Identification Platform for Iterative Reasoning and Exploration
把模型结构搜索、参数拟合与物理约束核验串成智能体流程，并封存测试集。实测系统验证了可执行性，也暴露推理理由缺乏证据的问题；适合研究可审计的建模过程。
论文：https://arxiv.org/pdf/2611.13114
代码：https://github.com/erik-derner/aspire-sysid
15让潜在世界模型保留与动作有关的信息
Σ-LeWM: Learning World Models via S-Information for Robot Control
在前向预测之外加入反向与逆动力学训练信号，让潜在状态更适合控制。规划时仍使用前向模型；结果来自仿真控制基准，尚不能直接推及真实接触与传感噪声。
论文：https://arxiv.org/pdf/2611.12732
代码：https://github.com/zuxinrui/sigma-lewm
项目页：https://zuxinrui.github.io/sigma-lewm/
16已知旋转组合规则，就把它写进模型
LawKAN: Group-Law Layers for Composition—When Imposing a Known Law Pays Off
把已知群运算法则直接嵌入组合层，学习局部增量与修正，减少模型重复学习结构的负担。包含机械臂校准实验；收益依赖法则成立，错误的结构先验也会拖累结果。
论文：https://arxiv.org/pdf/2611.12567
代码：https://github.com/occamolab/lawkan
17视觉策略异步运行，怎样让动作接得更顺
Control-Aware Command Generation for Asynchronous Control and Action Inference with Vision-Based Policies towards Fast Execution
用控制感知指令生成平滑异步更新的目标，再利用不确定性调整部分动作跳步。真机操作表明推理与控制解耦值得优化；跳步机制的验证范围主要针对 ACT。
论文：https://arxiv.org/pdf/2611.11089
代码：https://github.com/TakaOsa/ACT_Async
18适配 VLA，验证损失未必选出最会动的模型
VLA Adaptation with Real and Simulated Data: A Case Study on Robotic Chess
围绕棋子移动比较真实、仿真数据与训练设置，提醒读者：离线损失和真机成功率可能不同步。学习率调度与检查点选择也会改变结论；任务范围仍是特定棋盘操作。
论文：https://arxiv.org/pdf/2611.12676
代码：https://github.com/kripner/vla-chess
19把第一视角视频和高精度全身动作对齐
RoboMoCap: A Large-Scale Whole-Body Motion Capture Dataset with Egocentric Vision for Robot Learning
同步第一视角视频、腕部视角与光学动作捕捉，为全身运动重定向和学习提供数据。采集包含下肢接触细节；主体仍是受控实验空间，不能等同于自然环境日常数据。
论文：https://arxiv.org/pdf/2611.10994
资源：https://huggingface.co/datasets/BitRobot/robomocap
项目页：https://bitrobot-foundation.github.io/robomocap/
20世界模型候选更多，规划就一定更好吗
How Many Candidates Does a One-Step World Model Need? A Random-Codebook Account of Nearest-of-m Training on Frozen DINO Tokens
分析多候选预测中的最近样本训练，解释随机码本如何覆盖多种未来。实验同时给出反例：预测指标改善未必转化为规划收益，候选数量也不存在跨任务通用最优值。
论文：https://arxiv.org/pdf/2611.11887
代码：https://github.com/thddydgnl/nearest-of-m-world-models
21机器人从环境状态中提出下一项任务
CAFE: Closed-Loop Autonomous Task Formulation and Execution without Instance-Specific Instructions
通过语义记忆与任务准入检查，在预设任务类别内主动发现待办，再把规划、执行和复核接成闭环。仿真之外给出真机轨迹，但任务自主性仍受预定义类别约束。
论文：https://arxiv.org/pdf/2611.12983
代码：https://github.com/submissionforpaper/CAFE-Closed-Loop-Autonomous-Task-Formulation-and-Execution-without-Instance-Specific-Instructions
项目页：https://submissionforpaper.github.io/CAFE-Closed-Loop-Autonomous-Task-Formulation-and-Execution-without-Instance-Specific-Instructions/
22自然语言车队调度，类型正确还不够
Correctness Boundaries of Typed Plans for Natural-Language Robot Fleet Control
把语言指令转成类型化计划，保留修正、执行与结果证据链。多机器人仿真揭示了检查器的能力边界：格式合规不能替代意图正确，抵达区域也不代表完成真实搬运。
论文：https://arxiv.org/pdf/2611.12654
代码：https://github.com/Innopolis-Robotics-Society/llm_swarm
项目页：https://innopolis-robotics-society.github.io/llm_swarm/
23边听边跳，人形机器人提前接住节拍
AHEADance: Anticipatory Humanoid Dance from Streaming Audio
用音乐预测补偿感知与动作延迟，再从已经承诺执行的动作接续生成舞姿。作者展示真机舞蹈；节拍指标主要评估生成参考动作，尚未形成机器人状态反馈闭环。
论文：https://arxiv.org/pdf/2611.13370
项目页：https://ihcr.github.io/AHEADance/
24从网络视频联合预测手与物体的未来
Grasping the Future: Joint 3D Hand–Object Forecasting From Internet Videos
用流匹配联合预测手—物体三维运动，显式建模接触关系，并探索机器人动作重定向。训练依赖自动恢复的三维监督；预测与重定向结果尚不等于闭环操作成功。
论文：https://arxiv.org/pdf/2611.12879
项目页：https://grasping-the-future.github.io/
25跨实验室成功率，先统一任务怎么抽样
Benchmarking Real-World Robotic Manipulation through Stochastic Challenges
用随机种子生成任务分布，让不同实验室重建可比较的操作挑战。两类机器人、两个实验室提供实例；仍需区分策略差异、任务抽样波动与人工重建带来的偏差。
论文：https://arxiv.org/pdf/2611.12804
项目页：https://rpl-bonn.github.io/stochastic-challenges/
26双臂规划，把可执行条件带进每一步
Bridging Task Planning and Execution for Language-Conditioned Bimanual Manipulation
把场景语义、可行性约束和阶段前后条件带入双臂任务规划，执行失败时局部修复。真机桌面任务展示了闭环价值；机器人基座保持固定，不能外推为全身移动操作。
论文：https://arxiv.org/pdf/2611.12409
项目页：https://advrhumanoids.github.io/bridging-planning-execution/
27让压力直接变成颜色，供操作策略读取
PCTac: Mechanochromic Photonic-Crystal Tactile Sensor for Dexterous Manipulation
利用光子晶体受压变色，把接触压力映射到色相信息并输入操作策略。三类真机任务比较视觉与触觉输入；材料响应、凹面接触灵敏度仍会限制泛化。
论文：https://arxiv.org/pdf/2611.12383
项目页：https://bey0ndt1me.github.io/PCTac-page/
28长程人形导航，让规划与控制各按节奏运行
LHNav: Long-Horizon Humanoid Navigation via Multirate Coordination and Physics-Grounded Action Data
把高层规划、动作生成和底层跟踪拆成多速率协同系统，用物理可执行轨迹监督动作。包含真机导航；多目标长程评测与未见场景的单目标评测，需分开看待。
论文：https://arxiv.org/pdf/2611.12136
资源：https://github.com/robbyant-research/LHNav/tree/main
项目页：https://robbyant-research.github.io/LHNav/
29仅看外观不够，让 VLA 感知物体冷热
Thermo-VLA: A Thermal-Aware Vision-Language-Action Model with Gated Residual Modulation for Robotic Manipulation
通过语言门控的热成像残差，在需要时把温度线索注入视觉动作模型。真机冷热相关任务展示增益；温度表征主要区分冷热，常规视觉任务的能力保持仍有待改善。
论文：https://arxiv.org/pdf/2611.11981
项目页：https://j-x-chen.github.io/Thermo-VLA-web/
30机器人翻物体，让代码智能体在试错中修策略
Flip Anything with Agentic Robotics Self-Learning
由编程智能体修改分阶段翻转策略，利用仿真反馈处理形状差异和失败恢复。真机演示提供迁移线索；大幅成功率提升来自仿真评估，不能直接套到真实物体上。
论文：https://arxiv.org/pdf/2611.11941
项目页：https://berkeleyautomation.github.io/flipanything/
31手术视频三维追踪，先把接触关系保留下来
SurgTAP-3D: A Large-Scale Cross-Surgery Video Dataset with Contact-Aware 3D Motion Tracking
从手术视频中恢复接触感知三维轨迹，把二维跟踪、几何优化与时序一致性结合起来。数据适合研究运动表征；恢复结果属于伪标签，不能当作临床精度的度量真值。
论文：https://arxiv.org/pdf/2611.11697
项目页：https://wrld.github.io/surgtap-3d/
32部署后的失败监测，持续更新最后一层
Streaming Failure Forecasting: VLA Failure Monitors That Keep Learning After Deployment
冻结监测网络主体，用新增回合标签重拟合线性读出层，让失败预警随数据更新。真实记录支持回放评估，但它不是持续在线真机实验，报警阈值更新也有条件限制。
论文：https://arxiv.org/pdf/2611.11688
项目页：https://sff-research.org/
33物体被遮挡以后，世界模型还要预测整体运动
ObjectWorld: Scaling Object-Centric World Models with A Large-Scale 3D Full-Object Flow Dataset
以完整物体三维流描述刚体和形变物体运动，包含不可见部分，再连接到机器人控制。真实与仿真数据共同扩量；自动标注的几何质量仍是使用这类数据的关键条件。
论文：https://arxiv.org/pdf/2611.11085
项目页：https://nus-lins-lab.github.io/ObjectWorld/
34何时请人接管，要分清换阶段与真失败
MoE-DAgger: Transition-Aware Router-Entropy Gating for Selective Human Correction in Phase-Structured Manipulation
结合路由熵和阶段历史，区分正常阶段切换与危险回退，选择人工纠正时机。同等人工控制预算下做真机比较；目前主要验证两种相近货架推送任务。
论文：https://arxiv.org/pdf/2611.10825
项目页：https://sj824.github.io/MoE-DAgger/
35简单状态少算几步，复杂状态再多想一会
Learning When to Think: Adaptive Internal Computation for Reinforcement Learning
把内部计算深度也纳入策略学习，联合决定何时停止推演和采取什么动作。多个仿真基准显示计算节省，但部分任务回报下降，速度与控制质量仍需按任务权衡。
论文：https://arxiv.org/pdf/2611.10822
项目页：https://rohitsalla.github.io/aic-neurips2026/
36编程智能体接上机械臂，风险会怎样变化
When Coding Agents Get Arms: Safety Risks Across the Digital-Physical Boundary
把代码智能体放到具身任务中，检查误用、指令注入和异常环境下的物理行为风险。配合少量真机案例提示：数字权限与动作权限需要共同审视，测试通过也非完整安全保证。
论文：https://arxiv.org/pdf/2611.11438
项目页：https://ymsun2020.github.io/CodingAgentEmbodiedSafety/
37多设备共享世界状态，证据链也要一起评测
A Benchmark and Layered Evaluation Protocol for Evidence-Grounded Shared Belief Updating in Connected Robotics
用分层协议检查共享信念更新，沿观测前缀追踪证据、关系与最终判断。基准强调关系信息的作用；当前主要是合成单实体二元状态，尚不代表真实机器人群部署效果。
论文：https://arxiv.org/pdf/2611.11842
资源：https://creg-bench.github.io/download/
项目页：https://creg-bench.github.io/
38动作看起来要失败时，再触发仿真搜索
Look Ahead: Search-Driven Reasoning in Embodied AI
冻结动作策略，用复制的仿真状态搜索候选动作块，并在预判变差时才启动前瞻搜索。证据来自推物仿真与较弱基线；需要模拟器状态，尚未验证语言 VLA 或真机。
论文：https://arxiv.org/pdf/2611.10769
代码：https://github.com/AhmedZeer/lerobot-tree/tree/main/examples/tree_search/pusht
项目页：https://azeer.co/research/lookahead/
39越野车倾覆预警，把不同风险信号分开学
Hierarchical Multi-Channel Tip-Over Prediction via Isolated Temporal Models for Planetary Rover Safety
隔离不同时间通道建模，再融合规则、时序预测与残差检测，争取倾覆预警提前量。基于地面试验记录评估；识别风险事件不等于已验证闭环避险或行星现场部署。
论文：https://arxiv.org/pdf/2611.12261
资源：https://zenodo.org/records/21258126
40大范围三维探索，把精细地图留在近处
Beyond Memory Limits: A Hierarchical Spatio-Temporal Map Representation for Large-Scale 3D Autonomous Exploration
用局部稠密地图和全局关键帧结构分工，压缩长期探索所需的地图记忆。包含大型仿真及实地测试；固定的是局部地图规模，全局信息仍会随探索增长。
论文：https://arxiv.org/pdf/2611.11489
项目页：https://yizhezhang0418.github.io/flavor.github.io/
41灵巧操作，把未来空间变化蒸馏进策略
DexForesight: Grounding Dexterous Policies with Implicit Spatial Foresight
先学习手部三维运动与视觉变化预测，再将空间前瞻表征对齐到策略。训练用深度和运动学，执行只需图像与状态；真机验证规模有限，跨任务泛化仍需继续检验。
论文：https://arxiv.org/pdf/2611.11166
项目页：https://dexforesight.github.io/
42用激光几何约束，让高斯表示更贴近表面
PLS-GS: Probabilistic Lidar Scaffolds for Compact, Surface-Aligned Gaussian Splatting
以激光点云的概率支架约束高斯位置和方向，用表面结构先验压缩场景表示。户外数据验证了紧凑性，但依赖配准点云，渲染质量仍与更自由的表示存在取舍。
论文：https://arxiv.org/pdf/2611.10699
代码：https://github.com/k-harikumar/PLS-GS
43窄走廊导航，把定位和避障接到一起
Maze Runner: Autonomous Maze Navigation with Line-Based Localization and Control Barrier Functions
以激光线特征统一定位与建图，再结合路径规划、非线性预测控制和控制屏障函数。验证集中在仿真迷宫；遇到阻挡时能安全停下，仍不等于保证继续通行。
论文：https://arxiv.org/pdf/2611.12949
代码：https://github.com/alvgaona/maze-runner
44可穿戴机器人，把传感与助力模块解耦
ExoStack: A Modular and Reconfigurable Software Framework for Wearable Robotics Using ROS 2
以 ROS 2 与 micro-ROS 组织感知、步态估计和助力控制，强调配置复用与嵌入式连接。多平台时序和外骨骼实例支持工程可用性，尚不构成临床疗效证据。
论文：https://arxiv.org/pdf/2611.12306
代码：https://github.com/debadrata/ExoStack
项目页：https://debadrata.github.io/ExoStack_documentation/
45生成动作反复修改，也需要版本和验收记录
Kinora: A Versioned Workflow for Text-to-Motion Generation, Robot Retargeting, and Interactive Validation
将文本生成、动作重定向和交互验证组织成可追溯版本流程，保存修改与验收结果。价值主要在工具链管理；回归测试通过不等于动作语义更准确或真机执行更安全。
论文：https://arxiv.org/pdf/2611.11601
代码：https://github.com/PoirotJin/kinora
46多机器人搜救，把感知误差带进协同决策
PA-SaR: Perception-Aware Learning, Control, and Coordination for Multi-Robot Search and Rescue
把局部控制、探索分配与目标线索结合，显式考虑感知不确定性。仿真比较说明救援数量、覆盖率和碰撞应分别衡量；更多发现并不自动意味着更低伤亡。
论文：https://arxiv.org/pdf/2611.11648
代码：https://github.com/svenschoonebeek/Perception-Aware-Search-and-Rescue
47只靠局部本体感知，估计软体整体运动
DeepSoMo: Deformable Object Motion Estimation under Proprioceptive Partial Observability via Predictive Latent Dynamics
用潜在动力学预测作为辅助训练，让有限内部传感更能反映全局位姿、形变与接触。评估来自有限元仿真；真实材料变化和传感噪声下的迁移仍需硬件实验。
论文：https://arxiv.org/pdf/2611.11242
项目页：https://ai4ce.github.io/DeepSoMo
48生成的人类动作，怎样变成机器人学习数据
DreamVLA: From Generated Human Motions to Humanoid Vision-Language-Action Policies
用生成动作引导强化学习专家，再从学生失败状态补采纠正轨迹，训练人形视觉动作策略。评估集中在仿真任务，专家奖励与轨迹筛选仍包含任务相关设计。
论文：https://arxiv.org/pdf/2611.10872
项目页：https://ember-center-berkeley.github.io/DreamVLA/
49静脉压缩仿真，速度之外还要看误差
Simulating Vein Compressibility for Ultrasound-Guided Femoral Access: Speed, Accuracy, and Phantom Validation
在 GPU 上模拟软组织与血管受压，并用实体仿体校准压缩响应。工作连接了仿真性能和形变误差；验证仍是低保真仿体，不含血流、脉动等完整生理条件。
论文：https://arxiv.org/pdf/2611.12818
代码：https://github.com/MRDLab/femoral-sim-calibration
50同一状态多采几种动作，保留示范多样性
Explicitly capturing demonstrator multi-modality for distillation into diffusion policies
围绕同一观测重复采集专家动作，用分叉采样显式保存可行行为的多模态。合成任务和二维卸载实验支持其价值；目前证据尚未覆盖真实机械臂的复杂接触操作。
论文：https://arxiv.org/pdf/2611.10949
项目页：https://shaoyuan.cc/research/vine/
51从顶视相机起步，搭建小车编队实验
Tutorial: Vision-Based Formation Control with Differential-Drive Robots
用顶视相机识别标记，计算相对位置并生成差速轮命令，串起视觉编队控制的基础流程。真机教程适合教学与复现入门；像素坐标控制和基础避障不提供完备安全保证。
论文：https://arxiv.org/pdf/2611.11164
代码：https://github.com/SMARTlab-IU/formation-control-tutorial
52把 ROS 2 轨迹搬进混合现实预览
ROS2MR: Connecting ROS 2 Motion Planning to Mixed-Reality Trajectory Visualization
将机器人轨迹转成统一文件，再按时间重采样供头显显示，保留关节与夹爪时序。数值测试验证转换一致性；尚未充分测量真机同步延迟或操作人员使用收益。
论文：https://arxiv.org/pdf/2611.11830
项目页：https://mr-lab.se.cit.tum.de/
53量子策略到机器人动作，中间链路如何追溯
Qubit-to-Motion: Traceable Action Provenance in Hybrid QRL for Robotic Bin-Packing Sequencing
把量子观测、动作选择与装箱执行串成可验证来源记录，检查策略输出如何影响物理动作。验证基于确定性仿真场景；核心贡献是可追溯性，尚无量子优势或真机证据。
论文：https://arxiv.org/pdf/2611.11580
代码：https://anonymous.4open.science/r/Qubit-to-Motion-Quantum-Reinforcement-Learning-F7DC
关注本号，持续跟进具身智能开源论文前沿动态。
整理不易，欢迎点赞、转发，留言交流。
