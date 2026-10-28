# 人类视频怎样变成机器人能用的动作经验？

> 原始抓取（2026-10-28，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/JSya9kmE3Fcon3BY_DcCVA

人类视频怎样变成机器人能用的动作经验？
具身智能小站 具身智能小站
 Oct 28, 2026, 6:00 AM

阅读路线

今天更新 59 篇具身智能新论文，覆盖视觉语言动作与世界模型、灵巧操作与接触控制、自主导航与多机协同、数据采集与评测等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进56 篇扫读

先深读免渲染事件仿真与机载部署，再跟进 RynnVLA-Latent 的跨身体潜在动作预训练，以及 AuRA 的自主交互数据采集；其余按兴趣浏览方法与复现入口。

先看这一篇

1让微型无人机绕开事件仿真瓶颈

Rendering-Free Event Frame Simulation for Fast Reinforcement Learning of Autonomous Robot Navigation

 

一句话判断：事件相机适合低功耗、低延迟机器人，但先渲染图像再转换事件的仿真流程会拖慢强化学习。本文从运动边缘直接生成事件观测，把训练吞吐与微控制器部署连起来，值得关注的是整条工程路径，以及近似传感模型能否支持真实迁移。

你可以带走的两点

• 生成事件，无需先生成图像：用场景边缘几何与相机运动构造事件计数观测，再训练闭环避障策略；这是针对控制需求设计的传感近似，不能当作完整事件相机物理模型。

• 迁移要分层验证：先检查向基于渲染的物理事件仿真的迁移，再量化网络并进行机载飞行。真机展示使用特定传感器与微控制器，复现应逐项对齐观测窗口、量化和控制接口。

方法亮点：把闭环事件观测本身做成可并行解析近似，而非仅优化渲染或用特权状态替代事件输入；再将传感近似、策略训练与机载部署放在同一验证链条中。

论文：https://arxiv.org/pdf/2610.37543

代码：https://github.com/tudelft/rendering-free-event-nav

另外 2 篇值得跟进

2把异构视频转成三维潜在动作监督

RynnVLA-Latent: Scaling Robotic Manipulation Foundation Models with 3D-Grounded Latent Actions

 

看点：不同机器人的动作接口难统一，人类视频又缺少控制标签。RynnVLA-Latent 用带三维约束的潜在动作组织异构视频，为跨身体预训练提供共享监督，再适配目标机器人。

结果：先区分场景运动与相机运动：未来特征预测、三维动态流与相机运动解耦共同约束潜在动作，减少把视角变化当成动作的风险。

论文：https://arxiv.org/pdf/2610.34356

代码：https://github.com/alibaba-damo-academy/RynnVLA-Latent

项目页：https://alibaba-damo-academy.github.io/RynnVLA-Latent.github.io

3让机器人自己采集可变形物体数据

AuRA: From Atoms to Bits through Autonomous Robot Interaction

 

看点：数字孪生重建常被逐物体人工交互与标注拖慢。AuRA 把可执行抓取、机器人状态和交互前场景先验接起来，让机器人自主采集可变形物体交互，直接对应资产建设成本。

结果：用机器人状态给出夹爪运动：减少视觉跟踪遮挡与误差，再以交互前几何、外观先验补足被机械臂挡住的信息。

论文：https://arxiv.org/pdf/2610.34452

代码：https://github.com/Yoonkyo/AuRA

项目页：https://aura-robot.github.io/

56 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04让机器人通过操作获得新知识

EmbodiedLab: Benchmarking Embodied Agents That Act to Know

以物理实验室为评测场，智能体需自主选择实验步骤、操作仪器并从读数得出结论。区分知道答案与能够完成实验，实验参数重采样，避免把记忆中的结果当成发现。

论文：https://arxiv.org/pdf/2610.35385

代码：https://github.com/HexaFuture/EmbodiedLab

05让低成本机械臂也有空间评测

SO-101 Bench: Accessible Spatial Manipulation

为 SO-101 建立数字孪生与语言条件空间操作基准，区分见过对象、未见实例和未见类别。部分策略的仿真与真机排序一致，但仿真仍高估成功率，不能直接替代实测。

论文：https://arxiv.org/pdf/2610.36420

代码：https://github.com/5hadytru/so101_bench

项目页：https://so101-bench.github.io/

06用时序规范补足成功率评测

Specola: A Tool for Specification Driven Robot Evaluation

将信号时序逻辑接入仿真执行过程，除完成与否外检查满足余量、持续性和约束违反。提供多种环境适配器，适合为策略增加可解释的过程指标与回归验收条件。

论文：https://arxiv.org/pdf/2610.35730

代码：https://github.com/specola-eval/specola

07开放家庭环境需要综合自主性评测

Towards Open-world Robot Autonomy

构建家庭布局与分层任务，覆盖任务发现、上下文推理、操作、恢复和鲁棒性，并以开放场景检验组合能力。帮助区分单项任务成功与持续满足高层需求的自主执行。

论文：https://arxiv.org/pdf/2610.34649

代码：https://github.com/AetherLabsAI/RobotAgentBench

项目页：https://huggingface.co/AetherLabs-AI/InternVLA-A1.5-RAB

08让灵巧手在追踪中持续修正参考

AdaDexTrack: Dynamic Modulation for Adaptive and Generalizable Dexterous Manipulation Tracking

先蒸馏通用灵巧追踪器，再用闭环调制器持续调整参考轨迹、物体表征和位置目标。在未见物体、轨迹及真机迁移中考察鲁棒性，核心是避免固定参考让接触误差逐步累积。

论文：https://arxiv.org/pdf/2610.33852

代码：https://github.com/Janebek/AdaDexTrack

项目页：https://janebek.github.io/AdaDexTrack/

09多一根拇指，辅助单手操作

EgoThumb: A Vision-Driven Supernumerary Robotic Thumb for Autonomous Grasp and Holding Assistance

用掌心相机驱动额外机械拇指，把手与物体的视觉线索映射到关节位置。前臂驱动减轻末端负担，提供抓握、保持与释放的辅助思路，复现仍需配套硬件与训练。

论文：https://arxiv.org/pdf/2610.37582

代码：https://github.com/JayeFu/ego_thumb_firmware

项目页：https://ego-thumb.github.io/

10无人机穿越移动窄缝

Learning Visuomotor Quadrotor Flight Through Dynamic Narrow Gaps via Differentiable Simulation

将穿缝时机、对齐与碰撞写成可微目标，从随机初始化直接学习视觉惯性控制。作者报告真机十次试飞成功八次，适用范围仍受测试场景与样本量限制。

论文：https://arxiv.org/pdf/2610.37557

代码：https://github.com/uzh-rpg/agile_narrow_gap_diffsim

11感知模型调度提前考虑芯片升温

Thermal-Headroom Routing for Efficient Robot Perception in Extreme Heat

用热模型预测温度，在选取检测器规模和分辨率前避开热降频区间。比较多温度与真实输入轨迹；已知固定环境下，调好的温控阈值也可能匹配其效果，优势有条件。

论文：https://arxiv.org/pdf/2610.35231

代码：https://github.com/ImSpxrsh/thermal-headroom-routing

12执行日志会左右视觉裁判

Execution Logs Sway VLM Judges of Robot Progress

在相同机器人视频旁加入相反进展声明，检验视觉语言裁判对文字的依赖。一句日志即可改变部分判断，提醒评测时隔离未验证的执行叙述，并测试提示警告的缓解效果。

论文：https://arxiv.org/pdf/2610.34151

代码：https://github.com/sunnyadn/exec-logs-robot-judges

13判断机器人是否完成任务

RoboJev: An Open Vision-Language Judge for Robot Success Detection

以任务指令、操作视频抽帧和可选辅助视角判断执行成功，用低秩适配训练视觉语言裁判。提供成功检测研究入口，使用时仍需确认新场景的标签定义、失败类型与泛化。

论文：https://arxiv.org/pdf/2610.33718

项目页：https://huggingface.co/josefchen/RoboJev-27B

14世界模型能否复现策略怎样失败

WorldFailBench: Do World Models Understand Policy Failures?

从失败原因、发生阶段和成功失败边界三个维度评测，而非只看最终输赢。覆盖仿真操作与人形真机任务，揭示生成环境用于策略诊断时必须额外检查的可靠性缺口。

论文：https://arxiv.org/pdf/2610.33565

代码：https://github.com/DAXIAORobotics/WorldFailBench

项目页：https://huggingface.co/datasets/ACERobotics/WorldFailBench

15低比特压缩需要保住动作敏感权重

AGQ: Action-Guided Ultra-Low-Bit Quantization for Vision-Language-Action Models

联合动作敏感度与权重恢复开销分配矩阵位宽，辅以校准、无损编码和分块恢复。LIBERO 结果显示压缩率与成功率明显取舍；权重存储减少不能直接解释为推理加速。

论文：https://arxiv.org/pdf/2610.33074

代码：https://github.com/Qiuxu00/AGQ

16不改权重，利用执行经验修正决策

RoboDoctor: Training-Free Recursive Self-Improvement for Robot Policies

围绕控制权分配、干预时机与操作方法记录诊断、尝试和结果，支持固定权重下的执行调整。现有任务档案显示可行性，但跨回合经验的独立贡献仍需受控实验。

论文：https://arxiv.org/pdf/2610.35604

代码：https://github.com/THUSI-Lab/RoboDoctor

项目页：https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robodojo

17让人形机器人同时跟踪运动与力

Learning Force-Reactive Humanoid Loco-Manipulation

用带力传感的手持夹爪与身体追踪器采集示范，策略联合预测动作和双手力命令。混合柔顺全身控制兼顾受力与平衡，真机比较强调同示范、同控制器下的力反馈价值。

论文：https://arxiv.org/pdf/2610.35862

项目页：https://louis-leee.github.io/FLoRe/

18用仿真教会视觉语言模型执行动作

Pretraining Knows What, Simulation Teaches How: Compositional Generalization in a Sim-Trained Robot Agent

先用预训练模型提供理解与推理，再通过仿真监督微调和强化学习学习工具控制。作者报告真实任务迁移与新目标组合能力，结论依赖其工具接口、任务集和评测预算。

论文：https://arxiv.org/pdf/2610.35817

项目页：https://zero-real.github.io/

19用统一潜在动作驱动世界模型

RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions

以三维落地潜在动作对齐人类、机器人与仿真视频，学习可控未来，并通过数字遥操作和动作迁移合成示范。生成数据可用于策略中间训练，但目标机器人仍需后训练适配。

论文：https://arxiv.org/pdf/2610.36791

代码：https://github.com/alibaba-damo-academy/RynnWorld-Latent

项目页：https://alibaba-damo-academy.github.io/RynnWorld-Latent.github.io

20用三维重建生成腕部相机到达数据

Robot-Aligned 3D Gaussian Splatting for Precise Reaching with Wrist-Camera VLA Policies

把真实工作区重建为高斯场景，对齐机器人坐标后渲染腕部视角，并配对平滑到达轨迹。以主板场景验证接触前定位，还加入到达校验器；结果不代表完整拆卸能力。

论文：https://arxiv.org/pdf/2610.35626

代码：https://github.com/001-Wang/PrecisionReach-GS

项目页：https://001-wang.github.io/PrecisionReach-GS/

21第一视角经验支持人形全身策略

HuGe: Scaling Humanoid Generalist Vision-Language-Action Models with Egocentric Experience

将第一视角人类与机器人数据用于分阶段训练，预测全身控制器的潜在运动标记和手部信号。通过仿真回放补本体状态、重渲染调整外观，缓解跨身体的观测差异。

论文：https://arxiv.org/pdf/2610.34186

代码：https://github.com/yliu-cs/HuGe

22异构机器人各自控制，共同搬运

HA-DiPO: Diffusion Policy Optimization for Heterogeneous Cooperative Transport

将团队协调与各身体的运动控制分层，以扩散策略表达协同动作分布。执行依赖本地机器人和载荷观测，无需伙伴状态或通信；通过仿真、异构真机和人机搬运展示。

论文：https://arxiv.org/pdf/2610.33584

代码：https://github.com/ha-dipo/ha-dipo-sim2sim

项目页：https://ha-dipo.github.io/

23把动作当作视频模型的补丁

Tapestry: A Single Backbone for Video Generation and Robot Control

将动作投影为视频潜空间中的单补丁标记，用同一骨干学习视频、联合视频动作及纯动作扩散。执行时只生成动作序列，适合研究视频先验复用与推理成本取舍。

论文：https://arxiv.org/pdf/2610.36823

项目页：https://tapestry-wam.github.io/

24手持采集的瓶颈在定位精度

OpenJaws: An Open Hardware UMI Framework with Stereo Localization for Generalist Manipulation

用低成本双目相机恢复有尺度的手持轨迹，配合模块化夹爪减少采集到部署的接口变化。定位基准同时比较误差、覆盖率与耗时，提醒读者不要只看轨迹误差。

论文：https://arxiv.org/pdf/2610.36598

项目页：https://open-jaws.github.io/

25看清通用智能体与专用策略的分工

Profiling a Frontier Agent with EBench: From Direct Control to Harness with Policy

在统一视觉和本体观测协议下，比较直接控制与调用专用策略两种配置。混合执行能互补，但调用时机、子目标缺失和重新接管造成的抓握破坏仍限制整体表现。

论文：https://arxiv.org/pdf/2610.35832

项目页：https://internrobotics.github.io/EBench-Astra-Tech-Report/

26识别危险物体还不等于安全绕行

Benchmarking Obstacle-Conditioned Safety in Household Robot Navigation

用空间匹配、仅障碍类别不同的场景评测安全语义能否转为间距、速度和平滑性变化。模型即使认得风险，也常未改变运动；显式参数推断为语义落地提供了改进线索。

论文：https://arxiv.org/pdf/2610.34674

项目页：https://huggingface.co/ocsnav/groot-vlm-graft28

27让人类视频提供动作监督

ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning

把动作推理写成离散运动标记组成的动力学链，再用未来预测与真实状态转移的一致性强化训练。在仿真与真机中考察跨任务泛化，仍需机器人示范初始化。

论文：https://arxiv.org/pdf/2610.37666

项目页：https://thinkworld-project.github.io/

28重定向人手动作时保住接触力

CARD: Contact-Anchored Retargeting for Dexterous Manipulation

把视频、动作捕捉和触觉手套统一为手与接触表示，重定向时保留接触力目标，并用触觉控制追踪。重点从几何相似转向接触保持；视频输入还需估计接触信息。

论文：https://arxiv.org/pdf/2610.36539

项目页：https://card-retargeting.github.io/

29多机械臂规划利用 GPU 并行

cuArms: GPU-Accelerated Conflict-Based Planning for Multi-Arm Manipulation

复用单臂多模式路线图，以延迟冲突搜索处理多臂碰撞与死锁，再联合任务分配和调度。给出仿真与工业真机示例，适合关注共享工作站吞吐与规划扩展性的团队。

论文：https://arxiv.org/pdf/2610.36250

项目页：https://cuarms.github.io/

30在仿真中复现运动诱发无线感知

Enabling Motion-Induced WiFi-CSI Sensing in Multi-Robot Systems: A Simulation Framework

把 WiFi 信道状态与机器人运动结合，在 Gazebo 中生成信号，支持虚拟天线阵列方位估计。以 TurtleBot3 仿真和实测对照，降低无线定位实验门槛。

论文：https://arxiv.org/pdf/2610.34034

代码：https://github.com/baharhaghighat/DARS_2026_Dijkstra_et_al

31让行人轨迹预测贴近真实输入

Real-Time Human Trajectory Prediction from RGB-D for Mobile Robots Using Bi-Mamba

统一训练与部署中的姿态检测、深度反投影和平滑流程，再用双向 Mamba 与解码器预测短期速度。部署于移动机器人，重点是减少理想训练数据与噪声传感输入的差距。

论文：https://arxiv.org/pdf/2610.33916

代码：https://github.com/basheeraltawil/NIT-TrajMamba

32从旧示范中检索几何纠偏线索

Retrieve to Act: Motion Primitive Graph Retrieval for Robust VLA Execution

把成功示范构成时空运动基元图，推理时检索兼容轨迹，并用逐维门控与冻结策略融合。跨多种视觉语言动作模型考察扰动鲁棒性，基础策略无需微调，检索模块仍需学习。

论文：https://arxiv.org/pdf/2610.37215

项目页：https://retrieve2act.github.io/r2a/

33让程序搜索主动检验物理假设

RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding

围绕物体、场景与交互建立分层知识和场景落地机制，通过自设计实验检查阶段效果。跨多个操作基准评估，启发是把执行反馈变成可验证的物理假设，而非只改代码。

论文：https://arxiv.org/pdf/2610.36533

项目页：https://roboauto-ai.github.io/

34像人的行为未必最适合训练机器人

Behavioural Quality Is Not Learning Utility: Objective Mismatch in Human Behaviour Models for Robot Learning

在驾驶与物体交接中，把人类行为生成质量与下游强化学习收益分别评测。发现常见行为指标不能稳定预测策略表现，提醒训练伙伴模型的选择应服务于最终交互能力。

论文：https://arxiv.org/pdf/2610.34839

项目页：https://objective-mismatch-hri.github.io/supplementary/

35世界模型评测加入机器人几何约束

RoboWorM: Evaluating movement realism, consistency and acceleration affects in Embodied Video World Models

利用运动学树和固定相机等硬件先验，分别检查单视角运动稳定与多视角几何一致性。进一步审计加速后的物理质量，提醒读者画面观感不等于可部署的交互保真度。

论文：https://arxiv.org/pdf/2610.33807

代码：https://github.com/ltdoanhwork/RoboWorM

项目页：https://huggingface.co/datasets/doanh25032004/vgp_datasets

36接触前向保持硬约束，反向平滑

Differentiable Contact Dynamics for Legged Locomotion: Policy Learning and Contact-Implicit Trajectory Optimization

只在反向传播替换摩擦接触灵敏度，前向仍使用硬接触求解。平滑梯度帮助跨越接触模式边界，应用于腿式策略与接触隐式轨迹优化，并展示无需微调的硬件迁移。

论文：https://arxiv.org/pdf/2610.35852

项目页：https://diffcontact.github.io/

37密集微型集群如何保留安全间距

Distributed Predictive Flocking for a Swarm of Miniature Vibration-Driven Robots in a Confined Arena

让每个振动驱动微型机器人独立求解预测控制，仅与邻居交换预测轨迹。在数字孪生中比较六类安全约束；接近几何密堆极限时，碰撞仍明显增加，不能泛化为无碰撞保证。

论文：https://arxiv.org/pdf/2610.35582

代码：https://github.com/prajolshrestha/bristlebot-swarm-mpc

项目页：https://doi.org/10.5281/zenodo.22642863

38双臂专业分工与角色互换的取舍

Learning Functional Specialisation for Bimanual Robotic Control

在双循环控制器的肌骨系统中，比较共享与差异目标、固定与变化手部角色。固定分工可提高特定任务效率，却降低角色反转时的灵活性，指标必须连同训练组织一起看。

论文：https://arxiv.org/pdf/2610.33947

代码：https://github.com/jingwen-ava-li/code202609

39多机器人巡查与围堵分开表示

DSGN: A Dual-Stream Gated Network for Scalable Simultaneous Surveillance and Containment

将区域巡查和目标围堵分别编码，再以情境门控协调分散覆盖与聚拢包围的切换。概率先验约束模糊决策，适合研究随群体规模增长而加剧的任务冲突与协同。

论文：https://arxiv.org/pdf/2610.34084

代码：https://github.com/zuozysama/DSGN

40把视觉采集修成可交互仿真

SceneAgent: An Agentic Pipeline from 3D and 4D Captures to Simulation Environments

组织重建、物性估计、部件拆分与关节建模，并在仿真中检查和修复中间错误。支持多类三维与动态输入；真实策略迁移仍属初步展示，应区分重建指标与控制收益。

论文：https://arxiv.org/pdf/2610.34745

项目页：https://computationalrobotics.seas.harvard.edu/SceneAgent/

41让机器人的手势符合交流语境

Context-Aware Generation and Modulation of Expressive Motion Behavior using Multimodal Foundation Models

把动画原则写入动态运动基元，再由多模态基础模型生成和调节全身手势。结合蒸馏降低运行开销，通过用户研究考察语境一致性，并展示不同运动学平台上的表达。

论文：https://arxiv.org/pdf/2610.35615

代码：https://github.com/TillHielscher/gen-mod-expressive-motion-behavior

项目页：https://gen-mod-expressive-motion-behavior.github.io/

42只修正策略最容易失手的局部

When, Where, and What to Correct: Human-Guided Adaptive Residual Correction for Robot Policies

冻结原策略，用少量人工纠正确定残差控制的激活位置与探索子空间，再由任务奖励优化动作。在真机精细操作中验证，适合关注末端精度与低成本后训练的读者。

论文：https://arxiv.org/pdf/2610.37547

项目页：https://anonymous.4open.science/w/arc-6DD5/

43成功之外，还要控制动作质量

Learning Quality-Controllable Robot Policies with Iterative Hindsight Relabeling

把速度、平滑性等质量指标作为策略条件，通过在线采样、成功轨迹回收与事后重标注迭代扩展可控范围。在仿真及传送带、酒杯搬运真机任务中验证。

论文：https://arxiv.org/pdf/2610.37393

项目页：https://sinnnj.github.io/qcp

44把走过的地方存入空间记忆

Remember Where You Walk: Learning Spatial Memory for End-to-End Legged Robot Exploration

以位置索引的局部空间矩阵记忆替代单纯循环状态，在迷宫探索中保留离开视野的布局信息。探索覆盖率与强基线相近，价值主要体现为更好的空间重建与记忆。

论文：https://arxiv.org/pdf/2610.37203

代码：https://github.com/LeonMGNS/RememberWhereYouWalk

项目页：https://leonmgns.github.io/RememberWhereYouWalkWebiste/

45双臂倾斜物体时留足摩擦余量

Keeping Contact: Task-Adaptive Friction Margins for Bimanual Non-Prehensile Manipulation

用有符号摩擦距离衡量接触力距滑移边界的余量，再依据任务力矩需求自适应调整。通过二阶锥规划分配接触力，并用真机物性变化与模型失配实验检验稳定接触。

论文：https://arxiv.org/pdf/2610.36961

项目页：https://keeping-contact.github.io/

46用三维运动监督动作表征

Trace-WAM: Learning to Act via 3D Motion Prediction for World Action Models

在视频预测与动作生成之外加入未来三维点运动预测，以三分支模型突出交互动力学。视频和运动预测用于训练，执行时保留当前观测到动作的路径，避免额外生成步骤。

论文：https://arxiv.org/pdf/2610.36840

项目页：https://trace-wam.github.io/

47把单臂示范转成双臂监督

RoboSTD: Zero-Shot Single-to-Dual Transfer via Sagittal Mirroring for Bimanual Learning

对观测、本体状态与动作做矢状面镜像，再用语言模型生成时空协同约束，构造双臂伪示范。在仿真与真机 ALOHA 上验证，迁移价值依赖镜像可执行性与任务结构。

论文：https://arxiv.org/pdf/2610.36685

代码：https://github.com/HuaHungy/RoboSTD

项目页：https://huahungy.github.io/RoboSTD/

48把导航与操作统一到决策模型

OmniBrain-0: A Unified Agentic Model for Embodied Intelligence

通过具身预训练、领域专家蒸馏和多教师在策略蒸馏整合导航与操作决策。再把执行证据变为评分规则和验证轨迹，考察长任务恢复，收益仍需结合具体工具接口理解。

论文：https://arxiv.org/pdf/2610.36013

项目页：https://public-bots.github.io/OmniBrain-0/

49先学习动作表示，再学习生成

Scaling Dexterous Manipulation via Action Representation Pretraining

利用大规模人类操作预训练动作编码器，冻结后在表示空间训练紧凑生成模型。双手灵巧操作基准显示收益，提供独立于视觉预训练的扩展路线，仍需任务条件动作学习。

论文：https://arxiv.org/pdf/2610.34453

项目页：https://arpg-project.github.io/

50预测下一段动作，获得隐式前瞻

AdaPolicy: Action-Driven Anticipatory Robot Policies via Next-Chunk Prediction

当前动作头同时输出动作与演化状态，训练时再由展开动作头预测下一段，使动作损失塑造未来感知。无需编码未来观测，执行不增加展开分支，覆盖仿真与精细真机任务。

论文：https://arxiv.org/pdf/2610.34280

项目页：https://teee111.github.io/AdaPolicy/

51视频先验适配不同相机数量

X-World: Transferring Video Priors to Variable-View Robot Control

用跨视角残差注意力和固定长度查询表示连接视频骨干与共享动作专家，并支持异构动作接口。结果同时呈现泛化优势与相机、背景、噪声扰动弱点，部署仍需逐类验证。

论文：https://arxiv.org/pdf/2610.34105

项目页：https://logosintelligencelab.github.io/X-World/

52长任务奖励需要记住前置步骤

RoboLongR: Causally Consistent Rewards Teach Robots to Progressively Master Long-Horizon Manipulation

用阶段乱序监督和循环记忆约束进度判断，避免只凭最终画面奖励漏步骤的执行。结合行为克隆、在线强化学习与失败反馈纠偏，目标任务仍需示范和交互适配。

论文：https://arxiv.org/pdf/2610.33863

项目页：https://chenzixuan99.github.io/RoboLongR-public/

53用排序任务检验机器人视觉推理

RoboReason: Boosting Robotics-Relevant VLM Reasoning Through Scalable Data Generation

构建帧时序、物体深度与左右排序任务，并自动生成训练数据，支持监督微调和强化学习。排序规模可控，适合低成本检查空间时间推理，再单独验证下游机器人收益。

论文：https://arxiv.org/pdf/2610.33778

项目页：https://sagnikanupam.com/papers/roboreason/

54把环境变化保存进三维情景记忆

Gaussian Episodic Memory for Long-Term Spatiotemporal Reasoning

以持续更新的高斯地图只记录变化区域，再按需渲染视角，支持跨时间问答与物体定位。基准聚焦半静态环境，适合长期运行的场景记忆研究，不等同连续动态世界建模。

论文：https://arxiv.org/pdf/2610.33158

项目页：https://abwerby.github.io/GEMS/

55为机器人架构复用 ROS 2 建模词汇

A SysML v2 Domain Library for ROS 2 Robot Architectures

在 SysML v2 中定义节点、通信、生命周期与坐标等类型，对照 ROS 2 Jazzy 校验。模型生成监测的案例检出人为偏差；公开领域库与完整生成管线需分开理解。

论文：https://arxiv.org/pdf/2610.33362

代码：https://github.com/SEIL-ISR/ros2-sysmlv2

项目页：https://sysand.com/projects/seil-isr/ros2-sysmlv2/

56反思规划需要按能力给予指导

FISRF: Flexible Instructional Scaffolding Reflection Framework for Robotics Task Planning with Large Language Model

由智能体与导师协作估计当前规划能力，动态选择反思策略与指导强度，减少早期错误经验积累。在 ALFWorld 与真机中考察纠错效率，适合研究交互式任务规划。

论文：https://arxiv.org/pdf/2610.35686

项目页：https://fisrfplanning.github.io/

57把人的目的转为机器人的驱动力

From Language to Drives: A Multimodal Framework for Aligning Autonomous Robots with Human Purposes

通过语义对齐、任务生成和驱动力建模，把自然语言目标嵌入自主学习架构。语言与视觉语言模型构成反馈环，在仿真和实体机器人上检验任务与用户意图的对应。

论文：https://arxiv.org/pdf/2610.35629

代码：https://github.com/DianaCerdasV/From-Language-to-Drives

58比较空中机械臂的抗扰控制

Nonlinear Control of Aerial Manipulator

为带双自由度机械臂的四旋翼建立完整动力学，比较笛卡尔阻抗与自适应滑模控制。实验聚焦相同初始条件下的受扰悬停，适合控制实现参考，不宜外推到复杂接触任务。

论文：https://arxiv.org/pdf/2610.34454

代码：https://github.com/sandeshthapa/Cartesian_Impedance_Control_Aerial_Manipulator

59用脉冲反馈稳定周期运动

Spiking Neuromorphic Control for Periodic Orbits Stabilization in Underactuated Robots: A Greedy Lyapunov-Based Approach

将欠驱动机器人的周期轨道稳定问题化为周期线性时变系统，再把基于李雅普诺夫的贪心设计扩展到脉冲控制。以蝴蝶机器人轨迹稳定为例，关注范围是局部轨道控制。

论文：https://arxiv.org/pdf/2610.37310

代码：https://gitlab.com/-/snippets/4807338

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment