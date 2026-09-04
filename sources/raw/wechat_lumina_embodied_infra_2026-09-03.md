> 原始抓取（微信公众号 HTML 去标签）。入库日 2026-09-04。原文：https://mp.weixin.qq.com/s/qVqpihnA4GezsE2MIJjKDw

# 2026 具身智能 Infra 全景总结：真正拉开差距的，已经不只是模型

author: Lumina具身智能
date: 1788430184

一台机器人在货架前伸出手，夹住一袋薯片，却在抬腕时把旁边两袋一起带倒。演示视频里，这只是一次失败的操作；而在成熟的系统里，它应该立刻变成一组有用的信息：哪台机器人在执行任务、使用的是哪版程序、从什么时候开始偏离、人工何时接管、最后有没有恢复成功。

如果这些信息只是躺在硬盘里，失败就只是发生了一次。如果系统能够自动找出真正有用的片段、判断机器人做到哪一步、归入对应的失败类型，再进入下一轮训练和复测，那么同一次失败就可能推进整个机器人团队的能力提升。

这就是具身智能基础设施（具身智能 Infra），Infra 是 Infrastructure 的简称。在具身智能领域，Infra 不只是服务器、显卡和机械臂的简单集合，而是一套让机器人能够记录经验、学习技能、接受考试、根据结果继续改进的生产系统。模型决定机器人此刻会做什么，Infra 决定着机器人能不能从每一次成功和失败的过程中中继续变强。

图1：具身智能 Infra 的核心不是单点工具，而是数据、模型、评测与后训练构成的闭环

01 一场失败，为什么会牵动整套系统

先别急着把那袋掉落的薯片送进训练集。工程师需要知道，夹爪碰倒邻近商品之前发生了什么：视觉系统是否一直盯着目标，动作指令有没有在网络传输中迟到，机器人以前是否见过这种高度的货架，还是它一开始就选错了抓取顺序。不同答案会把这次失败送往完全不同的地方。

如果把 Infra 压缩成四个动作，就是记录、学习、检查和复盘。

数据系统负责记录机器人经历了什么，模型把这些经历变成动作能力，评测系统检查它是不是真的学会了，后训练再把新发现的问题送回下一轮学习。它们不是四条并排的赛道：前一环的输出的结果是后一环的输入，评测发现的问题又会决定下一次机器人应该如何决策。

这也解释了为什么具身智能 Infra 不能按“数据平台、模型平台、评测平台”完全分开理解。机器人的动作会改变它下一秒所看到的世界，一个微小的误差也可能沿着整条链路扩散。采集设备若出现延迟，错误会进入训练数据；模型若只在熟悉场景里拿高分，部署后仍可能失效；未经筛选的失败回放若直接混入训练集，下一版机器人甚至可能学到错误动作。

因此，理解 Infra 的最好方式，是跟着一次失败的案例走一个闭环：把失败的数据记录下来，发现造成错误的原因，修正模型的工作方式，评测确认问题是否真正解决，再将修正后模型再次部署到机器人进行操作。

2026 年密集出现的工作，恰好把行业研究的重心从“做出一个能演示的机器人”推向了“怎样让整套系统持续运转”，本文就把镜头放在这场正在发生的变化上。

02 数据：先把机器人所经历的过程说清楚

DROID 是这条路线绕不开的起点。它用统一硬件和跨站点流程收集7.6 万条示范、约 350 小时交互，覆盖 564 个场景和 84 项任务。[1] 当年的难题，是让不同地点采到的数据能够放在一起。到了 2026 年，数据采集的入口已经从一套设备变成一项在线服务：Cobalt 允许操作者用手机、VR、3D 鼠标或键盘远程控制仿真与真实机器人。系统每秒接收 20 次控制指令，单张 GPU 最多支持 8 人同时操作，五天内就从九个国家获得超过 7500 条、50 多小时的手机示范。[2] AXIS 更进一步，把浏览器遥操作、任务生成、成功判断、质量筛选和数据增强串成一条可以持续扩张的生产线；首版覆盖 207 个任务、超过 5 万条轨迹，并让 π0.5 在未见任务上的总体成功率提高 5.8%。[4]

图2：AXIS 把任务生成、浏览器遥操作、离线清洗、数据增强和真机验证接成一条数据生产线

这三项工作之间的变化，不只是采集规模变大。现在采集数据，已经不只是录下机器人做了什么，还要把当前的环境、指令和结果一起记下来，方便事后判断它为什么成功或失败。

如果货架前的相机画面、关节状态、控制指令、人工接管和成功判断没有使用同一套时间记录，事后看到的只是一段失败视频，而不是可以用于诊断的样本。MagicSim 用同一套运行规则连接场景、控制器、技能、规划、评测与记录，让每条指令都能留下语言、动作、画面、空间位置和任务状态彼此对齐的轨迹。[5] ACE-Data-0 在真实家庭环境里做了相似的“重新接线”：第一人称和多视角视频、全身与手部运动、物体的位置与姿态、声音和触觉都完成空间与时间对齐，首版包含 150 小时、1700 万帧、200 类任务和 7.5 万个交互片段。[7]

但完整不等于昂贵数据越多越好。HumanNet 用100 万小时第一、第三人称视频扩大模型对物理世界的认识；在相同测试条件下，1000 小时第一人称视频继续训练的效果超过 100 小时真实机器人数据。[3] 这个结果不能被解读为“人类视频可以替代机器人轨迹”，因为视频里没有机器人能够直接执行的动作指令。它更像是在提醒团队：物体关系和做事步骤可以从廉价视频中学习，但接触力度和精确控制仍要靠真机数据校准。

Data Pyramid 把这种分工画得更清楚。真实机器人、便携设备采集的人类示范、人类视频、仿真数据和普通图文数据，各自拥有不同的质量、成本和用途，不能简单换算成一个总小时数。[6] Xiaomi-Robotics-U0 则尝试在这些来源之间自动生成和改写场景；论文报告，在真实操作实验中，π0.5 面对陌生环境时的成功率从 36.9% 提高到 63.2%。[8]合成画面是否逼真只是表层问题，真正的验收标准是画面里的空间关系、机器人限制和物体运动能否在真机上成立。

于是，数据预算不再是一笔总账。人类视频负责扩大世界知识，便携式采集示范提供做事步骤，仿真制造可控的极端条件，真机轨迹校准接触与执行；最稀缺的部分，是让这些多源的数据能够为同一个任务服务。对于失败的任务，系统最终需要产出的也不是一个视频文件，而是一份证词：它来自哪台机器人和哪版程序，在哪个时间点偏离，属于哪种错误，是否值得进入下一轮训练。

数据 Infra 在今年最明显的进展，是采集、判断质量和自动整理开始连成一条流水线。下一步的重点是让系统自动知道缺什么、哪些样本最值得补，以及采集到一条数据后用于改善了哪个问题。

03 模型：把不同经验变成可执行的能力

2024 年，OpenVLA用97万条真实机器人数据训练了 70 亿参数的模型，并开放权重、训练代码和数据工具。[9] 它让研究者第一次可以在相对统一的底座上讨论机器人如何生成动作、适应新任务并部署到真机。此后模型规模迅速扩大，但今年在模型上的工作透露出一个不那么直观的事实：一个模型吃下的数据越多，在不同机器人上的身体差异、视觉信息损失和动作快慢差异就愈加凸显。

LingBot-VLA 用约 2 万小时、九种双臂机器人配置的数据训练，并在四种平台上评测；配套代码在 8 张 GPU 上达到每秒处理 261个样本，相对既有训练代码实现 1.5—2.8 倍的加速。[10] 半年后的 2.0 版本把数据规模提升到约 6 万小时，其中包括覆盖 20 种机器人配置的 5 万小时机器人轨迹和 1 万小时第一人称人类视频，动作范围也扩展到头部、腰部、底盘和灵巧手。[11] 这里的门槛早已不只是模型大小，而是不同数据怎样混在一起、不同机器人怎样描述身体、同一个动作怎样翻译到不同的机器人上，以及训练与验证能否顺畅衔接。

货架失败也可能暴露模型内部的信息断层。DeepVision-VLA 发现，画面进入模型后，重要的视觉细节会在层层处理时逐渐变淡，于是增加一条独立的视觉通道，把与动作有关的细节重新送回深层网络。[12] 当机器人在确定“看见”旁边两袋薯片的情况下却仍然碰倒它们，这里出现的问题不一定是数据有没有遮挡，而可能是关键画面没有真正影响最后的动作。

图3：DeepVision-VLA把重要的画面细节重新送回深层网络，避免机器人“看见了却没有用上”

统一还会遇到思考速度与动作速度的冲突。Qwen-VLA 把机器人轨迹、人类示范、仿真、导航和普通图文数据放进同一训练框架，同时明确告诉模型“这台机器人身体是什么样、动作应该怎样表达”。[14] τ0-VLA 没有要求每个细小动作都“思考更久”，而是让高层先决定下一步子任务、预测结果并判断进度，低层模型继续负责快速控制；其低层策略使用 40115 小时异构真实数据训练。[13] 这种分工不是为了把模型做得更复杂，而是承认“先抓哪一袋”和“手腕下一帧转多少”本来就不该用同一种思考速度。

《The Embodiment Gap in Robot Foundation Models》把这个问题概括为“本体鸿沟”：同一个模型或同一份数据搬到另一台机器人上，距离真正跑起来还剩多少工程工作。[15] 如果跨机器人本体的模型没有同时说明相机怎样校准、动作怎样翻译、控制器改了多少、后来又补了多少训练，所谓迁移能力就很难比较。模型 Infra下一步更重要的资产，是身体描述、动作含义、执行记录和版本管理这些稳定接口。否则模型每升级一次，货架现场都要重新做一遍系统集成。
04 评测：不是考高分，而是确认机器人真的学会了

设想两个实验室测试同一个机器人程序。一个机器人把薯片摆回到预定的大致位置，另一个用定位工具恢复到毫米级；一个机器人在固定灯光下测试十次，另一个还会改变货架高度、背景和相机噪声。即便最后都写“成功率”，两个“数字”所描述的也不是同一件事。

评选Infra首先要消除的，就是这种隐藏在分数背后的实验差异。

RoboTwin 2.0 是这个领域的早期成果。它把数据生成、物体库、任务生成和统一规则放在一起，RoboTwin-OD 包含 147 个类别的 731 个对象实例，并系统地改变杂乱程度、光照、背景、桌面高度和语言指令，覆盖 50 项双臂任务与五种机器人。[16] 2026 年的基准审计检查了 LIBERO、CALVIN、SimplerEnv、RoboCasa和 RoboTwin 2.0 等是否存在视觉捷径、反复刷题和数据来源重合，指出高分有时只是模型记住了训练环境，或者来自极少数量的测试波动。[17] LIBERO-X 则把空间、物体和指令变化逐层叠加，让读者看到模型怎样一步步掉分，而不是把脆弱点藏进一个平均数。

如果失败来自环境变化，评测还需要记录“世界究竟变了多少”。RADAR改变物体摆放、初始姿态、光照和传感器噪声，并用三维位置进行自动评测；加入传感器噪声后，论文中的期望 3D IoU 从 0.261 降到 0.068。[21] PhAIL 从另一端修正统计方法：它不再只看规定时间内成功或失败，还比较完成任务所需要的时间，并用人类在使用同一套设备的速度作为参照。[22] 这两类工作合在一起，才回答了“环境如何变化”以及“两个策略的差异到底稳不稳”。

然而，仿真里的可控并不能替代真机的麻烦。ManipulationNet 用标准化硬件和统一软件组织跨地点的真实测试，把任务分成物理技能与看懂指令两条赛道。[19] ManipArena 提供 20 项桌面与移动任务、10812 条专家轨迹、约 188 小时数据，并同时检查熟悉环境、画面变化和陌生语义下的表现。[20] UMI-Bench 1.0 则把范围收窄到“手持设备采集示范之后，机器人能否真正学会”的完整链条，连同场景复位、程序执行、日志和任务分析一起复现。[23] 它们的共同判断是：真实评测不是下载一个数据集，而是长期维护硬件状态、复位流程、软件版本和检查记录。

RoboDojo 在7 月把这些分散要求收束成一套持续运行的系统。它提供 42 项仿真任务和 18 项真实任务，在仿真中检查模型能否适应变化、是否只是背题、动作是否精确、长任务能否坚持完成，以及陌生指令能否听懂；RealEval 则通过远程访问、标准硬件、场景复位和统一接入方式组织真机测试，XPolicyLab已接入 30 种机器人程序。[24] RoboDojo 真正补上的并不是“再多几个任务”，而是模型接入、批量运行、结果汇总和真机复现之间的那段工程。对于无法拥有整套硬件的团队，这段工程决定一个结论能否被别人重复。

图4：RoboDojo用统一接口连接 42 项仿真任务、18 项真机任务、XPolicyLab 和持续更新的排行榜

评测边界还在继续外扩。SIMPLE 把 MuJoCo 更准确的接触模拟与 IsaacSim 更丰富的画面结合，提供 60 项全身任务、50 个室内场景和超过 1000 个物体，并加入自动生成动作和低延迟 VR 遥操作。[25] RoboSynChallenge 允许大量合成数据参与训练，却把最终判断留给从未见过的真实操作环境。[26] 一个扩大可以重复测试的机器人和任务范围，一个迫使合成数据接受真机验收，二者都在阻止模型只对排行榜优化。

成熟的评测系统应该像软件发布前的自动体检：数据、模型、相机或控制器版本一变化，就自动跑一遍仿真测试、真机抽检和失败分析。公开任务适合横向比较，隐藏任务和真实故障负责发现模型是不是只会刷题。

评测 Infra 在今年的进展，是从“公布一张排行榜”走向“同时控制环境变化、统计方法和真机复现”。下一步需要的是长期运行的公共测试网络，让一项改进换个实验室、换台机器人后仍然成立。

那袋薯片再次被抓起时，团队需要看到的不是一个更漂亮的平均分，而是原来的失败原因是否消失，又有没有引入新的问题。

05 后训练：把失败变成下一版的进化能力

假设评测确认问题来自抓取末段，最直接的做法似乎是把失败与人工恢复一起加入训练。这里又有一个陷阱：失败片段可能包含长时间空转、无效接管或已经偏离任务的动作，未经筛选的“更多经验”并不等于更好的监督。后训练的价值，正是把现场回收压缩成模型能够吸收、又不会破坏旧能力的学习信号。

LifeLong-RFT 把一段长动作切成多个小块，并根据每一步是否正确给予反馈；在LIBERO 持续学习实验中，论文报告其相对普通继续训练平均成功率提高 22%，适应新任务只使用 20% 的训练数据。[27] AtomVLA 则把长任务拆成更小的子任务，让模型先预测每段动作可能带来的结果，再从离线数据中挑选更好的动作。[28] 一个关心机器人学会新技能后是否忘掉旧技能，另一个关心最终成败之前能否得到更细的过程反馈，恰好补上真实回放最常缺的两种信息。

当机器人数量增加，算法还必须和人的工作流对齐。HELP让两名分工不同的操作员同时监督 12 台机器人：远程操作员集中处理关键接管与恢复，现场人员负责监控、启动任务和复位环境；系统再把回放分成有效推进、空转、导致失败和恢复四类，只让有用片段进入下一轮训练。[29] 这里最重要的不是“完全不需要人”，而是把人的注意力投向机器人最不确定的位置。

图5：HELP将操作员控制台、分布式机器人、集中推理和后训练服务器组织成一套机器人队伍学习系统

DEED 把同一问题带回了本文开头的超市场景。它在 Unitree G1-Edu 和 GR00T N1.6 上处理控制快慢不一致、数据筛选、视觉重点标记和环境变化带来的问题。论文的判断相当克制：从实验室到商店，主要障碍首先是系统集成，不一定是再发明一个网络结构；当机器人自己生成的回放越来越多时，反复训练还可能开始损害性能。[30] 闭环速度因此不能单独追求，回到训练集的经验必须有来源、有理由，也必须经过复测。

图6：DEED在真实货架补货任务中记录抓取、放置和复位的完整执行过程

现在，那次货架失败终于走完一圈：采集系统保存了前后经过，模型团队定位了画面或动作问题，评测系统重新制造同样的困难，后训练只吸收真正有帮助的片段。后训练 Infra 在下一步的关键，是让更多机器人共同提供经验，同时防止模型反复学习自己生成的错误。

Infra 的壁垒不在于某个步骤从不出错，而在于错误能否沿着这条路径变成一个可以提升能力的经验。

06 下一步：公共规则与企业闭环缺一不可

具身智能 Infra 中，有一部分天然需要开源。不同团队只有采用相近的动作格式、数据质量标准、隐藏测试任务和失败分类方法，实验结果才具有比较价值。ManipulationNet、RoboDojo 这类跨机构真机网络如何组织设备、任务和运行流程，同样需要一套共同规则。单个实验室很难长期维护足够多的机器人和场景，公共研究则可以把实验方法、技术标准和验证结果沉淀下来，让后来者能够复现、比较并继续改进。

另一些能力只能在长期运营中形成。相机脏污、网络抖动、硬件磨损、任务排队、权限控制、版本回滚和安全停机，都不会随着论文截稿而停止。机器人队伍每天产生的失败记录、恢复过程、人工接管和环境变化数据，也无法由一次性测试替代。产业界真正难以复制的资产，是企业判断哪些现场经验值得送回系统的能力，以及经验回流后，如何避免新版本在其他场景中出现意外退化。

这并不意味着企业必须公开所有数据。更可行的方式是进行分层交换：企业保留原始客户数据，同时公开数据结构、匿名化的失败类型、评测套件和接口规范；研究机构不必承担全天候运营，也可以贡献对照实验、统计方法和跨平台复现结果。公共层负责让研究结论可信，企业内部的运营层负责维持闭环持续运转，两者通过可审计的接口连接起来。

未来，具身智能公司的核心差距很难只体现在评测榜单领先几个百分点。更关键的指标是“闭环周转时间”：从真实故障发生，到定位原因、补充数据、更新模型、完成分层测试，再安全地重新部署到机器人，整个过程究竟需要几天、几小时，还是几分钟。

笔者认为，模型依然重要，但它更像一台发动机。具身智能 Infra 决定企业能否持续提供燃料、监控运行状态、完成上线测试，并留下可追溯的维修记录。机器人进入家庭、商店和工厂后，错误无法完全避免。企业真正需要的是一套能够解释错误如何发生、及时修复问题，并减少同类错误再次出现的底层系统。

参考资料

[1] Khazatsky et al., DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset, 2024/2025.

https://arxiv.org/abs/2403.12945

[2] Agarwal et al., COBALT: Crowdsourcing Robot Learning via Cloud-Based Teleoperation with Smartphones, 2026.

https://arxiv.org/abs/2605.19138

[3] Deng and Zhou, HumanNet: Scaling Human-centric Video Learning to One Million Hours, 2026.

https://arxiv.org/abs/2605.06747

[4] Zhao et al., AXIS: A Growable Community-Driven Data Engine for Scalable Robot Manipulation, 2026.

https://arxiv.org/abs/2607.21588

[5] Lu et al., MagicSim: A Unified Infrastructure for Executable Embodied Interaction, 2026.

https://arxiv.org/abs/2606.17511

[6] Ye et al., Data Pyramid for Embodied Manipulation: A Survey, 2026.

https://arxiv.org/abs/2607.24744

[7] Cao et al., ACE-Data-0: Human-Centric Ambient Capture as Embodied Data Engine, 2026.

https://arxiv.org/abs/2607.28625

[8] Li et al., Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Model, 2026.

https://arxiv.org/abs/2607.11643

[9] Kim et al., OpenVLA: An Open-Source Vision-Language-Action Model, 2024.

https://arxiv.org/abs/2406.09246

[10] Wu et al., A Pragmatic VLA Foundation Model, 2026.

https://arxiv.org/abs/2601.18692

[11] Wu et al., From Foundation to Application: Improving VLA Models in Practice, 2026.

https://arxiv.org/abs/2607.06403

[12] Luo et al., Look Before Acting: Enhancing Vision Foundation Representations for Vision-Language-Action Models, 2026.

https://arxiv.org/abs/2603.15618

[13] Cai et al., τ0-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation, 2026.

https://arxiv.org/abs/2608.16885

[14] Wang et al., Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments, 2026.

https://arxiv.org/abs/2605.30280

[15] Domae et al., The Embodiment Gap in Robot Foundation Models, 2026.

https://arxiv.org/abs/2608.18433

[16] Chen et al., RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation, 2025.

https://arxiv.org/abs/2506.18088

[17] Jiang et al., What Are We Actually Benchmarking in Robot Manipulation?, 2026.

https://arxiv.org/abs/2606.04233

[18] Wang et al., LIBERO-X: Robustness Litmus for Vision-Language-Action Models, 2026.

https://arxiv.org/abs/2602.06556

[19] Chen et al., ManipulationNet: An Infrastructure for Benchmarking Real-World Robot Manipulation with Physical Skill Challenges and Embodied Multimodal Reasoning, 2026.

https://arxiv.org/abs/2603.04363

[20] Sun et al., ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation, 2026.

https://arxiv.org/abs/2603.28545

[21] Chen et al., RADAR: Benchmarking Vision-Language-Action Generalization via Real-World Dynamics, Spatial-Physical Intelligence, and Autonomous Evaluation, 2026.

https://arxiv.org/abs/2602.10980

[22] Arkhangelskiy, PhAIL: A Real-Robot VLA Benchmark and Distributional Methodology, 2026.

https://arxiv.org/abs/2605.29710

[23] Jin et al., UMI-Bench 1.0: An Open and Reproducible Real-World Benchmark for Tabletop Robotic Manipulation with UMI Data, 2026.

https://arxiv.org/abs/2606.10382

[24] Chen et al., RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies, 2026.

https://arxiv.org/abs/2607.04434

[25] Wei et al., SIMPLE: Simulation-Based Policy Learning and Evaluation for Humanoid Loco-manipulation, 2026.

https://arxiv.org/abs/2606.08278

[26] Zhao et al., RoboSynChallenge: Mastering Real-World Dexterity via Generalizing Synthesized Manipulation Skills, 2026.

https://arxiv.org/abs/2608.12416

[27] Liu et al., Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning, 2026.

https://arxiv.org/abs/2602.10503

[28] Sun et al., AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models, 2026.

https://arxiv.org/abs/2603.08519

[29] Zhai et al., HELP: Human-Efficient Large-Scale Robot Post-Training with Rollout Segmentation, 2026.

https://arxiv.org/abs/2607.09776

[30] Sala Sisó et al., Closing the Lab-to-Store Gap: A Data-Efficient Post-Training and Experience-Driven Learning VLA Framework for Retail Humanoids, 2026.

https://arxiv.org/abs/2607.20345
