---
title: Science Robotics最新综述｜一文读懂仿生多模态机器人如何“七十二变”
author: 机器人大讲堂
date: "2026-07-25 08:00:00"
source: "https://mp.weixin.qq.com/s/U-6QiMO1Au_77R6fKDQFBg"
---

# Science Robotics最新综述｜一文读懂仿生多模态机器人如何“七十二变”

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/LJiau2qPWAcUB7NhD61gK7qibQdRHQvQ1KsvwsUpbAUepZeNzW349LvNXKzu9jH4ibhtKqJm9Tlz1AOr2tgibJEuQA/640?wx_fmt=png&from=appmsg#imgIndex=0)

为应对复杂多变的生存环境，自然界中许多生物都演化出了多模态运动能力。例如，鸟类既能展翅飞翔，也能在陆地行走跳跃，还能停栖于枝头；章鱼既可以通过喷射推进在水中游动，也可以凭借柔软的触手在海底、滩涂爬行。多种运动模态的灵活切换与协同融合，赋予了生物卓越的环境适应能力。

![Image](https://mmbiz.qpic.cn/mmbiz_png/yNRdEJS7ubQwiampnsicZnqmYaJyQvly2etwYmX7vYc7AnNFWialC9vmibMVF9r9U9wEPicHGoibWKyicApD3MbtmLp6BmfQQKlRQJSJY4J4eF9wM0/640?wx_fmt=png&from=appmsg#imgIndex=1)![]()

图1 自然界中生物（从无脊椎动物到哺乳动物）均广泛存在多模态运动能力

相比之下，单一运动模态的机器人，其环境适应能力难以媲美具有多模态运动能力的生物。但从真实应用需求来看，从深空星际探测、深海资源勘探，到工业管道巡检、人体自然腔道诊疗，各类复杂动态场景都迫切需要机器人具备多模态运动的能力，从而胜任多样化的作业任务。

为突破单一运动模态的局限，研究人员开始尝试从自然生物的运动机制中汲取灵感，赋予机器人多模态运动与模态切换的能力。然而，多模态机器人设计并非多种运动机构的简单叠加：要研制兼具轻量化、低能耗和高灵巧度的多模态机器人，需要系统解决本体结构设计、模态切换策略、全局规划与实时运动控制等一系列关键问题。若多模态集成方案设计不当，不仅难以发挥不同运动模态的协同优势，反而可能导致整机结构臃肿、能耗显著增加以及运动灵活性下降。

基于上述背景，北京航空航天大学文力教授团队，联合洛桑联邦理工学院（EPFL）的Auke Ijspeert教授、Jamie Paik教授、大连理工大学刘行健教授，以及清华大学张一慧教授在国际著名学术期刊Science Robotics上发表了题为《Bioinspired Multimodal Robotics》的综述。北京航空航天大学的任子宇教授，博士生朵有宁、徐浩原为论文的共同第一作者。本论文获得国家重点研发计划，国家自然科学基金等项目的支持。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/yNRdEJS7ubSq2fnRcGGj1rCOzg74CPCBhnhRPAAxMoeRia74yR1s5BhRBoWsTsNuNu3SXCJz2J3Q6FACkg3mRFy3yqcCoibgZQe9WxtQlcXKk/640?wx_fmt=png&from=appmsg#imgIndex=3)

文章系统梳理了仿生多模态机器人的进化历程，总结了仿生多模态机器人的设计范式、评价指标、模态切换策略，并提出了融合计算智能与物理智能的自主规划及控制框架。

**01.**

**仿生多模态机器人的发展历程**

论文首先将仿生多模态机器人（Bioinspired Multimodal Robot）定义为：在同一平台上集成至少两种运动模态，或同时具备移动与操作能力，其中至少一种模态受生物启发，并能够在不同模态之间切换的机器人。

回顾历史，仿生多模态机器人的发展经历了三个重要阶段：

1. 工程化集成时期（1959年-20世纪末）：代表性的机器人有：MOBOT Mark I（1959）、WABOT-1（1973）和Odex I（1983），它们依靠独立驱动模块的堆叠实现多种运动，运动多为单一功能硬件的集成。
2. 仿生多模态设计时期（21世纪初期）：代表性的机器人有：PolyBot（2000，提出形态自适应重构）和AZIMUT（2003，首次正式提出“Multimodal Robot（多模态机器人）”概念，并实现了腿-轮-履带驱动一体化）。仿生多模态运动开始嵌入机器人形态本身，而非靠子系统的简单叠加。
3. 计算智能+物理智能融合时期（21世纪初-至今）：现阶段，计算智能与物理智能两条技术路线同步演进：在计算智能层面，先后涌现出连续自建模机器人（2006）、采用中央模式发生器的仿蝾螈机器人（2007）以及采用分层强化学习的轮足机器人（2024）等代表性成果，提升了机器人在复杂环境下的自主控制能力；在物理智能层面，以微型磁控软体机器人（2018）为典型代表，其利用柔性材料和结构，把控制逻辑嵌入机械本体，大幅降低算法控制压力。

纵观多模态机器人的发展历程，其正由不同功能和运动方式的工程化集成，向仿生运动与工程运动深度融合、不同模态协同增效、计算智能与物理智能深度耦合的方向发展。

从功能目标看，其任务已由复杂地形通行和跨介质运动，进一步拓展至通过模态切换与协同提升续航能力和运动性能，并增强机器人与环境的交互和操作能力，最终实现复杂开放环境中的高适应性、高效运动与移动操作一体化。

**02.**

**设计仿生多模态机器人的几大“核心考量”**

设计一台仿生多模态机器人需要考量多个方面并进行权衡，论文指出设计时必须权衡的六大核心考量：

- 空间分配：需要在有限的机身体积内布置多种运动部件；
- 质量特性：某一运动模态的专用部件在其他运动模态下可能会成为无效负载；
- 模态协同：各运动模态难以产生1+1>2的协同增益效应，常出现性能抵消；
- 刚度矛盾：机器人变换构型时往往需要降低机身刚度，而执行高负载、高精度的运动时又需要足够刚度来保证力传递与稳定性，同一结构难以同时兼顾柔性与刚性；
- 驱动冲突：不同驱动方式在能耗、输出力等方面的需求不兼容；
- 变形能力：机器人需要具备改变自身构型的能力以适应不同环境。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/yNRdEJS7ubT0xdGQuAj6ibtTPfZiaRt7WtVzurTcicUoJDqprQleFKfAZfyKbowP4eGaahwzDms6KddDv18tmKicgHLudlRHia2wlUXA1yBxiayB8/640?wx_fmt=png&from=appmsg#imgIndex=4)![]()

图2 仿生多模态机器人的发展历程与运动模态集成

**03.**

**不“凭感觉”！提出5项定量评价指标**

过去虽然有不少多模态运动的机器人文献报道，但缺乏定量的“核心指标”以及统一的评价标准，进而难以定量测算每新增一个模态带来的收益与代价，本综述提出了专门针对仿生多模态机器人的量化评价体系—5项指标可以分为设计效率和运动性能两大维度。论文的表1中列举多款机器人样机案例辅助对5项指标的理解。

- 设计效率方面：

1.模态数量Number of Modes (Nmode)

首先，多模态机器人需要明确具有的模态数量，这是衡量机器人功能多样性最基础的指标，也是所有性能测算的基准参数。

定义：![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/yNRdEJS7ubRPmw93zVwWdZSPicIUokWz53blF2CuHOYwJZHWw4Ukypt1JhTFpCQF9uunSX2FMkCXibGnToLXiaBUeT1Yu0CaHTwjxpVrEx5icJc/640?wx_fmt=gif&from=appmsg#imgIndex=6)![]()（Ndom表示机器人可以工作的不同域的数量，Ci表示在第i个域中可以实现的构型的数量，Mi,j表示机器人在第i个域和第j个构型下能够实现的运动模态的数量）。Nmode数值越高通常意味着适应性和通用性越强。

典型样机案例：Tribot机器人[1]能够实现垂直跳跃、水平跳跃、翻滚跳跃、后翻式行走、仿尺蠖爬行共五种运动模态，Nmode=5。

2.模态边际成本Marginal Cost of Modality (MCM)

在多模态机器人的设计中，新增一种模态往往需要付出一定的成本和代价（如重量、体积等）。需要评估多模态集成的硬件效率，判断新增的模态是否“划算”。

定义：![Image](https://mmbiz.qpic.cn/mmbiz_gif/yNRdEJS7ubSESfaH0PfvAXcVS9UM2FicWabyia6HiaJpcGC5sVfBWd7YeA5vz7x2yUSvJibWEOqvb3JIL55yPMwV7vejnCNDqLCTw4cJibzibT52s/640?wx_fmt=gif&from=appmsg#imgIndex=8)（Mn表示机器人配备有n种运动模态的总成本。Mn-1表示机器人在加入新模态前，配备有n-1个模态下的总成本）。MCM越低代表机器人多个模态的集成效率越高。

典型样机案例：轮足式机器人[2]在ANYmal四足机器人的基础上增加了轮式运动，其中轮式驱动带来的质量增量约为ANYmal四足机器人重量的0.38，因此轮足机器人MCM=0.38。

3.不同模态下部件的复用率Component Repurpose Percentage (CRP)

多模态机器人在设计时应尽量避免采用“一种模态一套专属硬件”，这会导致模态越多冗余的部件越多，应尽量减少不必要的部件，实现部件在不同模态之间的共用。

定义：不同运动模态之间共享部件的百分比，![Image](https://mmbiz.qpic.cn/mmbiz_gif/yNRdEJS7ubQQWfE4yMMqsB5VqfmnjUSszbvXm6LKTXfKIUdctJ9m63XZssPBiaBibzI9ic149hsBx6ygjSEAxYQXekqlovNLibrO8vQ8VmibPqFM/640?wx_fmt=gif&from=appmsg#imgIndex=9) 。CRP数值越高证明设计越有效，最大限度减少了冗余部件。

典型样机案例：微型深海变构形机器人[3]有两组手性驱动单元同时用于游动和爬行模态，占所有驱动单元总数量的比例为0.4，因此其CRP=0.4。微型片状磁控机器人[4]的全部部件适配于其每一种运动模态，因此CRP=1。

- 性能方面：

1.模态切换成本Transition Cost (Tij)

多模态机器人执行任务需要频繁切换运动模态，切换消耗的时间、能量会直接关系到任务续航与作业效率，该指标量化两种模态互相切换的损耗。

定义：模态i切换至模态j所需的成本（如时间、能量等），可以构成模态切换成本矩阵[Tij]。

典型样机案例：跨水空吸附机器人[5]从空中入水的切换时间约0.13秒，从水中飞入空气中的切换时间约0.35秒，因此其模态切换成本为Tair-water = 0.13 s，Twater-air = 0.35 s。

2.模态互促性能提升系数Performance Improvement (PI)

多模态集成的目标不仅是“拥有多种功能”，更是让不同模态之间产生正向协同。PI量化了多模态协同完成特定任务时，相比单模态下最佳性能的提升比例。

定义：![Image](https://mmbiz.qpic.cn/mmbiz_gif/yNRdEJS7ubQrRPPMgDvYsu6FluYW1nm0AIkCiaibBH2yVgK2aXzB4WkXhJ47jcCJXpnicqAbp7dnZ37PZ4Hys5sibJZicWef3rJibSkHknOofhlJE/640?wx_fmt=gif&from=appmsg#imgIndex=10) （S表示运动模态的集合）。PI>1说明实现了“1+1>2”的协同相互促进的效果。

典型样机案例：Hopcopter机器人[6]通过跳跃与飞行的协同，将续航时间从纯四旋翼的379秒提升至1246秒，PI=3.29，说明仅增加不到三分之一的重量代价，便换来了三倍以上的续航增益。

文中也指出，随着多模态机器人领域的发展，未来的评价指标也将不断地完善。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/yNRdEJS7ubTP06x2WzvGAlcZq9lk0u5se53xc56oUeqedm1GTPrekwvXOtibaOiczqaVt6rJmUwDHEXJXCaR2ccrNpg9WTcI7JttWSTrqibOrk/640?wx_fmt=png&from=appmsg#imgIndex=11)![]()

图3 仿生多模态机器人定量评价指标

**04.**

**仿生多模态机器人面临的两大核心挑战**

设计和制造仿生多模态机器人面临着设计、规划及控制的多重挑战。

在硬件设计方面，传统的“专件专用”模式（比如轮子在地面运动效果很好，但到了飞行模态下就变成了无效载荷），这导致多模态系统极其臃肿复杂。要解决这一问题，就要实现单一部件的多功能化，让单一部件同时承载感知、驱动与计算功能，打破硬件模块间的壁垒。

在规划及控制方面，传统机器人通常在相对稳定的动力学条件下运动，路线的选择也有限；而多模态机器人在模态切换时（例如从行走模态切换为飞行模态），系统动力学会发生剧变，这就导致路径选择的难度增加，同时要求控制器必须在短时间内适应动态的变化。

如解决上述挑战，仿生多模态机器人将可能兼具精简的机械结构与智能的决策能力，理想的仿生多模态机器人应具有以下特点：机械复杂度低、部件复用率高，并能通过智能控制根据任务需求灵活规划路线、平滑切换模态。

![Image](https://mmbiz.qpic.cn/mmbiz_png/yNRdEJS7ubRfJj8vIibDRj4siaR8Rjb6G6OdickQcmloicOUw8PiaHcIahqwVsgEdmw2jic82uk2BTUzaJpkPLF14CqicPuBriaNazX9gcFCGgJeQcQ/640?wx_fmt=png&from=appmsg#imgIndex=13)![]()

图4 仿生多模态机器人设计与控制的挑战与进展

**05.**

**多模态机器人几大设计范式**

具体如何设计出一款仿生多模态机器人？文中总结了三种典型的设计范式：

1. 柔性材料与结构：利用软材料的高自由度和柔顺性实现复杂变形，结合变刚度技术（例如轮式微型机器人[7]利用形状记忆聚合物，变换构型时降低机身刚度，在运动或搭载重物时提升机身刚度），兼具柔性适应性和负载能力，进而在不显著增加驱动部件的前提下，利用软体材料的高自由度特性以及与环境的被动适应性实现多模态运动。
2. 结构复用：其核心逻辑是最大化提升CRP，通过同一机械结构或驱动单元的跨模态共享，在不显著增加硬件冗余的前提下，依托精巧的机构设计方法（如M4机器人[8]通过调整四肢的构型以分别起到腿、轮子或空中螺旋桨的作用），用较少的机械结构和驱动单元实现多模态运动集成。
3. 集群协同的多模态涌现：通过多机器人的协同实现复杂功能，可分为异构集群和同构集群。异构集群（如无人机+无人车协同）或同构集群（如受蜗牛启发的机器人集群[9]）通过协作涌现出单机无法实现的多模态能力。

![Image](https://mmbiz.qpic.cn/mmbiz_png/yNRdEJS7ubRkKHl5U7e2RuGno2fHQSVicpCMXhjhjd5giaoSwFHnzsXHZM194KRWggAZlDfk4NdMJKQ14dsgFwXh0kZW7mIjrb1uxgSJdbUgM/640?wx_fmt=png&from=appmsg#imgIndex=15)![]()

图5 仿生多模态机器人的设计方法论

**06.**

**多种模态的切换策略该如何定义？**

对仿生多模态机器人来说，如何高效、低成本地在不同模态之间切换至关重要。

根据是否发生了结构变换可以将模态切换策略分为无结构变换和结构变换两大类：无结构变换(No structure transformation)，仅通过改变控制策略（如改变步态）实现。结构变换 (Structure transformation)，需要改变机器人身体结构，可以用nM-mM（M表示运动模态，nM-mM表示从具有n个运动模态的构型切换为具有m个运动模态的构型）来细分：

- 0M-1M：无运动能力模态→单一运动模态；
- 1M-1M：单一模态切换另一种单一模态；
- 1M-MM：单一模态切换为多模态；
- MM-MM：多模态状态切换到另一种多模态状态。

同时，模态切换也可以分为主动（Active）和被动（Passive）两大类：主动切换是指机器人通过主动控制驱动部件实现模态切换；被动切换是指依靠水流、碰撞等环境刺激触发模态切换。

文中对现有仿生多模态机器人的切换策略进行了统计，现有论文集中于主动切换以及1M-1M、1M-MM 的模态切换，而相较之下，0M-1M和MM-MM模态切换前期研究鲜见报道，这也是本综述提出的机器人模态切换的一个发展新方向。

![Image](https://mmbiz.qpic.cn/mmbiz_png/yNRdEJS7ubT7ic2X4YI7ibDcIDfmAhJJoNQRPLJ47jyvZlwxDk2Ecgb5GdTn1cWApZQO4gOafTHheouAO211QQBNAvP7T8wNtfXIdtbeTnzicc/640?wx_fmt=png&from=appmsg#imgIndex=17)![]()

图6 仿生多模态机器人的运动模态转换策略

**07.**

**多模态机器人的自主规划及控制，该采用什么架构？**

仿生多模态机器人的自主运动有赖于路径规划与运动控制算法的发展；而要进一步实现长距离、长时间的持续自主运行，还需要系统架构层面的革新。

在路径规划层面，传统方法通常将环境离散为栅格图或概率路网，并综合考虑运动时间、能耗、风险以及模态切换代价以搜索最优路径。但对于多模态机器人而言，运动模态的增加导致状态空间和计算成本迅速增长，实时重规划也更加困难。近年来，分层架构的方法逐渐兴起，全局规划器只提供关键路点，神经网络导航策略根据局部感知实时生成速度指令，再由底层策略自主选择具体运动模态。

在运动控制层面，分别为不同模态设计独立控制器最为直接，但容易在模态切换时产生冲击或动作不连续。以中央模式发生器为代表的仿动物神经系统控制器可以生成节律运动，适用于行走、游动等周期性行为，但难以处理非周期、高动态动作。强化学习能在缺乏精确模型的情况下协调全身运动并学习模态间的连续过渡，但仍面临训练成本高、仿真到现实迁移困难等问题。

在此基础上，本综述提出了面向复杂动态环境的一体化自主规划及控制框架，该结构包含了全局规划模块(Global Planning Module)、执行模块(Execution Module)和多模态感知模块(Multimodal Perception Module)，将计算智能与物理智能耦合在一起，打通从环境感知、任务规划到动作执行的全链路架构：

1.全局规划模块

机器人的“大脑”，它以任务目标（如时间最快、能耗最低等）为导向，结合实时获取的环境信息（地形类型、障碍物分布等），构建全域环境模型。在此基础上，通过多目标优化算法，自主规划出兼顾路径可行性与任务目标的最优运动序列，并动态选择适配的运动模态（如爬行、飞行、游动等），实现“路径规划+模态选择”的一体化决策。面对极端复杂场景，该模块甚至还可以具备主动规划改造环境的能力——例如通过机械臂搭建临时通道、清理障碍等方式，创造原本不存在的可行通路，进一步拓展机器人的作业边界。

目前全局规划仍以传统图搜索、分层优化类算法为主，未来以VLA模型、世界模型为代表的全局端到端规划范式也将具有广泛的应用前景。这类大模型可直接输入环境全景信息与高层任务指令，输出长程路径规划、全程模态切换时序，省去人工建图、分步寻路等中间环节，对动态变化的非结构化环境具备更强泛化能力，是全局决策层一个具备潜力的研究方向。

2.执行模块

机器人的“小脑”，它将高层决策转化为具体的动作执行，主要分为分层控制与端到端策略两条技术路线，以适配不同场景的控制需求：

分层控制（Hierarchical Control）：遵循传统机器人控制逻辑，先将全局路径拆解为局部轨迹片段，再通过多模态运动控制器，将轨迹指令映射为各关节的具体运动参数。该方式稳定性高、可解释性强，适合对可靠性要求严苛的任务场景。

端到端策略（End-to-End Policy）：依托VLA、世界模型等技术，直接将环境感知数据（图像、点云、触觉信号等）映射为整机运动指令，跳过显式的轨迹规划环节。这种方式对非结构化、动态变化的环境具备更强的泛化能力，可快速适配未知场景与突发扰动，是未来多模态机器人控制的重要发展方向之一。

两类技术路线生成的动作指令传输至驱动部件，作用于自适应结构，依靠机器人本体形变适配环境，实现机器人与环境高效、鲁棒的交互，完成计算决策与物理智能的闭环。

3.多模态感知模块

多模态机器人的“感官系统”，该模块包括本体感知和外部环境感知，通过采集多源传感器数据，为执行模块提供实时感知信息，形成闭环反馈回路。其中，本体感知负责持续监测机器人自身运行状态，实时获取位姿、速度、关节状态、执行器状态等数据；外部环境感知负责识别并解析地形结构、障碍物分布、外部干扰等复杂环境特征。

未来，随着多模态大模型、智能传感器与边缘计算技术的发展，机器人感知系统将进一步向高精度、强泛化、智能化的方向演进，为复杂开放环境下多模态机器人的自主决策与智能交互提供支撑。

![Image](https://mmbiz.qpic.cn/mmbiz_png/yNRdEJS7ubTaOMbLaZc3OfcJhq4iaZRjibAKFsp7yaAEunEtmE6LdOsHiccYqWfNibR1ia8re4gibpr2JPDDuCl8FDJGkW8e1dS6UqtCicYib3oINOg/640?wx_fmt=png&from=appmsg#imgIndex=19)![]()

图7 仿生多模态机器人在动态、非结构化环境中导航和规划的系统架构

**08.**

**未来研究展望**

将基于数据驱动的感知、规划与控制架构同具备物理智能的本体实现深度融合，是打造自主化、多功能化、智能化仿生多模态机器人系统极具前景的发展路径。

但要实现这一愿景，仍需攻克一系列关键工程与科学难题：在硬件层面，自适应材料、高功率密度驱动器以及多模态传感器均有待进一步突破；算法层面，不同运动模态间动力学差异大，路径规划与模态选择耦合复杂，以及复杂动态环境下自主性和适应性不足，是亟需攻克的核心挑战。

若能攻克上述跨学科难题，将促进仿生多模态机器人的发展，使其运动灵活性、能效与环境适应性比肩甚至超越自然界生物原型。

论文链接：

https://www.science.org/eprint/DSWZCRX276ZXEYUWFPQN/full?activationRedirect=/doi/full/10.1126/scirobotics.aea7639

推文参考文献

[1] Zhakypov Z, Mori K, Hosoda K, et al. Designing minimal and scalable insect-inspired multi-locomotion millirobots[J]. Nature, 2019, 571(7765): 381-386.

[2] Lee J, Bjelonic M, Reske A, et al. Learning robust autonomous navigation and locomotion for wheeled-legged robots[J]. Science Robotics, 2024, 9(89): eadi9641.

[3] Pan F, Liu J, Zuo Z, et al. Miniature deep-sea morphable robot with multimodal locomotion[J]. Science Robotics, 2025, 10(100): eadp7821.

[4] Hu W, Lum G Z, Mastrangeli M, et al. Small-scale soft-bodied robot with multimodal locomotion[J]. Nature, 2018, 554(7690): 81-85.

[5] Li L, Wang S, Zhang Y, et al. Aerial-aquatic robots capable of crossing the air-water boundary and hitchhiking on surfaces[J]. Science Robotics, 2022, 7(66): eabm6695.

[6] Bai S, Pan Q, Ding R, et al. An agile monopedal hopping quadcopter with synergistic hybrid locomotion[J]. Science Robotics, 2024, 9(89): eadi8912.

[7] Xu S, Hu X, Yang R, et al. Transforming machines capable of continuous 3D shape morphing and locking[J]. Nature Machine Intelligence, 2025, 7(5): 703-715.

[8] Sihite E, Kalantari A, Nemovi R, et al. Multi-Modal Mobility Morphobot (M4) with appendage repurposing for locomotion plasticity enhancement[J]. Nature Communications, 2023, 14(1): 3323.

[9] Zhao D, Luo H, Tu Y, et al. Snail-inspired robotic swarms: a hybrid connector drives collective adaptation in unstructured outdoor environments[J]. Nature Communications, 2024, 15(1): 3647.



**END**




![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/yNRdEJS7ubS4mq6X1ibsmx8kKUFT8ibqSAGY5FiaKlGQicLRJH5FT8tryUrcRa9oxkxsibXeI92Ouw6OJjuCQKLmIKby3iaO4iaKeKfpglhpbMZxWk/640?wx_fmt=jpeg&from=appmsg#imgIndex=21)



**工业机器人企业**

[埃斯顿自动化](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3257349072756359172&scene=21#wechat_redirect) | [埃夫特机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3257353338380304393&scene=21#wechat_redirect) | [法奥机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3286449098241556485&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [越疆机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3286454601252290560&scene=21&token=1458304635&lang=zh_CN#wechat_redirect) | [节卡机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3254648088418533381&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [松灵机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3456109198994046982&token=889435696&lang=zh_CN#wechat_redirect) | [珞石机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3254663109932433416&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [阿童木机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288958340173348870&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [极智嘉](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4048659988267401228#wechat_redirect) | [海康机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4219865391746514956#wechat_redirect) | [翼菲科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4553352285208297478#wechat_redirect)

**服务与特种机器人企业**

[亿嘉和](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288954695272841217&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [晶品特装](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288964066522382339&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [七腾机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3403025009986240513&token=889435696&lang=zh_CN#wechat_redirect) | [史河机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293738917174919171#wechat_redirect) | [普渡机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288953102410399750&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [施罗德机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4186250661224251397#wechat_redirect) | [库犸科技MAMMOTION](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4414226529300381703#wechat_redirect)

**人形机器人企业**

[优必选科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288979195142029317&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [宇树](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288981594753679361&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [云深处](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288967267548086273&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [星动纪元](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293732259774283776&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [伟景机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288992104941305856&token=889435696&lang=zh_CN#wechat_redirect) | [逐际动力](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288985434051788809&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [乐聚机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293731727953313793&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [大象机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3286447187786416133&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [魔法原子](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3927076758582722566#wechat_redirect) | [众擎机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3568736079773564935&token=889435696&lang=zh_CN#wechat_redirect) | [帕西尼感知](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3528541306714325007#wechat_redirect) | [赛博格机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4079468489180708880#wechat_redirect) | [数字华夏](https://mp.weixin.qq.com/s?__biz=MzI5MzE0NDUzNQ==&mid=2650365089&idx=1&sn=ff85dc766e7fd32ad5a38f96a91d6ae0&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [傅利叶智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288983966297047042#wechat_redirect) | [天链机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3467475229234692100#wechat_redirect) | [开普勒人形机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3782215292189507584&token=889435696&lang=zh_CN#wechat_redirect) | [灵宝CASBOT](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3867823579383201806#wechat_redirect) | [清宝机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4127283297804091396#wechat_redirect) | [浙江人形机器人创新中心](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3867825542837567498#wechat_redirect) | [动易科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4038448117375565829#wechat_redirect) | [智身科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4118149398758948881#wechat_redirect) | [PNDbotics](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4379070829221855242#wechat_redirect) | [卓益得机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4233170348381831192#wechat_redirect) | [鹿明机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4247165971359596550#wechat_redirect) | [擎朗智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3925601359369601037#wechat_redirect)| [伽利略GALILEO](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4272156348059484177#wechat_redirect) | [松延动力](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4220207003328577542#wechat_redirect) | [天机智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4456105130681286660#wechat_redirect) | [卧安机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4480879061984198662#wechat_redirect) | [理工华汇](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293731065135841287#wechat_redirect)

**具身智能企业**

[跨维智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3506492927482265606&token=889435696&lang=zh_CN#wechat_redirect) | [银河通用](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3560375973176541192&token=889435696&lang=zh_CN#wechat_redirect) | [千寻智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3583630309381767178&token=889435696&lang=zh_CN#wechat_redirect) | [灵心巧手](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3528517636042260481&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [睿尔曼智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3286456343213850632&scene=21&token=2007103472&lang=zh_CN#wechat_redirect) | [微亿智造](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3676539632905977857&token=889435696&lang=zh_CN#wechat_redirect) | [推行科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3853054033649565699&token=889435696&lang=zh_CN#wechat_redirect) | [中科硅纪](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3925610458861797378#wechat_redirect) | [枢途科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3764538143521472514#wechat_redirect) | [灵巧智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4019816341174485000#wechat_redirect) | [星尘智能](https://mp.weixin.qq.com/s?__biz=MzI5MzE0NDUzNQ==&mid=2650377149&idx=1&sn=57b82dd2669354fe6233a58a639c7c71&scene=21#wechat_redirect) | [穹彻智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3695298879357550600#wechat_redirect) | [方舟无限](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3541439704800280581#wechat_redirect) | 科大讯飞 | [北京人形机器人创新中心](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3867826685114318856#wechat_redirect)| [国地共建人形机器人创新中心](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4406986649210060801#wechat_redirect) | [戴盟机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3732838062997209090#wechat_redirect)| [视比特机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4069131772078850060#wechat_redirect)| [星海图](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3553073620187676675#wechat_redirect) | [月泉仿生](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3712634851543842821#wechat_redirect) | [零次方机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3845725810946834432#wechat_redirect) | [中科深谷](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288997360286777350&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [智平方](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3887683097352994816#wechat_redirect) | [大咖机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4207993344179306505#wechat_redirect) | [灏存科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4217231638863806480#wechat_redirect)| [具识智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4309809120817135624#wechat_redirect) | [Xynova曦诺未来](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4115724607930236932#wechat_redirect) | [非夕科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3286451235054895108&scene=21&token=549237372&lang=zh_CN#wechat_redirect) |[未来动力](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4329049620250050569#wechat_redirect) | [博登智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4406896146061852676#wechat_redirect) | [千诀科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3889399441580621834#wechat_redirect) | [灵生科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3700743633692098562#wechat_redirect) | [集萃智造](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4054349753574752272#wechat_redirect) | [欣佰特科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4421124848006070273#wechat_redirect) | [晨昏线科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4430046512785784841#wechat_redirect) | [Dexmal 原力灵机](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4435836540514336768#wechat_redirect) | [优理奇](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4126017250866233362#wechat_redirect) | [自变量](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4544645769735290881#wechat_redirect) | [睿研智控灵巧手](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4553444607459704832#wechat_redirect) | [启物科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4537087451391180801#wechat_redirect) | [RoboScience机器科学](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4579500655438053378#wechat_redirect) | [中科第五纪](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4585265868573622274#wechat_redirect) | [临界点](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4585266256546742273#wechat_redirect)| [当虹科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4588495256077320193#wechat_redirect)| [桥介数物](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4592324947435438082#wechat_redirect) | [Vbot维他动力](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4592543154020663298#wechat_redirect) | [他山科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4592543623866597381#wechat_redirect) | [具脑磐石](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4592641269830631425#wechat_redirect) | [优艾智合机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4592641686861889536#wechat_redirect) | [智行腱](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4593843994908016641#wechat_redirect) | [阿米奥机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4593881097368879106#wechat_redirect)

**医疗机器人企业**

[元化智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293696134166822923&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [天智航](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293721766665863172&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [思哲睿智能医疗](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293724274507333641&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [精锋医疗](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293725067264344065&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [佗道医疗](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293726173956620290&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [真易达](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293690023988641800&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [术锐®机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293727229444833285&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [罗森博特](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293728506727841795&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [水木东方](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3867537296475815940#wechat_redirect)｜[康诺思腾](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4186246230193733632#wechat_redirect) | [迪视医疗](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3783757252540858369#wechat_redirect)

**上游产业链企业**

[绿的谐波](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288991540572536835&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [因时机器人](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288990101775269890&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [坤维科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3350322715362279430&subscene=159&subscene=&scenenote=https%3A%2F%2Fmp.weixin.qq.com%2Fs%2FsSxMupFE9pStdngL2V_iUw&nolastread=1#wechat_redirect) | [脉塔智能](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293732796057993221&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [青瞳视觉](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3288995537375150084&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [本末科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3286444169649143812&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [蓝点触控](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293735422497603591&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | 鑫精诚传感器 | [BrainCo强脑科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3867833261128679426#wechat_redirect) | [宇立仪器](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3695294705689526278#wechat_redirect) | [极亚精机](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3782219886042906625&token=889435696&lang=zh_CN#wechat_redirect) | [思岚科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3705062863023472640&token=889435696&lang=zh_CN#wechat_redirect) | [神源生](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3969551293420404743#wechat_redirect) | [非普导航科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3867821529895272457#wechat_redirect) | [因克斯](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3293734699584143361&scene=21&token=889435696&lang=zh_CN#wechat_redirect) | [巨蟹智能驱动](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3467268504405671937#wechat_redirect) | [凌云光 元客视界](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4139975363126362115#wechat_redirect) | [璇玑动力](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3959060537383583757#wechat_redirect)| [意优科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3887722376775073798#wechat_redirect)| 瑞源精密 | [灵足时代](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=3635794238312120322#wechat_redirect) | [HIT华威科](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4047668338367922180#wechat_redirect) | [星汇传感](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4176334678934159371#wechat_redirect) | [凌迪科技](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4382101213241098240#wechat_redirect) | [泉智博](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4431462829355040773#wechat_redirect)| [CubeMars机器人动力](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4463304443329101827#wechat_redirect) | [旺龙机器人乘梯](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI5MzE0NDUzNQ==&action=getalbum&album_id=4583649631477284867#wechat_redirect)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/yNRdEJS7ubQhYKibr2meukpqHpOuFeT6VmAlarC9jeY88lW0ox9UXZefgw0yTOQQTtI6KEiaFicayOibBIxYhYaN1xSvhiblCGcjhqRpxvYcQTNk/640?wx_fmt=png&from=appmsg#imgIndex=22)
