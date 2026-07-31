---
title: "Science Robotics最新综述:6国顶尖机构联合梳理腿式机器人的进展、挑战与机遇"
author: 机器人大讲堂
date: "2026-07-31 18:00:00"
source: "https://mp.weixin.qq.com/s/yFZs7SLN5naqty0PBTk0Xw"
---

# Science Robotics最新综述:6国顶尖机构联合梳理腿式机器人的进展、挑战与机遇

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/LJiau2qPWAcUB7NhD61gK7qibQdRHQvQ1KsvwsUpbAUepZeNzW349LvNXKzu9jH4ibhtKqJm9Tlz1AOr2tgibJEuQA/640?wx_fmt=png&from=appmsg#imgIndex=0)

曾经只存在于科幻叙事里的腿式机器人，如今已经能把包裹直接送到住户家门口，也能在搜救场景中钻进人类难以进入的空间。国际权威期刊《Science Robotics》最新发表的一篇综述，由苏黎世联邦理工、斯坦福大学、加州大学伯克利分校、爱丁堡大学、KAIST、NVIDIA、图宾根大学、马普所、牛津大学、莫纳什大学等十余所顶尖高校和机构的研究者联合完成，系统盘点了腿式机器人在硬件、运动控制、自主性、数据与应用五个维度的进展，也罕见地把伦理、经济与政策一并摊开来谈。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/yNRdEJS7ubROU0z7Uvse2LNoqeibtbFuMibIb3wxsHerxhdBtcA3Kdib0puWBicglNpF4J3ibPjib8MODaJmiayib8MxwpHsiakp1FHSnicRzBxjKfPic8/640?wx_fmt=png&from=appmsg#imgIndex=1)

本文主要梳理这份综述中清晰论述的内容:我们是怎么走到今天的,还缺什么,接下来该做什么,以及世界可能因此发生什么变化。

**01.**

**为什么非要装腿：从"行走卡车"到动态稳定的六十年**

技术上,腿式机器人的灵感来自动物的生物力学。与轮子相比，腿可以跨越障碍、攀爬楼梯,同时提供主动悬挂和动态稳定,而占地面积却很小。空中无人机受制于短促的续航,并且在空中操作物体本身就很困难;腿式系统则能长时间作业、承载重载,甚至完成精确的移动操作。更重要的是,它们独特的形态天生适合在以人为中心的环境中活动。

这条路走得并不短。上世纪六十年代,液压驱动的通用电气"行走卡车"(Walking Truck)是一台遥操作平台,目标是在复杂地形上搬运重物;七十年代末的俄亥俄州立大学六足机器人成为数字控制的试验台;在日本,加藤一郎是关键人物,他开发的 WABOT-1 等早期人形机器人聚焦于打造全尺寸拟人机器。真正的范式转折来自 Marc Raibert 的工作——研究重心从"静态稳定"转向"动态稳定"。此后本田推出系列人形机器人,其中 P2 成为第一台使用机载计算机行走的自持式双足机器人,也是后来那台家喻户晓的 ASIMO 的前身。到了本世纪初,波士顿动力的 BigDog 与 LS3 标志着动态稳定四足机器的成熟。

而过去十年,能力、多样性与可获得性出现了爆发式增长。

![Image](https://mmbiz.qpic.cn/mmbiz_png/yNRdEJS7ubS3kgN2IkLDFBNb9rQC7znCvHQUsniazROCcSH8oDRs6u8TibbWu19FPenWTJ91coLZr4wPdscNRFTVb6vRTic9teVztOX40zKFJ4/640?wx_fmt=png&from=appmsg#imgIndex=2)

**02.**

**硬件：让关节能被反推，是这一轮爆发的起点**

四足与双足是最主流的两种形态。四足机器人从静态稳定走向动态稳定,非常适合承载与操作任务,目前部分四足平台的负载能力可超过180公斤;人形机器人则拥有更大的操作工作空间,但由于更依赖动态稳定,其操作与行走都更具挑战。

执行器的核心要求是产生足够高的力矩。跑动时会产生巨大的关节力矩与地面冲击力,这要求很大的关节力矩;而在与地面瞬时接触的时刻,执行器还必须迅速响应巨大的冲击。这些需求指向同一个结论:需要低机械阻抗、高可反驱性的高力矩执行器——这与工业机械臂的设计取向完全不同。

历史上的电机为了效率和制造便利被设计成高转速输出,力矩不足,只能配大减速比。但高减速比必然带来摩擦、巨大的折算转子惯量和更高的机械阻抗,结果是关节无法反驱,没有额外传感器或机构就做不了关节力矩控制,既容易被地面冲击损伤,也限制了动态运动能力。为此,串联弹性执行器(在齿轮组之后加入弹簧)被提出并进入 StarlETH 等设计,但柔性会降低控制带宽:共振拉低了可实现的闭环刚度,高频力矩跟踪变差,而且难以建模。

液压方案功率密度高,MIT 腿部实验室的早期机器人和波士顿动力的 BigDog、Atlas 系列都把泵、阀与油箱集成进机身,展示了极具冲击力的动态动作。但液压系统机械复杂、生产与维护成本高、效率低,还有噪声和漏油等现实问题,兴趣因此下降。

真正的转折点是:用定制电机实现高力矩、可反驱的电驱动方案——大气隙半径、短轴向长度的电机拓扑,配合低减速比传动。这带来了低机械阻抗和高反驱性;由于传动高度"透明",关节力矩可以近似为电机电流乘以力矩常数与减速比的线性关系,于是既不需要额外的力/力矩传感器,也不需要复杂控制回路,就能做到精确、高带宽的力矩控制。再加上执行器设计的开源发布(如 Katz 的低成本模块化执行器),四足与人形设计随之井喷,宇树的四足与人形产品线、UCLA 的 ARTEMIS 都属于这一设计哲学的产物。

与此同时,传感器小型化让机器人得以"全身武装":关节编码器与惯性测量单元构成本体感知,融合后可估计身体姿态与运动;LiDAR 提供稠密三维点云,RGB-D 相机结合视觉与深度,用于地形建图与环境重建,让机器人能预判地形、规划落足点、规避障碍;接触与力传感提供足-地交互信息,简单的二值传感器可检测触地与离地,多轴力/力矩传感器则量化地面反作用力与力矩,可用于改进滑移检测与意外碰撞识别。触觉皮肤的研究有望让机器人感知全身接触、改善导航,但耐久性与集成难题限制了它的普及。

至于生物启发的人工肌肉、腱驱动、以及从刚性走向柔顺形态的探索,综述的判断很客观:潜力在形态紧凑、柔软和力重比上,但制造、耐久与控制问题仍在,尚不足以进入商业系统。

**03.**

运动控制：强化学习解决了四足行走，但没解决双足

传感器更新率、任务延迟要求与算力需求的差异,自然导致了模块化的控制结构:执行器控制约200至1000赫兹,运动控制约50至200赫兹,而负责场景理解、规划与导航的高层模块通常低于30赫兹。

传统方法把运动拆解成手工设计的技能。早期依赖静态稳定的质心判据,要求质心投影严格落在支撑多边形内;随后零力矩点判据放宽了限制;线性倒立摆、弹簧负载倒立摆等降阶模型的引入使动态行走成为可能;模型预测控制进一步实现了行走、跳跃与奔跑。但实时性要求限制了模型的表达能力,面对真实世界的随机性与不确定性,鲁棒性会打折,因此才有了关节级反射等启发式手段。如今 MPC 更多地与强化学习结成混合系统,在训练与部署中承担长时程规划。

完全在仿真中用强化学习训练策略,已经成为开发运动控制器与摔倒恢复策略的主流范式,它把数以千小时的仿真经验蒸馏成一个鲁棒策略,训练算法以 PPO 为主。

成功背后是一系列关键设计:仿真环境要覆盖部署时可能出现的状态;观测空间被刻意限制,通常偏好几何外感知而非 RGB 图像;策略输出执行器位置目标、再由阻抗控制转成力矩,而不是直接预测力矩,这样更利于探索。但真正的关键是弥合"仿真到现实"的鸿沟:执行器模型的系统辨识让策略得以零样本迁移到真机;域随机化带来对变化的韧性,代价是最优性——学到的行为是一个"平均解",并非为特定机器人与环境定制;域自适应策略学习则在线持续估计系统参数,让机器人适应环境、更好应对噪声与不确定性。

值得注意的是规模:为了满足严苛的实时要求并便于泛化与优化,运动策略普遍使用低容量模型,如多层感知机或循环网络, 绝大多数参数总量在一千万以下——与动辄数十亿参数的视觉、语言模型不是一个量级。

强化学习的老问题依然存在:需要大量奖励塑形(可借助对称性先验、中枢模式发生器、约束强化学习、自动课程设计缓解);PPO 的样本效率低导致探索困难,于是出现了利用特权信息的师生方法;而离策略与离线强化学习等更省样本的替代方案,在这个领域尚未被证明有效。此外,像速度跟踪这样的目标本身可能过于死板,阻碍了跨越沟壑等复杂行为的涌现,因此更稀疏的位置奖励、落足点跟踪,以及用人类或动物参考数据做模仿学习,成为有希望的方向。把视频或动捕的参考动作重定向到肢长、质量分布、自由度都不同的具体机器人身上,通常被建模为逆运动学优化问题;离线重定向比实时重定向能获得更高的物理真实性与动作保真度,重定向后的动作再由强化学习控制器跟踪。模仿人类行为不只是简化奖励调参,还能提升社会接受度,让人机交互更自然。

当下的痛点很具体:很多方法需要流程复杂的多阶段训练,而把多个技能或策略蒸馏进单一策略仍是未解难题。形态选择也存在根本性权衡:增加腿数提高协调难度但扩大了恢复动作的空间;四足减少了人形所需复杂连杆机构带来的机械复杂度;双足依赖动态稳定,在崎岖地形上格外困难,但占地面积更小,而且人形构型便于规模化采集人类参考动作数据。

安全性上,综述坦率地指出:控制策略的验证与认证仍是开放问题。汉密尔顿-雅可比可达性分析、控制屏障函数、李雅普诺夫稳定性方法都有应用,但"实践中能实现的运动行为"与"能给出形式化安全保证的行为"之间仍有很大差距,尤其是在部分可观与复杂非线性环境动力学下定义"安全状态"依然困难。

从平坦地形到崎岖地形,再到不连续地形,如今的焦点正从纯几何感知转向融入视觉与语义理解。人类用眼睛规划接下来几步的落足点,把注视投向地形中最相关的区域,并解读各种影响行走方式的线索。综述为这一新兴范式起了个名字:"灵巧语义化运动"(dexterous semantic locomotion)——机器人不仅要看几何,还要预判与地形的交互(松动的石块、树枝),理解环境以规划精确细腻的运动响应,并在意自己施加于环境的力。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/yNRdEJS7ubQ7svjKsutGTyEYib6ItG0yVB4CicicRNcL4lFljTHSmWjeXuBPVSufuRoqcibMggHMhTFp6UPt039TO9JxFzibsV29qJ2arg0J8KPE/640?wx_fmt=png&from=appmsg#imgIndex=3)

**04.**

**自主性:分层拆解,还是端到端融合?**

自主系统让机器人真正会"自己走"。传统做法是把任务分解为多个子系统,用清晰定义的接口做分层数据处理。感知子系统让机器人理解自身状态与环境,其状态估计可以包含接触状态、外力和地面反作用力。位姿与速度估计(即里程计)一直是开发重点,而腿本身就是传感器,通过腿式里程计提供运动学信息;现代状态估计依赖多传感器融合,从运动学-惯性方案发展到相机、LiDAR、雷达等模态的互补集成。

腿式机器人的语义理解不止于几何和标签,导航系统还必须把硬件和控制器纳入分析——这就是可通行性估计:给地形图或占据栅格中的区域打分,评估机器人穿越的难易,分数通常由人工标注或自监督训练的模型给出,可直接被传统运动规划器使用。但结构化表征恰恰难以整合机器人状态与本体感知信号,这也催生了不显式建立场景表征、而是学习前向动力学模型来隐式编码可通行性的工作。

用强化学习训练导航策略同样有效,可以直接从原始传感输入或地图表征中学习。不过,仅仅把导航模块换成学习版本,导航与运动依然是解耦的,这限制了机器人在更复杂场景中的部署。

于是出现了另一种思路:融合模块,打破分层信息处理、手工接口和启发式表征的限制。机器人跑酷(parkour)是第一块试验田——在有沟壑、高障碍、狭窄空间的不连续地形中奔向目标,传统解耦极可能失败,近期工作通过分层强化学习与潜在场景表征成功协调了不同运动行为。

在整个自主系统中,从语言学到的抽象正扮演关键角色,支撑对边缘情况的推理与高层规划;学到的语义表征以场景图等多种记忆形式维护,支持抽象概念与原始感知信息的分组与检索。扩散模型、视觉-语言-动作模型和大行为模型已展现出泛化与灵巧操作能力,在大规模互联网与机器人遥操作数据上预训练后,仅凭少量演示就能学到有效行为。但综述给出了一个冷静的判断:目前基于模仿学习范式的导航方法,在鲁棒性和可解释性上仍落后于经典方法,并且相比仿真中训练的强化学习策略,难以有效整合本体感知信息。

导航基础模型让人们能用自然语言给机器人下指令,展示了把原本分离的模块融进单一架构的潜力。然而硬实时要求仍在持续塑造架构设计:各组件之间的最优接口、控制频率与功能边界应当如何划分,甚至是否需要划分,仍在激烈争论中。哪些先验应该显式写进系统,哪些交给端到端的可扩展学习目标和海量数据,也依然是开放问题。

![Image](https://mmbiz.qpic.cn/mmbiz_png/yNRdEJS7ubQkUQ9nyzZcicn6iaribj0Fhq4sWDclePNVBp94NmdYdevzcSLR0hFW1Sj1MwsXOP5ibMUpHSYqhQ4pbFIdXKa4zeHiahGeHdMic1h3Q/640?wx_fmt=png&from=appmsg#imgIndex=4)

**05.**

**数据:比自动驾驶更贵、更难的一件事**

数据是根本挑战。腿式机器人的数据深度绑定于本体、传感模态、驱动动力学与环境条件,高度多模态、经常异步,还受到时间相关噪声和系统延迟的影响。这使得大规模、可靠的数据采集异常困难,尤其是在机器人必须稳健应对的那条"长尾"上。

GPU 加速仿真框架的进步,让消费级硬件也能做大规模策略学习。仿真能快速生成带特权监督信号的数据,但保真度与真实感往往以计算吞吐为代价,物理建模与视觉渲染中的近似造成了 sim-to-real 域间隙。完美仿真依然遥不可及,但广泛的域随机化、学习残差模型和精确的系统辨识,已被证明能有效收窄这一间隙——因此绝大多数底层运动、导航和全身控制策略如今主要在仿真中训练,依赖本体感知信号和深度图、语义图这类中层视觉表征。

但仿真远不够用。设计环境、建模系统不确定性以覆盖足够多样的场景,仍是一个手工且脆弱的过程;足部缠入植被、在可变形地形上行走、语义丰富的杂乱环境,都还超出当前仿真器的能力范围。生成高保真 RGB 数据的算力代价很高,而更根本的瓶颈是模拟与视觉数据相对应的正确物理交互,这最终导致了尚未解决的视觉sim-to-real鸿沟。

真实数据要么靠直接遥操作采集,要么间接来自人类或动物示范。从视频或动捕系统提取的动作参考为运动控制提供了宝贵先验,简化奖励设计、稳定模仿学习。而与自动驾驶相比,腿式机器人的数据采集要昂贵得多——需要把机器人部署到复杂非结构化地形,而不是马路上。GrandTour 与 Sub-T 等数据集已开始填补空白,覆盖从城市街道到森林与工地、不同光照与天气的场景,但规模与多样性仍远不及自动驾驶,大规模部署中的数据完整性与校验也依然棘手。

未来的路径已经清晰:神经渲染与神经场景重建可能带来大规模、高保真的照片级渲染与网格化;神经增强物理引擎有望更好地刻画复杂动力学、覆盖今天建模不佳的长尾失效模式;可微仿真的改进支撑更高效的学习算法;生成模型则有望自动构建多样、语义丰富的环境,减轻人工工程负担。同时,腿式机器人更广泛的部署(初期可能借助人工引导的遥操作)将自然扩充真实数据集,形成类似视觉与语言领域的互联网级训练范式。

一个容易被忽略的瓶颈是:大多数 sim-to-real 研究依赖定制硬件,缺乏标准化的共享基准与实践,系统性比较很困难;而纯仿真比较又无法体现策略迁移到真机的挑战。解决它需要超出腿式机器人范畴的更大努力。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/yNRdEJS7ubRU2RpX9bdRm7WyR2gkT4azWoJcLD0sBhuy12raFcWMmFPmQYa0c6k1bnzOiahXib6y57wkupx53BdDIQfGlUVdDZuoqozmXtu8s/640?wx_fmt=png&from=appmsg#imgIndex=5)

**06.**

**它们已经在哪里干活了**

巡检与监测是四足平台的主要商业场景。ANYbotics的ANYmal已部署于海上油气设施,波士顿动力的Spot被用于矿山以及包括核电在内的能源设施的巡检监测;工地的长期安全与进度监测是另一个正在落地的挑战性应用。

农业与林业这类开阔野外场景带来额外难题——缺乏通信基础设施、缺乏参考地标、环境条件不断变化,因此多数演示仍停留在科研项目中,例如农业监测、林业采伐与森林资源清查。这些部署都集中在四足平台上,因为它们天生比人形更稳定。

配送近来成为新的商业应用,瞄准"最后一公里"。RIVR 等公司已在真实住宅完成包裹与餐食配送,展示了在多样城市环境中结合遥操作与共享自主完成"货车到门口"投递的可行性。

始于 2022 年的"人形热潮",目标是把这些平台送进工厂,支撑制造与仓储搬运。尽管这在过去被视为腿式平台"不合适"的任务,私营部门在打造能用同一套硬件软件承担多样任务的通用机器人上进展迅速;商用试点人形机器人承诺在2026 年于私人家庭环境中完成简单任务。

另一个长期愿景是协助人类照护者完成日常生活活动。人形形态便于融入家居环境、支持直觉交互,日本因人口老龄化在这一方向领先。但这个领域的主要挑战不在移动,而在柔顺操作、人身安全交互,以及场景与意图理解。

国防与灾难响应是另一条主线。军事野外作业中"机器骡"搬运重载的需求,正是 BigDog 与 LS3 的开发动因,那一时期也产生了腿式平台有记录以来最大规模的野外部署。用人形机器人测试防化装备的需求催生了 Atlas,它后来参加了 DARPA 机器人挑战赛;该赛事聚焦灾难响应任务,启发了消防等后续研究,但也暴露了当时人形机器人的硬件与控制局限,促成了 2010 年代后期向四足等更动态稳定平台的转向。2019 至 2021 年的 DARPA 地下挑战赛成为四足机器人成熟度的关键证明——它出现在几乎所有决赛队伍中。地下与战斗环境的共同点是对自主决策的极高需求:无线电可能被干扰,地下也难以建立高带宽通信;有线通信带宽够但范围有限、易受攻击。因此腿式机器人必须具备无需人在回路的高级自主能力。综述也直言:军事应用是目前最清晰的商业化路径之一,但这同时带来了严肃的伦理问题。

在科学侧,四足与双足机器人历史上被用作理解运动的物理模型,如今其机动能力使它们成为环境科学的数据采集工具(如土壤采样)和野生动物交互的监测工具。空间与行星科学自上世纪六十年代起就是腿式机器人的关键用例:NASA 的 LEMUR 项目研究了腿式平台在非结构化环境中行走、爬行与攀爬的适用性;欧洲空间局"太空资源挑战赛"展示了四足平台如何为洞穴、陨石坑等轮式巡视器不适合的场景提供替代方案——但热管理、能源管理与可靠性仍是重大挑战。

艺术与娱乐也是天然舞台。AIBO、QRIO、NAO 等消费平台被广泛用于社交互动与艺术表演;RoboCup 把研究挑战与年度世界比赛结合,并启发了 2025 年世界人形机器人运动会等新赛事。现代平台的硬件与控制鲁棒性提升,让它们开始成为流行影视角色的实体载体——迪士尼的双足机器人已成功在乐园与现场演出中部署,这也为探索超越行走和操作的表达性动作打开了空间,而这对人机互动与情感投入至关重要。

![Image](https://mmbiz.qpic.cn/mmbiz_png/yNRdEJS7ubT4rQcDSfSrHxERx0v8DwQeXibibiccfQR29Uia8g0OtPTC3mwewlu2kql6hvDvnANLxgic2mmwyXpGX7IpkLgMAvOGYv9IicHdUYn7U/640?wx_fmt=png&from=appmsg#imgIndex=6)

**07.**

**伦理:大部分问题不新,但因腿而更尖锐**

综述明确指出:腿式机器人引发的伦理问题,大体上与其他机器人没有本质区别,但由于其可能的应用场景,某些问题会以更强的力度出现。

机器人承担传统上由人完成的工作,会产生技术性失业风险。这值得惋惜,因为(某些)有酬劳动也让人有机会对社群做出有意义的贡献;岗位流失还可能扩大社会与经济不平等,引发社会动荡——这些担忧正是"全民基本收入"讨论的由来。

更深一层的问题是:关于机器人如何发展、由谁决定,本身就是伦理与政治问题。如果一位政治家宣布要彻底改变社会,公众会要求参与决策;而工程师及其资助者绝大多数是男性、来自相对狭窄的社会人口背景,这就把"这些项目在多大程度上拥有民主授权"的问题推到了前台。

人形机器人让"机器人管家"的梦想更近了一步,相关研究常以"人口危机"论证正当性——未来照护老年人的人手会更少。但综述提出两个未被回答的问题:老年人是否愿意被机器人照护?机器人提供的"照护"能否满足老年人的社会与情感需求?而在家庭、医疗与养老场景中,机器人传感器所采集数据的访问权与控制权问题格外尖锐。

给机器人装上腿,可能扩大它们在战争中承担的角色。批评者认为,军用机器人降低了杀戮的心理门槛,也降低了冲突的门槛;"自主武器系统"的使用会让责任归属变得难以厘清,并侵犯敌方战斗人员的人权。

还有一类问题关于我们自己。在机器人具备感受能力之前——如果那一天真会到来——人如何对待机器人,其伦理性质取决于这揭示了我们的什么、以及如何影响他人。外形像人或像动物的机器人更容易让用户产生情感联结,这可能带来益处也可能造成伤害;当机器人损坏或被摧毁、关系意外终止时,人可能受到真实的伤害。反过来,若伴侣机器人鼓励人们用与机器人的关系替代与人的关系,反而可能加剧社会孤立;而人从与机器人互动中学到的东西,也可能影响他们对待他人和动物的方式。踢一只机器狗所暴露的残忍,和踢一台机器人吸尘器并不相同;用户虐待"女性"机器人,则传递出他们对真实女性的态度。

综述特别提醒设计者关注机器人的种族与性别政治:历史上机器人最初被构想为"机械奴隶",因而被种族编码为黑色;而当代人形机器人的图像——即使实物未必如此——几乎总是白色或闪亮的金属表面,暗示它们如今被种族编码为白色。"人造妻子"的幻想在机器人学史上扮演的角色,也让性别政治变得复杂。机器人越是行走在我们中间,这些问题就越重要。

**08.**

政策与经济：从"会走"到"会社交"，可能只剩10到15年

腿式机器人既能在崎岖地形作业,又能进入为人类双腿设计的环境,这让它成为工作自动化的有力候选。当前商用的"只会走路"的四足机器人价格区间约在 3 万至 9 万美元(2025 年),服务于配送、巡检、娱乐和安防等细分市场,受电池限制(90 分钟至 6 小时)作业半径约 4 至 20 公里;而入门级小型四足与人形的起售价已分别低至 2700 美元和 4900 美元,提示着快速的商品化趋势。能力的阶梯式推进——从只会走路,到制造业中的操作,到家务操作,再到具备社交能力从而进入养老照护这类大市场——对应着不断攀升的市场潜力。

有研究估计,美国47%的就业面临来自新兴技术(含机器人)的高自动化风险;也有研究发现,每千名工人增加一台工业机器人,会使工资下降0.42%。但综述强调腿式机器人与工业机器人有明确差异:起初,它们更可能创造家用机器人、机器人最后一公里配送这类新市场,这既源于新颖性,也源于其能力局限;然而随着操作与社交能力增强,它们可能成为人类劳动的直接替代者。关键区别在于:工业机器人只影响制造业约 7.5% 的劳动者,而腿式机器人有潜力冲击服务业——后者在一些发达经济体中占用了约 80% 的劳动力。

全球监管路线正在分化。欧盟《人工智能法案》自 2024 年 8 月生效,按风险等级分类,对高风险应用要求广泛合规;日本"社会 5.0"采取推动姿态,投资打造机器人协助养老与制造的"超智能社会";中国结合积极的产业政策与明确目标,工业和信息化部提出 2025 年实现人形机器人批量生产、2027 年达到世界领先;美国则缺乏综合立法,依赖分行业指引。

这种分化在经济上很可能是重要的。国际机器人联合会报告称,全球有 428 万台工业机器人在运行,其中中国安装 276,288 台(占全球总量 51%),亚洲占全部部署量的 70%;中国的工业机器人密度已达每万名员工470台,超过德国与日本,而亚洲与西方机器人平台之间存在实质性价差。这些事实表明,协调一致的国家战略与本土供应链能带来竞争优势。

综述提出四项政策优先事项。其一是基于能力的监管:要求应随机器人能力提升而升级——只会走路的系统监管可最小化,特定领域操作能力适度监管,通用操作与社交能力则需全面监管;这既避免扼制创新,又能在风险上升时保障安全;欧盟的风险分级提供了基础,但可能需要校准,以免让西方制造商在与亚洲竞争者的较量中处于劣势。其二是国际协调,当新能力带来安全风险时尤为必要,其中必须特别提及"致命性腿式机器人"的可能,联合国自主武器框架提供了治理范本,而国际标准也须持续扩展。其三是战略性产业政策:西方国家若不干预,存在技术依赖风险;但支持重点应是研发投入、制造基础设施与技能培养,而非保护主义。其四是前瞻性劳动力计划,必须认识到调整窗口被压缩了——不同于以往横跨数代人的自动化浪潮,从"机器人会走"到"机器人具备社交能力"这一进程可能只需 10 到 15 年,因此需要在每个能力阶段识别并培养与之互补的人类技能:今天是机群管理与技术支持,明天是人机团队协作。

**09.**

**结论：四足行走已解，前沿转向语义、灵巧与治理**

综述最后给出的判断相当明确。

在硬件上,由电磁驱动推动的腿式硬件已成熟到具备广泛适用性。四足机器人已在多家公司实现商业化,形态与设计正在收敛;人形硬件在学术、政府与产业投资激增下快速演进,多种竞争性设计并存,量产已在进行或计划于2026年落地。

在算法上,强化学习让四足行走成为一个可解的问题,在地形几何已知时能跨环境稳定工作,而前沿已转向语义理解与灵巧交互。对双足机器人而言,纯反应式行为不够——许多任务需要精确落足与细致的环境理解。应对这些挑战,可能需要重新思考 sim-to-real 范式,超越刚体仿真,发展新的学习型控制算法;而由于这些"灵巧性"问题与操作领域高度同构,跨领域互相促进的机会也随之打开。

在自主性上,它是释放巨大经济与社会价值的基础,而它将由学习与数据驱动,超越遥操作和脆弱的启发式方法。通向自主与通用机器人的核心未解难题,是如何从多样数据源中学习、并注入恰当的归纳偏置以应对真实世界的复杂性——这将决定未来部署的自主架构长什么样。此外,持续的投入加上其通用属性,让腿式机器人处于一个独特位置:弥合数字AI智能体与物理世界之间的鸿沟,为智能在真实交互中"落地"提供基础。

综述也提醒了一个反直觉的观察:今天很多人形机器人在平整厂房地面上的工业应用,其实并不需要高级移动能力,但这些部署可以是通向通用机器人的重要第一步;当规模经济与量产叠加上来,这类机器人甚至可能在最初看起来不合逻辑的应用中成为可行方案。

而在技术之外,作者的态度很克制也很坚决:要安全负责地把这项技术引入社会,需要基于能力的政府监管、国际协调、战略性产业政策与前瞻性劳动力计划;伦理考量必须体现在立法之中;开发者与研究者应对腿式机器人可能引入的隐性偏见与潜在危害保持警惕。归根结底,我们必须认真面对那个关键问题——谁拥有开发、部署与控制这些技术的权力。

论文地址：

https://www.science.org/doi/10.1126/scirobotics.aee0787



**END**




![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/yNRdEJS7ubS4mq6X1ibsmx8kKUFT8ibqSAGY5FiaKlGQicLRJH5FT8tryUrcRa9oxkxsibXeI92Ouw6OJjuCQKLmIKby3iaO4iaKeKfpglhpbMZxWk/640?wx_fmt=jpeg&from=appmsg#imgIndex=7)



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

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/yNRdEJS7ubQhYKibr2meukpqHpOuFeT6VmAlarC9jeY88lW0ox9UXZefgw0yTOQQTtI6KEiaFicayOibBIxYhYaN1xSvhiblCGcjhqRpxvYcQTNk/640?wx_fmt=png&from=appmsg#imgIndex=8)
