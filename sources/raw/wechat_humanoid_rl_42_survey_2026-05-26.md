---
title: 两万字长文，读懂人形机器人强化学习运动控制：42篇论文搭起的算法圣经
author: 具身智能研究室
date: "2026-05-18 09:00:00"
source: "https://mp.weixin.qq.com/s/hz9JXtJeUPRfUGzfD-pZuA"
---

# 两万字长文，读懂人形机器人强化学习运动控制：42篇论文搭起的算法圣经

我原来以为，人形机器人运动控制的核心是 **“让机器人动起来”**。

读完这 42 篇运动控制 / 移动操作论文之后，我的判断变了：

**真正难的不是做出一个动作，而是让动作在真实世界里变成可持续、可恢复、可精细交互的身体能力。**

至于 **VLA 和世界模型的调用**，更像是这层能力成熟之后的下一阶段。

这篇文章比较长，可以把它当成一张 **技术地图**。

单看 demo，人形机器人已经会跑、会跳、会踢球、会开门、会搬东西。

但把这些论文放在一起看，会发现真正缺的不是某一个动作，而是 **把动作稳定接入真实任务的系统能力**。

这也是我读完这批论文后最大的判断：人形机器人的竞争，正在从单点动作能力，转向身体能力的系统化。

## 写在前面：这 42 篇到底在讲什么

我没有按时间线写，也没有逐篇堆摘要，而是按 **五个问题** 组织：

•动作数据：人类动作、视频和遥操作，如何变成机器人可执行数据；

•参考跟踪：参考动作，如何通过 RL / tracker 变成稳定控制能力；

•感知运动：机器人如何在视觉参与下完成跑、跳、爬、踢等高动态动作；

•精细交互：行走、操作、视觉、接触和力控，如何进入同一个精细交互闭环；当这些能力稳定之后，语言指令、VLA 和世界模型又如何进一步调用它们；

•接触恢复：接触、柔顺、负载和失败恢复，如何成为真实部署的一部分。

我的整体判断是：

人形机器人运动控制正在从“动作跟踪”走向“精细交互控制”：机器人不仅要跟住动作，还要根据视觉、接触、力、负载和失败反馈持续调整身体。等这些能力足够稳定之后，才会进一步变成 VLA 和世界模型可以调用的身体接口。

## 一、为什么“会动”已经不够了

人形机器人最容易打动人的地方，当然是 **动作**。

一个机器人会跑，会跳，会翻越，会踢球，会开门，会抱人，视频一放出来就很吸引眼球。但这批论文放在一起看，会发现很多好看的动作背后都有同一个问题：

动作本身不是能力，动作进入真实世界之后才是能力。

同样是 **“翻越障碍”**，如果障碍物位置固定，机器人可以背一段参考轨迹；但如果障碍物高度、距离、角度有变化，机器人就必须根据视觉调整起跳时机、手落点、身体姿态。

同样是 **“开门”**，如果只看把手位置，那像是一个视觉定位问题；但真实开门还有门板旋转、把手约束、手腕姿态、身体跟随、重心保持和持续接触。

同样是 **“拥抱”**，如果只看手臂轨迹，机器人只要把双臂合拢；但真实拥抱必须控制接触力，否则不是帮助人，而是在压人。

所以我现在看人形机器人，会把问题拆成 **四层**：

•动作层：不成熟时只是“动作像”，但接触错位或落地不稳；成熟后应该物理可执行，并能随状态调整。

•感知层：不成熟时只是看见目标，但动作执行中不再纠偏；成熟后视觉、深度、本体感知都要进入闭环。

•接触层：不成熟时只会刚性追踪，碰到人和物就危险；成熟后要能控制力、柔顺、负载和反作用。

•任务层：不成熟时只能展示单个技能；成熟后要能组合成长任务。再往后，才是被语言 / VLA 稳定调用。

这也是这篇长文的主线。

## 二、第一组论文：动作数据、重定向、遥操作和交互保真

### 1GMR：重定向不是小工具，而是控制器上限

🔗 **项目链接**：https://jaraujo98.github.io/retargeting\_matters/

📄 **论文标题**：Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking

🏫 **机构**：斯坦福大学

GMR 的核心命题很直接：**retargeting matters**。论文指出，humanoid motion tracking policies 依赖人类动作重定向，但人和机器人之间存在 **embodiment gap**。重定向阶段留下的脚滑、不可行姿态、起始姿态不合理等问题，会直接影响后面的 **RL 控制器**。

![GMR 重定向流程](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEWWPJXxeLruGhWJfsr7Vhbwdmwoh10kZRlicGMBtLib8OxYA8zMhNySVarFW4icoBPNH3icSEVpicx5XDySx6SHEbB3u4GIwqjmnbicA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)GMR 重定向流程

这篇论文的价值在于，它没有把 retargeting 当成 **“训练前处理一下数据”** 的小步骤，而是系统评估了 **重定向质量对 motion tracking policy 的影响**。

论文比较了 GMR、PHC、ProtoMotions、Unitree 官方重定向等方法，并通过用户研究和 sim2sim 成功率说明：一个重定向结果既要接近源动作，又要适合训练控制器。

这里有一个很重要的观点：

后面的 RL 策略不应该被迫修补前面重定向阶段留下的所有错误。

如果参考轨迹本身物理上很糟，策略会陷入两难：严格跟踪会摔，保持稳定又会偏离参考。最后得到的控制器可能既不像人，也不够稳。

我把 GMR 放在技术地图的最前面，因为它提醒我们：**运动控制不是从策略网络开始的，而是从数据变成参考动作那一刻就开始了。**

### 2NMR：把重定向从优化问题变成学习问题

🔗 **项目链接**：https://nju3dv-humanoidgroup.github.io/nmr.github.io/

📄 **论文标题**：Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control

🏫 **机构**：南京大学；华为

NMR，也就是 Neural Motion Retargeting，进一步推进了 GMR 的问题。它认为传统优化式 retargeting 是非凸的，容易出现局部最优，从而带来 self-penetration、foot sliding、物理不可行等伪影。

![NMR 方法流程](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEW42FYAroawNIvbRCsbibyicQ3m1yQCa0ic9avvecFRcKxNkFUWobtMKY6piarOobWcJjT8zo8zMdu9Sa4ACXufQVAeBZ33I5wA4ib4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)NMR 方法流程

它的核心思路是：不要只把 retargeting 看成逐帧几何优化，而要把它看成 **数据分布学习**。论文提出 CEPR 等机制，用层次化和 VAE-based motion clustering 的方式，把大量动作组织成 latent motifs，再学习更稳定的重定向过程。

NMR 的重要性在于，它把 **“重定向质量”** 进一步推到了模型化层面。GMR 强调 retargeting matters，NMR 则进一步问：如果传统 retargeting 本身不稳定，能不能训练一个神经网络来学习更好的 retargeting 分布？

这件事对人形机器人很实际。因为未来动作来源不会只有干净 mocap，还会有 monocular video、生成视频、遥操作、互联网视频、混合数据源。数据越杂，传统优化式重定向越难完全靠手工规则解决。

我的判断**未来人形机器人动作数据越大，retargeting 越不可能只是工程脚本，它会变成一个可学习的数据生产模块。**

### 3OmniRetarget：真正要重定向的不是动作，而是交互关系

🔗 **项目链接**：https://omniretarget.github.io

📄 **论文标题**：OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction

🏫 **机构**：Amazon FAR；MIT；伯克利；斯坦福大学；CMU

OmniRetarget 的题目里有一个词很关键：**interaction-preserving**。

![OmniRetarget 论文图](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEXia1MBw206PaLQzH4ppASk8h6KMBicCg68AqTugsicr3pn7Dp2Bko3bNBVOpe2Dnqay9gGPPjliaf3lSzW2vlZaywRgbkaPmfjORE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)OmniRetarget 论文图

很多 retargeting 方法只关心 **人体姿态到机器人姿态的映射**。但在真实任务里，人不是在空气中摆动作，而是在和 **物体、地形、环境交互**。比如翻越障碍时，手和障碍物之间的接触位置很关键；搬箱子时，手、箱子和身体之间的空间关系很关键。

OmniRetarget 试图保留这些关系。它基于 interaction mesh，把 agent、terrain、objects 之间的空间结构纳入优化，通过最小化 Laplacian deformation，并满足机器人运动学约束，生成更适合 loco-manipulation 和 scene interaction 的数据。

这篇论文透露出一个趋势：**人形机器人后面的数据格式会发生变化。**

过去我们说“动作数据”，可能指的是人体关节轨迹。

但未来真正有价值的数据会是：

人-物-环境交互数据。

也就是说，机器人需要学习的不只是“手臂怎么动”，而是“手臂相对物体怎么动”；不是“脚怎么抬”，而是“脚相对台阶、沟壑、边缘怎么落”。

这也是 OmniRetarget 和 PHP、Deep Whole-body Parkour、HumanX、HDMI、WholeBodyVLA 等论文相互呼应的地方。

### 4GenMimic：生成视频不能直接给机器人用

🔗 **项目链接**：https://genmimic.github.io/

📄 **论文标题**：From Generated Human Videos to Physically Plausible Robot Trajectories

🏫 **机构**：伯克利；纽约大学；约翰内斯开普勒大学

这篇论文关注一个很值得追踪的问题：视频生成模型越来越强，可以生成各种人类动作视频，那机器人能不能直接执行生成视频里的动作？

![GenMimic 训练与测试流程](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEXLsw0hj6GdsKdzjvg79FOCIXggqWicVegVWBibXicfI2cMJ9EAAH327q53bmk2m8TAzGgiaeVZmxJMWEAib4Oq8cMyuqJD4V2icyfIY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)GenMimic 训练与测试流程

论文的答案是：不能直接执行，必须经过物理化。

生成视频可能在视觉上合理，但会有形变、遮挡、肢体穿模、动作不连续、人体比例不稳定等问题。对人眼来说，这些瑕疵可能可以忽略；但对机器人来说，一点姿态错误就可能变成无法执行的轨迹。

论文提出两阶段管线：先把生成视频 lift 成 4D human representation，再重定向到 humanoid morphology；之后用 GenMimic 这样的 physics-aware RL policy 来跟踪 3D keypoints，并引入 symmetry 和 keypoint-weighted rewards。

这篇论文放在这里很有意义。因为它连接了两个世界：

•生成式视频模型负责想象动作；

•物理强化学习负责把想象动作变成可执行轨迹。

我的判断**未来视频生成模型可能会成为机器人动作创意来源，但不会直接成为机器人控制器。中间必须有物理过滤和机器人化过程。**

### 5HumanX：从人类视频合成交互技能

🔗 **项目链接**：https://wyhuai.github.io/human-x/

📄 **论文标题**：HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos

🏫 **机构**：香港科技大学；上海人工智能实验室

HumanX 也从视频出发，但它关心的是 agile and generalizable humanoid interaction skills。它想把人类视频转成机器人可学习的交互技能，覆盖篮球、足球、羽毛球等任务。

![HumanX 论文图](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEWjbBI7XzE5W3iccLf5QEDBuO73fyqn3LagJmwQEekiaNicy0EaNsfJsT3x2wbiceuCoVXYibiah5Uk8Z43eBcCRU0HDcvtR8OUzQ6v0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)HumanX 论文图

它的框架包括两个部分：XGen 和 XMimic。XGen 用来从视频中合成物理合理的交互数据，并支持对象 mesh、尺寸、轨迹等增强；XMimic 则学习这些数据里的泛化交互技能。

这篇论文最重要的不是“机器人会打球”，而是数据生产思路：真实机器人交互数据贵，人类视频多，但视频里的动作和机器人身体不匹配，物体状态也不一定完整。HumanX 试图把视频中的人-物交互转换成机器人可以训练的数据。

它和 OmniRetarget 的区别在于：OmniRetarget 更像一个交互保留的重定向引擎，HumanX 更像从人类视频到机器人交互技能的完整管线。

我的判断**未来人形机器人要扩大技能库，不能只靠真机遥操作，必须学会把人类视频、生成视频和仿真数据转成可执行交互数据。**

这里不是否定遥操作，而是说遥操作不能孤立存在。它必须和重定向、仿真过滤、可执行性验证和真机控制结合起来，才会变成稳定的数据来源。

### 6HDMI：从人类视频学会真正的全身物体交互

🔗 **项目链接**：https://hdmi-humanoid.github.io

📄 **论文标题**：HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos

🏫 **机构**：CMU

HDMI 的全称是 HumanoiD iMitation for Interaction。它也从人类视频出发，但比 HumanX 更进一步，把重点放在 contact-rich humanoid-object interaction 上。

![HDMI 全身物体交互任务](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEXWjDicT9MCG5rR0x7Yyx7v2lSCxEvT90GU2kjfATApglIzItlApZwsKvqlFicHqYhwH1zTMO2soicZXVeIYqcpjHa14I6OxbzXYY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)HDMI 全身物体交互任务

这篇论文要解决的是：如果视频里有人推箱子、搬箱子、开门、滚球、放倒木板，人形机器人能不能把这些人-物交互变成自己可以执行的全身技能？

HDMI 的管线分成三步：先从单目 RGB 视频里估计人体和物体轨迹，并重定向成人形机器人参考数据；再用强化学习训练一个 robot-object co-tracking policy，让机器人同时跟踪自己的身体状态和物体状态；最后把策略零样本部署到真实 Unitree G1 上。

这里最关键的变化是，控制器不再只追踪“机器人像不像人”，还要追踪“物体有没有按预期被改变”。这就把模仿学习从 body motion tracking 推到了 object interaction tracking。

论文里有三个设计值得注意：unified object representation 让不同物体能进入同一套表示；residual action space 让策略在参考动作附近探索，降低高难姿态下的训练难度；general interaction reward 则鼓励机器人建立并保持合适接触，而不是只把末端移动到目标附近。

当然，HDMI 也有边界。它的真实部署仍然依赖 MoCap 提供机器人和物体状态，而且目前还是 one policy per skill。也就是说，它已经证明“从人类视频到真实全身物体交互”能跑通，但距离完全依赖机载视觉、统一多技能策略还有距离。

我的判断**HDMI 的意义在于，它把“从人类视频学动作”推进到了“从人类视频学习改变物体状态”。这比单纯模仿姿态更接近真实人形机器人操作。**

### 7H2O：遥操作不是遥控，而是高质量身体数据入口

🔗 **项目链接**：https://human2humanoid.com/

📄 **论文标题**：Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation

🏫 **机构**：CMU

H2O 的完整名字是 Human-to-Humanoid。它要解决的问题很直接：能不能让一个人通过自己的身体动作，实时驱动一个人形机器人做全身动作，而不是只控制手臂或轮式底盘。

![H2O 论文图](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEX24b7ZrlicJY3mQyPqY5DfxJrjGP86WYGia7Nt0qq7SqwJRXwpy1LzcjTestu5uGTXYF2G1gyCSx30dIup7AupvL1JoiarYicItq8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)H2O 论文图

这篇论文放在 HumanX 和 HDMI 后面很合适。HumanX / HDMI 关心的是从人类视频里生产可训练的交互技能，H2O 则更进一步，把“人类动作到机器人身体”的转换做成实时遥操作系统。

它的关键不只是 pose retargeting，而是把数据、仿真和真机控制接成一条闭环。

论文先从 AMASS 这类人体动作数据出发，经过 SMPL 到 Unitree H1 的重定向，再用 privileged motion imitator 过滤掉机器人动力学上不可执行的动作，形成 feasible motion dataset。之后再训练只依赖真实部署时可获得状态的 imitator，让机器人在真实世界里跟随人体动作。

不过 H2O 的真实实验里，机器人线速度估计仍借助 MoCap。论文也提到这一步未来可以用机载视觉或 LiDAR 里程计替代，所以它还不是完全摆脱外部定位系统的形态。

这里有一个关键转折：

遥操作不只是“人控制机器人”，也是在生产机器人可执行的身体数据。

如果遥操作只是把人的姿态硬塞给机器人，机器人很容易因为比例、关节限制、接触和动力学不匹配而失稳。H2O 的价值在于，它把中间的“可执行性过滤”放进系统里：先让仿真中的 privileged controller 判断哪些动作真的能被 H1 执行，再把这些动作蒸馏给可部署控制器。

未来人形机器人数据不会只来自离线 mocap 或互联网视频。高质量遥操作会变成一类非常重要的数据入口：人提供意图和示范，仿真负责筛选可执行性，控制器负责把动作落到真实机器人身上。

我的判断**H2O 的意义不是多了一个遥操作 demo，而是把“人类身体意图如何变成机器人可执行控制”做成了一条可训练的数据管线。**

### 8OmniH2O：把遥操作升级成通用身体接口

🔗 **项目链接**：https://omni.human2humanoid.com/

📄 **论文标题**：OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning

🏫 **机构**：CMU；上海交通大学

如果说 H2O 证明了实时 whole-body teleoperation 这条路能走，OmniH2O 就是在问：这条路能不能变成一个更通用的身体接口？

![OmniH2O 论文图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEUj6iaHH7uos3NDkZmw0x5B6xOMNk7GE7Ug4ZtR7xO2IccL7QETOrn6d9B2nuQb64npniaLePDXEtLWDGPDAxXUVibHmBKnRQeVek/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)OmniH2O 论文图

OmniH2O 的目标比 H2O 更大。它不只想让机器人跟随人的身体动作，还要支持灵巧手、移动操作、户外行走，以及多种输入来源：VR、RGB 摄像头、语言指令，以及 GPT-4o / diffusion policy 这类自主策略。论文里一个核心说法，是把 kinematic pose 当成 universal control interface。

这个接口判断很关键。

因为上层系统最终很难直接输出每个关节的力矩。无论输入来自人、视频、运动生成模型还是学习策略，它都需要一个中间身体接口，把“想做什么”转换成机器人可以稳定执行的全身姿态和运动目标。等这个接口足够稳，语言模型和 VLA 才有可能可靠地接上来。

OmniH2O 试图把这个接口做宽：上层可以给稀疏输入，底层控制器负责补齐全身协调、接触稳定和灵巧手动作。

它和前面几篇论文的关系也很清楚：

•GMR / NMR / OmniRetarget 关心动作怎么重定向；

•HumanX 和 HDMI 关心视频怎么变成交互技能；

•H2O 关心实时人到机器人的全身控制；

•OmniH2O 则进一步把这些输入统一到一个可扩展的 whole-body interface 里。

这篇论文让我更确信一点：人形机器人的“身体 API”不会只是一组关节命令。它更可能是一层介于上层智能和底层控制之间的 kinematic / skill interface。近期它首先服务遥操作、运动生成和自主策略；更往后，才会成为 VLA 调用人形机器人身体的接口。

当然，这条路线也有边界。接口越通用，越容易把复杂性压到底层控制器里；灵巧手、物体接触、长期移动操作和真实环境扰动，都会让“姿态接口”暴露出不够的地方。

所以 OmniH2O 的意义不是终结问题，而是把问题推进到更清楚的位置：人形机器人先要形成什么样的身体接口，未来上层智能才能调用它？

我的判断**OmniH2O 把遥操作从数据采集工具推向了通用身体接口。它提前暴露的问题不是“VLA 现在怎么接管身体”，而是“身体能力要先被封装成什么接口”。**

### 9TWIST：全身遥操作真正难的是让“人”和“机器人”共用一套身体接口

🔗 **项目链接**：https://yanjieze.com/TWIST/

📄 **论文标题**：TWIST: Teleoperated Whole-Body Imitation System

🏫 **机构**：斯坦福大学；西蒙弗雷泽大学

TWIST 和 H2O / OmniH2O 属于同一条技术线，但它强调的是另一个问题：**人形机器人的遥操作不能只控制手，也不能只控制底盘，而是要控制整具身体。**

这件事听起来很自然，但落到系统里很难。

很多移动操作系统会把问题拆开：底盘负责移动，手臂负责操作，头部或相机负责观察。这样做工程上可控，但人形机器人不是简单的“移动底盘 + 双臂机械臂”。

它的腰、腿、手臂、头部和重心会互相影响。人一边走一边伸手、一边转身一边操作时，身体协调本身就是任务的一部分。

TWIST 的思路是，用运动捕捉系统采集人的全身动作，再通过重定向和强化学习控制器，把人的运动变成机器人可以执行的全身动作。它不是只追求“姿态像人”，而是要让机器人在真实任务里完成全身操作、腿式操作、移动和表达性动作。

![TWIST 全身遥操作系统](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEVhuXhBBozdQtoDy3ZJTyq1nuRPXSlAfUbJv1dyFpve88ZvicicyOfWPHIsZbQnDIcofWCaic2Yibnu2xmvym6X85xrVzTp9ZS6Uko/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)TWIST 全身遥操作系统

这里值得注意的是，TWIST 把遥操作看成了两个东西：

•一个实时控制接口：人可以直接驱动机器人完成复杂全身动作；

•一个数据生产入口：系统可以采集人形机器人可执行的全身示范数据。

这和前面几篇论文是连在一起的。H2O 证明人到人形机器人的全身遥操作可以跑通；OmniH2O 进一步把它扩展成更通用的身体接口；TWIST 则强调，遥操作系统本身要服务于 whole-body imitation，也就是让机器人不仅“看起来像人”，还要在物理世界里完成动作。

不过 TWIST 也有一个明显边界：它依赖运动捕捉系统。对于实验室研究，这可以换来比较高质量的全身动作数据；但对于大规模数据采集，这套设备和场地要求会限制扩展。

我的判断**TWIST 的意义不是多了一个遥操作方案，而是把“全身遥操作”推进成了人形机器人数据和控制之间的桥。**

### 10TWIST2：真正的大规模人形数据，不可能只靠昂贵动捕房

🔗 **项目链接**：https://yanjieze.com/TWIST2/

📄 **论文标题**：TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System

🏫 **机构**：Amazon FAR；斯坦福大学；USC；伯克利；CMU

如果说 TWIST 证明了全身遥操作这条路能走，TWIST2 关心的就是：**这条路能不能规模化。**

这是人形机器人非常现实的问题。

未来真正缺的不是几个漂亮 demo，而是大量高质量、可复用、覆盖全身任务的数据。问题是，传统 MoCap 系统贵、重、依赖场地，很难像互联网数据那样大规模铺开。

TWIST2 的目标，就是把全身数据采集从“实验室动捕房”往“便携式采集系统”推一步。它用相对轻量的 VR 设备采集人的全身动作，并给机器人加入带主动视觉的头部系统，让数据里不仅有身体动作，也有和机器人视角相关的视觉信息。

![TWIST2 便携式全身数据采集系统](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWsciaHasqXvUX0EPQq5f9wQqml4fLMLHJTEyBroHUTfxOtdpQM79FecOq96RvvcAkunLz2g3KnUib3tX6q2xicsQyygrCRlrCKJc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)TWIST2 便携式全身数据采集系统

这个变化很重要。

因为人形机器人不是只需要手和腿的轨迹。它还需要知道：

•当前从机器人视角看到了什么；

•头部和视线如何随任务移动；

•手、脚、身体和物体之间如何协调；

•长任务里，观察、移动和操作怎么连续衔接。

TWIST2 相比 TWIST 的关键变化，不只是“设备更便宜”，而是把数据采集对象从单纯动作扩展到了更完整的人形机器人行为。论文强调 scalable、portable、holistic，三个词放在一起，其实指向的是同一个瓶颈：

人形机器人想训练大模型，不能只靠少量精致示范，而需要一套能持续生产全身数据的系统。

当然，便携式方案也有代价。轻量设备的姿态精度和稳定性，天然很难完全等同于昂贵 MoCap；高速动作、复杂接触、精细手部操作，也会继续考验数据质量。

但这不影响 TWIST2 的方向价值。

我的判断**TWIST2 真正重要的地方，是把人形机器人数据问题从“怎么遥操作”推进到了“怎么低成本、可扩展地持续采数据”。**

## 三、第二组论文：物理模仿、通用 motion tracker 和身体基础模型

### 11DeepMimic：很多人形控制论文的源头问题

🔗 **项目链接**：https://xbpeng.github.io/projects/DeepMimic/index.html

📄 **论文标题**：DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills

🏫 **机构**：伯克利；英属哥伦比亚大学

DeepMimic 是这批论文里最经典的一篇。它来自物理角色动画领域，但今天再看，它仍然是很多 humanoid motion tracking 工作的起点。

![DeepMimic 动作示例](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEV2JMJic2NjDia7ZvbBkEKqickr2JLibm9fCdick3AxQV0JibOh8HZVicd7rrrhv7uWY38EJhuwEib8KErNNmsflicXGwLBiclrDKaBHT3B8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)DeepMimic 动作示例

它要解决的问题很朴素：给定一段 reference motion clip，怎么让一个物理仿真角色学会执行类似动作，同时还能满足任务目标。论文用深度强化学习，把 imitation reward 和 task reward 结合起来，训练角色完成走路、跑步、翻滚、武术、投掷等动作。

DeepMimic 的关键贡献不是“让角色动起来”，而是提出了一种后来被大量继承的范式：

参考动作提供运动先验，强化学习负责把它变成物理可执行控制。

这个中间层对人形机器人非常重要。因为纯 RL 从零学复杂全身动作，探索成本太高；而纯 kinematic 动作又不保证物理可执行。DeepMimic 夹在中间：既利用人类动作数据，又让动作通过物理仿真闭环。

但它也留下了后面所有论文都在继续回答的问题：

•reference motion 从哪里来；

•人体动作如何转到机器人身体；

•参考动作有脚滑、漂浮和接触错误怎么办；

•多个动作如何放进同一个策略；

•真实机器人动力学和仿真不同怎么办；

•视觉、接触和任务目标如何进入闭环。

所以 DeepMimic 更像一个起点，而不是终点。

我的判断**人形机器人运动控制的第一代范式，是“模仿参考动作”；后面的工作基本都在扩展这个范式的边界。**

### 12OmniTrack：不要让控制器追踪错误参考

🔗 **项目链接**：https://omnitrack-humanoid.github.io/

📄 **论文标题**：OmniTrack: General Motion Tracking via Physics-Consistent Reference

🏫 **机构**：华中科技大学；BIGAI；上海交通大学

OmniTrack 关注 physics-consistent reference。它的出发点是：从人类动作或重定向数据里得到的参考轨迹常常不干净，可能有浮空、脚滑、不稳定接触和噪声。如果训练策略时强迫控制器去追踪这些参考，策略就会在“像参考”和“保持物理稳定”之间冲突。

![OmniTrack 论文图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXZovQKBkQSHjqfvpfiaGYRrYYYQgBtbF0R6w0wcsp3xjlUeSMmaQ6PLezxNboT39JvYb2SJ1ePz5Gz23Zqwt0VO54t3iasYYOjQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)OmniTrack 论文图

OmniTrack 的思路是解耦：先由 privileged generalist policy 生成更严格满足物理约束的参考，再训练通用跟踪器。也就是说，不是直接追踪 raw retargeted motions，而是先把参考轨迹变得更物理一致。

这篇论文的意义在于，它把 reference quality 放在了 motion tracking 系统中心。它不是让控制器无限背锅，而是先清理参考本身。

我的判断**未来通用 motion tracker 的核心能力之一，不是“什么都追”，而是知道什么参考值得追、怎么把参考变成可追。**

### 13Any2Track：跟踪能力必须和动力学适应能力绑定

🔗 **项目链接**：https://zzk273.github.io/Any2Track/

📄 **论文标题**：Track Any Motions under Any Disturbances

🏫 **机构**：清华大学；北京大学；Galbot；上海期智研究院

Any2Track 的目标是 **track any motions under any disturbances**。它认为基础型 humanoid motion tracker 不应该只会在干净环境里跟动作，还要能在真实扰动下工作，包括复杂地形、外力、模型误差等。

![Any2Track 论文图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEV6WU2CtJ5lfTWSNK9GR0r3ibOLGib9c2iclCCCPKHM6H7v7KRxGrKgrsGwuanEtZnxzmxMWMSETDKAAdpKx3sg2ic2kglvVtMBgXQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)Any2Track 论文图

论文提出两阶段 RL 框架，把 dynamics adaptability 作为额外能力注入 motion tracking。它不是单纯追求更低 tracking error，而是让策略在不同真实条件下保持动作执行。

这篇论文也让 “motion tracking” 这个词本身开始发生变化。

过去的 tracking 是：参考动作给定，我尽量跟。

现在的 tracking 是：参考动作给定，但 **地面、外力、动力学、接触都可能变化**，我要在不摔的前提下尽量完成动作意图。

这就更接近真实机器人部署。

我的判断**motion tracking 要真正通用，就必须同时学会“跟动作”和“适应动力学扰动”。**

### 14RGMT：参考动作不一致时，控制器要学会选择性相信

🔗 **项目链接**：https://zeonsunlightyu.github.io/RGMT.github.io/

📄 **论文标题**：Robust and Generalized Humanoid Motion Tracking

🏫 **机构**：北京理工大学；人形机器人（上海）有限公司

RGMT 是 robust and generalized humanoid motion tracking。它的核心是 dynamics-conditioned aggregation：用 causal temporal encoder 总结近期本体状态，用 multi-head command encoder 选择性聚合参考命令。论文还设计 recovery curriculum 和 annealed upward assistance force 来增强恢复和抗扰。

![RGMT 论文图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWCraysJCnC77w3oKNCQhsBcB0IJUibRB1snuibJlDAZwWgHZrJmo6wal0OQ2gdCtkuZrtvO1hNJV1Ml2fWI1S5Odwt9adhYVvts/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)RGMT 论文图

我对这篇论文的理解是：它在教控制器“不要盲目信任参考轨迹”。

参考动作可能来自不同数据源，可能有局部错误，也可能和当前动力学状态冲突。控制器如果一味追踪，就容易失稳。RGMT 让策略根据当前身体状态动态判断参考片段的重要性，从而降低不一致参考对控制的伤害。

这和 OmniTrack 是一组互补思路：

•OmniTrack 先把参考变物理一致；

•RGMT 让控制器在执行时对参考做动态选择。

我的判断**通用运动控制的下一步不是更强 tracking loss，而是更聪明地处理参考质量。**

### 15BeyondMimic：从动作跟踪走向生成式动作修正

🔗 **项目链接**：https://arxiv.org/abs/2508.08241

📄 **论文标题**：BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion

🏫 **机构**：伯克利；斯坦福大学

BeyondMimic 的完整题目是 From Motion Tracking to Versatile Humanoid Control via Guided Diffusion。它放在 OmniTrack、RGMT 旁边很合适，因为它同样在处理“参考动作如何变成可执行控制”这个问题，但切入点更偏生成式。

![BeyondMimic 论文图](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEW5ku9nBlmsHe7t6tBkHXZB3hIW2qIl9MgvrAZaWN8T7MCeSQDicVOriaBbDCGjlDMEBZvvxvuD92U13FNafvrSVXkDv40frtXas/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=14)BeyondMimic 论文图

普通 motion tracking 通常假设参考轨迹已经给定，控制器只需要尽量追踪。但真实任务里，参考动作可能缺局部细节，可能和当前环境不匹配，也可能需要根据任务目标在线调整。

BeyondMimic 的思路是引入 guided diffusion，让策略不只是被动追踪参考，而是能在约束和目标引导下生成、修正更合适的动作。

这篇论文的价值在于，它把“模仿”往“可控生成”推进了一步。DeepMimic 那条线解决的是“如何跟着人类动作学”；OmniTrack 和 RGMT 解决的是“参考动作怎样更物理、更鲁棒”；BeyondMimic 则进一步问：如果参考本身不够，能不能让模型在物理约束下补出更可执行的行为？

我会把它看成运动控制里的一个过渡信号：**未来的人形控制器不会只接收固定 motion clip，而会越来越多地接收目标、约束、环境状态和动作先验，然后在身体动力学允许的范围内生成动作。**

### 16OmniXtreme：高动态动作会撞上硬件边界

🔗 **项目链接**：https://extreme-humanoid.github.io/

📄 **论文标题**：OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control

🏫 **机构**：BIGAI；BIGAI & 宇树科技；上海交通大学；中科大；宇树科技；华中科技大学；北京理工大学

OmniXtreme 的关键词是 generality barrier in high-dynamic humanoid control。它指出，当动作库越来越多、动作越来越极端时，通用性和跟踪精度之间会出现冲突。一个策略想覆盖更多动作，可能会损失高动态技能的执行质量。

![OmniXtreme 论文图](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEWfu6oSdHDWDJsPVpSMBxY3TWt96vWhqXficib8YuvJhrHKjvTJpzgIQhNPDRlEgicfQhJJTbtGaOut3b5n6xsOz99elM0SjwYNhY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=15)OmniXtreme 论文图

论文通过高容量架构和 actuation-aware refinement，把通用技能学习和具体技能精修解耦。它还非常现实地讨论了真机失败案例：一些失败出现在 impulsive landing phase，可能触发 motor overcurrent、power limits、battery undervoltage 等硬件保护。

这一点很重要。

很多论文在仿真里讲高动态动作时，容易忽略真实硬件边界。真实人形机器人不是无限力矩、无限散热、无限抗冲击的系统。越高动态，越容易碰到电机、电池、结构强度和控制频率限制。

我的判断**高动态控制的下一阶段不会只比谁动作更猛，而会比谁更懂硬件。**

### 17SONIC：把 motion tracker 做成可扩展基础模型

🔗 **项目链接**：https://nvlabs.github.io/SONIC/

📄 **论文标题**：SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control

🏫 **机构**：NVIDIA

SONIC 的题目是 Supersizing Motion Tracking for Natural Humanoid Whole-Body Control。它把 humanoid whole-body motion tracker 当成基础模型来扩展，研究参数规模、数据规模、训练计算对控制能力的影响。

![SONIC 论文图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWLwoHR59Iiau2ywJsKibnKia5jDVDI6DquLyM4SKvDWI5LsGYV5oPlFzHv6NBDBL58YTAITYaomKEeuCVHITDOxbSqUJDYulbPgk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=16)SONIC 论文图

这篇论文和传统 motion tracking 工作的区别在于，它不只问“某个策略能不能跟某些动作”，而是问：

如果我们把控制器规模做大，会不会出现更通用的身体能力？

它还讨论下游任务、交互式运动控制，以及 motion 和 VLA 表示的迁移价值。这意味着 motion tracker 不再只是一个执行模块，而可能成为上层任务模型的底层表征。

SONIC 的方向很重要，但也要谨慎：运动控制的 scaling 不会和语言模型完全一样，因为机器人还受硬件、物理和实时闭环约束。参数变大不一定能直接解决接触和安全问题。

但它打开了一个问题：**人形机器人是否会出现自己的 control foundation model？**

### 18AMS：敏捷和稳定不是天然兼容的

🔗 **项目链接**：https://opendrivelab.com/AMS/

📄 **论文标题**：Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data

🏫 **机构**：香港大学；NVIDIA；清华大学

AMS，全称 Agility Meets Stability，明确讨论一个矛盾：高动态敏捷动作和稳定恢复能力很难在同一个控制器里兼得。

![AMS 真实机器人动作示例](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWWcKez7soHjhhZSGIWSEoCaNwgwC1N0icxgr50fveoUOmbrez9keX5UKxWXBoKlxIP4eURzcBic29FsiaBviarhRLb42cpouicbGWI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=17)AMS 真实机器人动作示例

论文利用异构数据：人类动作提供敏捷技能，合成 balance motions 提供稳定性和恢复能力。它希望一个策略既能跟踪动态动作，又能在扰动或失衡时保持稳定。

这篇论文让我想到一个很现实的问题：很多机器人 demo 很敏捷，但边界状态下恢复能力弱；另一类控制器很稳，但动作保守。AMS 试图把这两种能力放在同一个框架里。

我的判断**真正可部署的人形机器人不能只做敏捷动作，也不能只做保守稳定，它必须在二者之间动态切换。**

### 19BFM-Zero：身体也需要一种可提示的“行为词表”

🔗 **项目链接**：https://lecar-lab.github.io/BFM-Zero/

📄 **论文标题**：BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning

🏫 **机构**：CMU；Meta

BFM-Zero 是这批论文里概念最值得单独拎出来的一篇。它的目标是训练一个 promptable behavioral foundation model for humanoid control。

它不是为每个任务单独训练策略，而是通过无监督强化学习学一个统一 latent space，让运动追踪、目标姿态、奖励优化等任务都能映射到同一个行为空间。

它的关键思想是 Forward-Backward representation。给定某类 reward 或 prompt，系统可以在 latent space 里找到对应的 z，策略再以 z 为条件执行行为。

我把它理解成一种“身体词表”的雏形。

语言模型有 token，视觉模型有 patch，人形机器人可能也需要一种行为 token 或 latent action。上层系统不一定直接输出 29 个关节目标，而是输出“向前走、低重心、右手支撑、保持柔顺、恢复站立”这类身体意图的 latent prompt。

![BFM-Zero 方法结构](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEUsdibv0a52ohI3HKynh6BcA1CcyNCMibmOe8yBbYUQWdSh6akJ9iarouxXMMBiaVvyX2k5icSfxpKXPYHTsl12zLS4ib0Mf0FFzhdMs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=18)BFM-Zero 方法结构

当然，BFM-Zero 还不能等同于完整的人形机器人通用控制。它主要展示底层运动、姿态、奖励优化和恢复相关能力，离复杂操作、灵巧手、长期任务还有距离。

但它提出的方向很重要：**底层控制器不只是执行器，而可以变成可提示、可组合的行为模型。**

### 20PvP：训练时的 privileged state 如何变成部署时的本体能力

🔗 **项目链接**：https://github.com/myismyname/SRL4Humanoid

📄 **论文标题**：PvP: Data-Efficient Humanoid Robot Learning with Proprioceptive-Privileged Contrastive Representations

🏫 **机构**：香港理工大学；逐际动力；宁波东方理工大学；中科大；ZJU-UIUC；ZGCA

PvP 的全称是 Proprioceptive-Privileged Contrastive Representations。它关注的是 whole-body control 中的样本效率和部分可观测问题。

![PvP 论文图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEVWAKjWhKB6xKLCjosOkBKbbwFKNMVCRTUkPHdvibL3UrszoDKkbNJxFlpprkzH16pcQ83yv9xclBxdeHayIaLNFxGh9d0fssJM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=19)PvP 论文图

真实机器人部署时能用的通常是 proprioception，例如关节角、关节速度、IMU、历史动作等；但仿真训练时可以获得更完整的 privileged states，例如身体速度、接触状态、地形信息等。PvP 试图利用二者之间的互补关系，通过对比学习得到紧凑、任务相关的 latent representation。

这篇论文不一定像跑酷、开门那样有很强的视频冲击力，但它解决的是很多控制器的基础问题：**怎么把训练时看得到、部署时看不到的信息，变成部署时仍然有用的表征。**

我的判断**这种 proprioceptive-privileged 表示学习会越来越常见，因为它正好连接了仿真训练和真实部署之间的信息落差。**

### 21Adaptive Humanoid Control：多行为蒸馏不是简单拼策略

🔗 **项目链接**：https://ahc-humanoid.github.io

📄 **论文标题**：Towards Adaptive Humanoid Control via Multi-Behavior Distillation and Reinforced Fine-Tuning

🏫 **机构**：哈尔滨工程大学；中国电信 TeleAI；中科大；上海科技大学；哈尔滨工业大学；西北工业大学深圳研究院

这篇论文提出 Adaptive Humanoid Control，通过 multi-behavior distillation 和 reinforced fine-tuning 训练统一控制器。它先训练多个基础行为策略，再蒸馏成一个 multi-behavior controller，最后用强化微调提升地形适应和跌倒恢复。

![Adaptive Humanoid Control 论文图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXy2IXg6lAzGrJE6M4Ria5HbVCAgUrgbLveFPCPXzR3B1gNzrNHUsJ0aLibtpapic1f0br5cI8dymA2FeGltH53kcnsBIoIbibyhpc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=20)Adaptive Humanoid Control 论文图

它要解决的问题很实际：如果每个动作都训练一个独立策略，部署时就会面临策略切换、状态不连续、技能组合困难等问题。一个统一策略更适合真实机器人，但多行为之间会互相干扰。

论文里使用 MoE、gradient projection、behavior-specific critics 等机制，目的都是降低多行为训练中的冲突。

我把它放在“身体基础模型”的早期形态里看。它还不是 BFM-Zero 那种 promptable latent model，也不是 SONIC 那种 scaling tracker，但它已经在处理一个共同问题：

机器人不能靠一堆孤立策略生活，它需要统一、可切换、可恢复的身体控制系统。

## 四、第三组论文：感知式高动态运动

### 22PHP：跑酷的难点不是单技能，而是长程组合

🔗 **项目链接**：https://php-parkour.github.io/

📄 **论文标题**：Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching

🏫 **机构**：Amazon FAR；伯克利；CMU；斯坦福大学

PHP 的完整题目是 Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching。它想让 Unitree G1 这类人形机器人在障碍课程中自主执行长程跑酷，而不是只展示一个单独动作。

它的管线很清楚：

1用 OmniRetarget 把人类跑酷动作转到 Unitree G1；

2用 motion matching 在特征空间里做近邻检索，组合长程参考轨迹；

3为不同技能训练 motion-tracking RL expert；

4把多个 expert 蒸馏成一个 depth-based student policy；

5学生训练中结合 DAgger 和 RL，让策略既模仿专家，又真正优化越障成功。

这篇论文最值得看的是 motion matching。它不是让模型凭空生成下一段动作，而是在动作库里根据当前状态和未来意图找最合适的帧，从而保持技能衔接自然。

![PHP 方法框架](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXvTDmAF4HVdjEh9l8j10DiaEqbjmNY1NcCH3TqE8ZbLAWJqpxbaq3LjzjaxUfAwf8CEU2tma9cNyL05qtDb8wicDy6G6dm7ujNs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=21)PHP 方法框架

实验里，PHP 展示了高墙攀爬、快速翻越、连续障碍通行等能力。论文特别强调高墙攀爬高度可达机器人身高的很大比例，并且多障碍连续任务能在真实环境中运行。

我的判断**跑酷不只是高动态动作，而是动作检索、技能衔接、视觉闭环和结果强化的综合问题。**

### 23Deep Whole-body Parkour：全身动作必须理解环境几何

🔗 **项目链接**：https://project-instinct.github.io/deep-whole-body-parkour

📄 **论文标题**：Deep Whole-body Parkour

🏫 **机构**：清华大学交叉信息研究院；上海期智研究院

Deep Whole-body Parkour 和 PHP 都做跑酷，但侧重点不同。

PHP 更强调长程技能组合，Deep Whole-body Parkour 更强调把 exteroceptive depth perception 接入 whole-body motion tracking，让机器人根据障碍物几何执行多接触全身动作。

![Deep Whole-body Parkour 概览](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEUE0Wbgib6YsQS4vEgufOC4l1TqLUmibhA4w5BaNP9DhKJyN8USho3ib3D3T3PibcicHEbSL6t9hDMFK12QXUZ4EmvIOv4gZSuMGXuY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=22)Deep Whole-body Parkour 概览

这篇论文的核心问题是：传统 perceptive locomotion 通常只处理脚部落点，而 general motion tracking 可以复现复杂动作却不看环境。Deep Whole-body Parkour 想把两者合起来，让机器人既能跟踪复杂动作，又能根据障碍物几何调整身体。

它的数据构建也很有意思：动作和场景不是分开的。跑酷动作依赖障碍物高度、距离和接触面，所以论文采集了人类跑酷动作，同时扫描真实障碍物几何，再放入仿真训练。

论文里最有价值的实验，不是简单证明“机器人能做某个动作”，而是验证视觉能否让机器人从不同初始位置自动收敛到正确接触位置。也就是说，深度图不是装饰，而是会改变动作执行时机和空间位置。

我的判断**复杂全身动作的本质不是姿态，而是身体和环境之间的接触关系。**

### 24Hiking in the Wild：脚落在哪里，比走得多快更重要

🔗 **项目链接**：https://project-instinct.github.io/hiking-in-the-wild

📄 **论文标题**：Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids

🏫 **机构**：清华大学交叉信息研究院；上海期智研究院；清华大学计算机系

Hiking in the Wild 关注复杂野外地形中的感知式徒步 / 跑酷。它和 PHP、Deep Whole-body Parkour 的共同点是使用深度感知，但它更强调持续通过复杂地形，例如楼梯、沟壑、高台、斜坡、边缘密集区域。

![Hiking in the Wild 论文图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWVUibq93F1rtrFOVUibTeycYiaiboicgbElicxbTGy8umKqNoKjLyfxTWicZAh43DNh9sMpMB3iaicTxXQ3TK7PdCFXL7KPq3O1W0sJsvY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=23)Hiking in the Wild 论文图

论文使用单阶段强化学习方案，把原始深度输入直接映射到关节动作。为了保证安全和训练稳定，它设计了 foothold safety mechanism，结合 Terrain Edge Detection 和 Foot Volume Points，避免机器人踩在危险边缘上。Flat Patch Sampling 则用于缓解 reward hacking，生成更合理的训练目标。

这篇论文给我的启发是：对于野外行走，脚落点安全可能比速度更重要。很多失败不是因为机器人不会走，而是脚踩到了边缘、沟壑或不稳定接触区域。

从系统角度看，它在把“地形理解”从高层规划下沉到运动控制里。机器人不是先规划一条路再盲走，而是在每一步里让深度图参与落脚决策。

我的判断**感知式 locomotion 的核心不是看见地形，而是把地形风险变成控制策略里的落脚偏好。**

### 25ASAP：sim-to-real 对齐不是调参数那么简单

🔗 **项目链接**：https://agile.human2humanoid.com

📄 **论文标题**：ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills

🏫 **机构**：CMU；NVIDIA

ASAP 的完整思想是 Aligning Simulation and Real Physics。它关注敏捷全身动作在仿真和真实之间的动力学偏差。

![ASAP 动力学对齐流程](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEV0QgVEBmDicc8ibk1xQtqibprerDcR986MUDn1a9al8xY3cHUJv0hDzw2XicLMyN1xe41HHVzEGibkQTDbX1NzzRYzpAOILbPhxQQU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=24)ASAP 动力学对齐流程

它的流程可以理解为三步：

1在仿真中用人类动作数据预训练 motion tracking policies；

2把策略部署到真实机器人上收集 rollout 数据；

3基于真实数据训练 delta action/model，修正仿真状态和真实状态之间的偏差。

ASAP 有一点非常现实：论文直接提到真实机器人上采集敏捷动作数据会受到硬件限制，比如电机过热、硬件损伤、数据规模受限等。这不像很多 sim-to-real 论文只强调算法优雅，它承认真机高动态动作本身就是昂贵且危险的。

我把 ASAP 放在这条线里，是因为它说明高动态动作不仅是控制问题，也是系统辨识和硬件管理问题。

我的判断**越敏捷的动作，越不能只靠 domain randomization 粗暴覆盖；真实轨迹反馈和动力学对齐会越来越重要。**

### 26视觉驱动足球技能：足球任务里，视觉和动作是同一件事

🔗 **项目链接**：https://humanoid-kick.github.io

📄 **论文标题**：Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots

🏫 **机构**：清华大学；字节跳动 Seed；中国农业大学

这篇论文要让人形机器人学习视觉驱动的反应式足球技能。它不是做一个传统规则系统，而是把视觉感知、运动先验和动态控制结合起来，让机器人在真实 RoboCup 类场景中完成更连贯的踢球行为。

![视觉驱动足球技能系统概览](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEUtbVmacs1B7LxXFTaDjmkmMaFLuw0VYFqUhiaK9UGPEHAESRsUZbYaiacf5uiaDgibrH54mScCmasfjpregHeriaSwfXYKwcJQB5ys/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=25)视觉驱动足球技能系统概览

论文提出 virtual perception system，模拟真实视觉误差，并用 encoder-decoder 从不完美观测中恢复 ball position 等 privileged-like state。这是为了解决真实视觉检测和控制之间的错位。

最有意思的是主动感知。机器人不是被动接收球的位置，而会调整躯干、头部和身体，让球保持在更好的视野里。也就是说，“看球”本身成为动作策略的一部分。

这和开门、跑酷、移动操作是同一个趋势：

视觉不只是输入，视觉会改变身体动作。

### 27Motion Generation + Tracking：复杂地形里，参考动作要在线生成

🔗 **项目链接**：https://wholebodylocomotion.github.io/

📄 **论文标题**：Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking

🏫 **机构**：苏黎世联邦理工机器人系统实验室；西蒙弗雷泽大学；ETH AI Center

这篇论文把 diffusion-based motion generation 和 RL-based motion tracking 结合起来，目标是 terrain-aware whole-body locomotion。

![运动生成与运动跟踪示例](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEXhaeRng3oUr6RbXTHliaAKZnn9XyhLVY2vDrWf4ibPzYnqt1bBTxvibQ6WhMVbaIDKbToBTicwuIZ2wldKlIyzd4eiay8GXVCIwgWk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=26)运动生成与运动跟踪示例

传统 motion tracking 通常依赖固定参考动作，但复杂地形需要动作根据地形实时变化。论文先训练 diffusion model 来预测 terrain-aware reference motions，同时训练 whole-body tracker。之后再和冻结的 motion generator 一起微调，提升鲁棒性。

这篇论文的关键是：参考动作不一定要离线固定，也可以在线生成。

这和 DeepMimic 范式相比是一个重要变化。DeepMimic 更像“给定动作，我跟踪”；这篇则更像“根据地形和目标，生成一段适合当前状态的动作，再跟踪”。

我的判断**未来人形机器人控制会越来越多地使用在线 reference generation，而不是只依赖预录动作库。**

## 五、第四组论文：视觉闭环、全身移动操作、任务接口和世界模型

### 28VIRAL：RGB 视觉 sim-to-real 的系统工程

🔗 **项目链接**：https://viral-humanoid.github.io

📄 **论文标题**：VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation

🏫 **机构**：NVIDIA；CMU；伯克利；香港中文大学

VIRAL 做的是 Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation。它要让人形机器人仅凭机载 RGB 摄像头，在仿真中训练后零样本迁移到真实机器人，完成移动抓取和放置任务。

VIRAL 的整体结构是 teacher-student。教师在仿真里拥有 privileged state，学习长时程 loco-manipulation；学生只看 RGB 和本体感知，通过 DAgger 和行为克隆蒸馏教师能力。

![VIRAL 训练流水线](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEUy9LT9oCvLic2LngN3IUyqYu6FBQuwpxkgHQeib3zmJpvm7ezBKNRkGc7unicfpegD3gRIFZTXu7Eia6bBQUxeO0dcQaCV2eK8DJU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=27)VIRAL 训练流水线

这篇论文里几个工程设计很重要：

•Reference State Initialization，避免长时程任务从头随机探索；

•大规模仿真和 tiled rendering，让视觉学生看到足够多的场景；

•视觉随机化，覆盖光照、材质、相机质量、传感器延迟；

•真实手和相机的延迟/误差建模。

论文还展示了训练规模的重要性。低计算规模下 teacher 和 student 都容易失败；扩大仿真规模后，视觉策略才更稳定。

我的判断**VIRAL 的价值不只是一个任务成功，而是把视觉 sim-to-real 变成一套可扩展工程流程。**

### 29DoorMan：开门是一个被低估的全身任务

🔗 **项目链接**：https://arxiv.org/abs/2512.01061

📄 **论文标题**：Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer

🏫 **机构**：NVIDIA；伯克利；CMU；香港中文大学

DoorMan 解决的是纯 RGB pixel-to-action 的人形开门任务。开门听起来简单，但对人形机器人来说非常复杂。

机器人需要接近门、识别把手、抓住把手、旋转或推拉、跟随门板运动，同时保持全身平衡。门把手和门板是 articulated object，接触状态会随着动作变化。

![DoorMan 真实开门任务](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXsETc7Va7q7KoTohBb8pOdZAxwQtNyAk7WRXWAiar4jSTgAibeibWeB3fXdTB51Y3aLhlZjArepQpQ67z9g7ENsYFu6pr87BZibicA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=28)DoorMan 真实开门任务

DoorMan 的方法是 teacher-student-bootstrap。第一阶段用 privileged teacher 学习完整开门流程；第二阶段蒸馏成 RGB student；第三阶段用 GRPO fine-tuning 处理残余部分可观测性差距。

DoorMan 最值得注意的是 staged-reset exploration。长时程开门任务如果从起点随机探索，策略很难经常到达关键中间状态。阶段重置把训练起点放到不同任务阶段，让策略能集中学习抓握、旋转、推拉等后半段难点。

我的判断**开门不是视觉定位任务，而是视觉、接触、铰链约束和全身平衡共同组成的长时程控制任务。**

### 30WholeBodyVLA：行走不是目的，行走是为了操作

🔗 **项目链接**：https://opendrivelab.com/WholeBodyVLA

📄 **论文标题**：WholeBodyVLA: Towards Unified Latent VLA for Whole-Body Loco-Manipulation Control

🏫 **机构**：复旦大学；OpenDriveLab & 香港大学 MMLab；智元机器人；SII

WholeBodyVLA 讨论的是全身 loco-manipulation VLA。它关注的问题是：人形机器人在大空间里完成抓取、搬运、推车等任务时，locomotion 和 manipulation 不能被简单拆开。

它的关键组件包括 Unified Latent Learning 和 LMO。前者把 action-free 视频转成 latent action token，后者则是面向 loco-manipulation 的底层 RL 策略。

![WholeBodyVLA 框架](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXAj8bicb83G6qHnjyUl0jCibyL4BJfXFud7WF2iaiaYGjzxafmZg5awC5P0w2zUyCfUynbAYyWV92bwEyvg1pozVfoK9UTvWTibV8c/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=29)WholeBodyVLA 框架

这篇论文里我最认同的一点是：行走不是为了追踪速度，而是为了到达适合操作的位置，并让身体稳定下来。

如果底层 locomotion 只会速度跟踪，上层 VLA 就要自己学习“如何走到适合抓取的位置”。这会让任务变得很难。WholeBodyVLA 通过面向操作的 locomotion controller，把行走重新定义成操作准备动作。

这也说明，VLA 调用运动控制不是第一步。第一步是底层控制器先把“走到哪里、怎么站稳、如何为手部操作创造姿态”这些问题做细。

我的判断**未来人形机器人 locomotion controller 不能只服务导航，它必须服务操作。**

### 31SENTINEL：端到端语言动作模型也绕不开机器人动力学数据

🔗 **项目链接**：https://arxiv.org/abs/2511.19236

📄 **论文标题**：SENTINEL: A Fully End-to-End Language-Action Model for Humanoid Whole Body Control

🏫 **机构**：北京大学；BeingBeyond

SENTINEL 直接把自然语言和本体感知映射到全身低层动作。它看起来是这批论文里最“端到端”的方向之一。

![SENTINEL 总体框架](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEWyiaELMC0j51oZqPUdVC6YVttlfrRjdDDfwxiaIQPdm6aw9YTYo1o3tA5BWgm6k2M1SG5GVJjhnJWAJyTicDg4JAUr95PgNSxOPY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=30)SENTINEL 总体框架

但读细之后会发现，它并没有绕过运动控制。它先训练一个能跟踪人类动作的全身控制器，然后用这个控制器在仿真里 rollout，得到机器人自己的 state-action trajectories，再用语言标注训练 language-action model。

也就是说，SENTINEL 的监督信号不是“人体应该怎么动”，而是“机器人在动力学里实际能怎么动”。

这点很关键：语言只是入口，动作数据必须先过机器人动力学这一关。

论文用 Transformer + flow matching action head 预测 action chunk，并用 residual action head 做后训练，提升真实部署和扰动下稳定性。它还讨论了长期观测、done prediction、classifier-free guidance、action chunk horizon 等细节。

我的判断**语言动作模型想控制人形机器人，关键不只是语言理解，而是训练数据必须来自机器人可执行动力学。**

### 32MetaWorld：语言语义、技能选择和物理控制要分层

🔗 **项目链接**：https://arxiv.org/abs/2601.17507

📄 **论文标题**：MetaWorld: Skill Transfer and Composition in a Hierarchical World Model for Grounding High-Level Instructions

🏫 **机构**：北京工业大学；复旦大学；清华大学

MetaWorld 是一个 hierarchical world model。它把系统分成三层：semantic layer、skill transfer layer、physical layer。

语义层用 GPT-4o 或 VLM 解析高层指令，输出技能权重；技能迁移层做动态专家选择；物理层用 TD-MPC2 和专家动作引导完成控制。

![MetaWorld 层次结构](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEWE6xMKAjrg5eYH7Aoxxu7UrVhLjIB2xBIu4S3Pts6cOnw3BLW9uo5lsfiaZdpZ57mbFPOYkeftJQC2kEudWOpzWv3knicKqP2ibQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=31)MetaWorld 层次结构

这篇论文没有真实机器人实验，主要在 HumanoidBench 中验证。但它适合放在这里，因为它代表了一种不同于 SENTINEL 的路线。

SENTINEL 更像把语言直接接到低层动作；

MetaWorld 更像把语言、技能和物理控制分层连接。

我的判断**未来不会是纯端到端或纯模块化的二选一，更可能是“可学习的模块化”：每一层都学习，但接口保持清楚。**

### 33Ego-Vision World Model：世界模型要预测身体接触后果

🔗 **项目链接**：https://ego-vcp.github.io/

📄 **论文标题**：Ego-Vision World Model for Humanoid Contact Planning

🏫 **机构**：伯克利；密歇根大学安娜堡分校；香港中文大学

这篇论文把 world model 用于 humanoid contact planning。它不是预测图像未来，而是预测接触相关任务结果，服务 sampling-based MPC。

![Ego-Vision 世界模型训练流程](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEWTrdeHx2RQpibbWvejNpheA5oIkXrhQ0pGZSSxibJ7X5YkApw0uQCAflHo86o8uzvTdyV3rNyMzAMMlwib7egpnBOJT7ebrKDo1s/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=32)Ego-Vision 世界模型训练流程

它解决的问题是：传统 optimization-based contact planning 面对复杂接触时很难扩展，online RL 又样本效率低。论文用 demonstration-free offline dataset 训练 world model，在压缩 latent space 中预测任务结果，并结合 value function 做更密集、更鲁棒的 planning。

我把它放在世界模型主线里，是因为它说明 world model 不一定只是视觉模型，也可以是身体接触模型。

未来人形机器人进入复杂环境后，很多任务都依赖主动接触：扶墙、撑桌、挡物体、钻过低矮空间、利用环境恢复平衡。世界模型如果能预测这些接触后果，就能为高层控制提供重要能力。

我的判断**人形机器人的世界模型不应该只理解“世界长什么样”，还要理解“身体碰上去会发生什么”。**

### 34GR00T N1：VLA 开始寻找可调用的身体接口

🔗 **项目链接**：https://github.com/NVIDIA/Isaac-GR00T

📄 **论文标题**：GR00T N1: An Open Foundation Model for Generalist Humanoid Robots

🏫 **机构**：NVIDIA 等

GR00T N1 是这次新增论文里很关键的一篇。它的目标不是再做一个单项 manipulation policy，而是把视觉、语言和动作放进同一个 humanoid foundation model 里。

![GR00T N1 模型概览](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEUs8MSh4ErHBFbzZRs3VQSfrFEKr1ZwZ2GpT96khhdcQibCwJcdLOdtbOksORp5LtibBMtWZMAE0ic14MGlFVjmxHCJA4C8zeiaXWY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=33)GR00T N1 模型概览

它的架构可以粗略理解成双系统：

•System 2 是 vision-language module，负责理解图像和语言；

•System 1 是 action module，用 diffusion transformer 生成连续动作 chunk；

•两者端到端联合训练，让高层语义和低层动作不是简单拼接，而是在同一个模型里对齐。

这听起来像“VLA 控制机器人”，但论文真正有价值的地方恰恰不是这句口号，而是它把机器人动作接口拆得很具体：状态历史、action chunk、embodiment tag、latent action、真实动作标签、合成轨迹和真机 post-training 都要一起进入系统。

GR00T N1 里很重要的一条线是 Data Pyramid。

最底层是大量人类视频和网络视频，中间是神经生成轨迹和仿真轨迹，最上层才是真机机器人数据。

这样做的原因很现实：真机数据最贵、最贴近身体，但覆盖度不足；人类视频和合成数据覆盖度高，但必须通过 latent action、inverse dynamics 或 post-training 才能落到具体机器人身上。

所以我不会把 GR00T 看成“语言直接控制身体”的论文。它更像是在回答一个更工程化的问题：如果未来有很多人形机器人、很多任务、很多数据来源，VLA 到底要通过什么动作格式和身体接口，去调用已经训练好的精细运动能力？

这也解释了为什么 GR00T 后续版本值得放进这篇文章一起看。官方 Isaac-GR00T 仓库已经把主线推进到 N1.5、N1.6 和 N1.7。

N1.5 强化了 grounding、语言跟随和空间理解；N1.6 把训练数据和任务覆盖扩展到更多 household / factory object manipulation；N1.7 则在更小、更高效的模型上继续做跨身体接口。

README 明确提到，N1.7 使用 Cosmos-Reason2-2B / Qwen3-VL 2B backbone、20K 小时 EgoScale 人类视频、相对末端执行器动作空间，以及来自 AgiBot、Physical Intelligence、Boston Dynamics AI Institute、Fourier、NVIDIA 等来源的机器人数据。

对人形机器人来说，最值得注意的是 N1.7 里的 relative end-effector action space 和 whole-body workflow。

它不是让 VLA 直接输出每个关节的力矩，而是先输出更紧凑的动作表示，再交给逆运动学或 learned whole-body controller 去变成具体身体动作。这个设计和前面很多论文的结论是同一方向：上层智能需要的是可调用的身体 API，而不是直接接管全部底层控制。

我的判断**GR00T 的意义不只是一个开源 VLA 模型，而是把“高层模型未来如何调用人形机器人身体”这个问题做成了一条持续迭代的系统路线。**

### 35DreamDojo：世界模型开始变成机器人策略的试验场

🔗 **项目链接**：https://dreamdojo-world.github.io/

📄 **论文标题**：DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos

🏫 **机构**：NVIDIA；香港科技大学；伯克利；华盛顿大学；斯坦福大学；KAIST；德州大学奥斯汀分校等

DreamDojo 也是这次新增材料里非常值得单独放大的工作。它做的不是 VLA，而是 robot world model：给定机器人当前观察和动作，预测接下来会发生什么。

![DreamDojo 世界模型示例](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEX5Wfml43LxTALT16FjWjwhr75OH2Y15zJgsjDg8Yg8UbzGtaNZEhw8bpgeQia60xIAA6YmRhVe9AomIlrE3gqSR60sf31ibXnEE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=34)DreamDojo 世界模型示例

这里最关键的不是“会生成视频”，而是它试图让世界模型具备机器人策略可用的物理和动作可控性。

论文用大规模第一视角人类视频做预训练，数据规模达到 44K 小时；然后用 continuous latent action 解决人类视频没有机器人动作标签的问题；最后再用少量目标机器人数据 post-train，让模型对具体机器人 embodiment 和 action space 变得可控。

DreamDojo 和 GR00T 刚好构成一对互补问题：

•GR00T 问的是：VLA 如何通过动作接口调用机器人能力；

•DreamDojo 问的是：动作执行前，能不能先在世界模型里预测后果；

•前者更像动作模型，后者更像策略评估和 planning 的试验场。

DreamDojo 的下游实验也很有意思。它不只是拿世界模型做长视频生成，而是拿来做 live teleoperation、policy evaluation 和 model-based planning。

比如在策略评估里，它用预测 rollout 去估计不同策略在真实任务中的成功概率；在 planning 里，它可以在执行前比较多个候选策略或轨迹，把更可能成功的方案选出来。

这对人形机器人很重要。因为未来真正贵的不是生成一个好看的 demo，而是真机反复试错的成本。

一个足够可信的世界模型，可以先在模型里暴露失败、接触错位、物体状态偏差和长时程漂移，再决定什么值得上真机。

当然，DreamDojo 还没有解决所有问题。论文自己也承认，它对少见动作、高速动作和细粒度失败的预测还不完美，而且模型预测的成功率可能比真实世界更乐观。

这说明世界模型不能替代真机实验，它更像一个降低试错成本、筛掉明显坏方案的中间层。

我的判断**世界模型真正进入机器人，不会只作为“未来视频生成器”，而会变成策略评估、任务规划和安全试运行的基础设施。**

## 六、第五组论文：接触、柔顺、负载和失败恢复

### 36CHIP：柔顺不是附加功能，而是任务接口

🔗 **项目链接**：https://nvlabs.github.io/CHIP/

📄 **论文标题**：CHIP: Adaptive Compliance for Humanoid Control through Hindsight Perturbation

🏫 **机构**：NVIDIA；斯坦福大学；德州大学奥斯汀分校

CHIP 的题目是 Adaptive Compliance for Humanoid Control through Hindsight Perturbation。它想让已有 motion tracking controller 获得可调 end-effector compliance。

![CHIP 任务展示](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEWKfjjogCuAGZCp6kAbfL6HfjZiaLxtcBTr2bI23Rscnqfcq77w34EOMm5vAphnGYnutnYsGFwIh2teCYVcrxwVWcWdPoyyP7Zk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=35)CHIP 任务展示

传统刚性 motion tracking 在高动态动作里很好用，但在擦白板、推车、开门、协作搬运等任务里会出问题。因为这些任务要求末端执行器在受力时产生合理偏移，而不是强行保持目标位置。

CHIP 的核心是 hindsight perturbation：训练时向末端施加扰动力，但在观测目标里事后扣除扰动偏移，让策略把受力后的偏移当成合理状态，而不是立刻强行纠正。

它还把柔顺系数变成可调输入。比如开门阶段可能需要刚一点，擦白板需要软一点，搬重物需要根据重量调整刚度。

我的判断**柔顺控制未来会成为精细操作和遥操作采集的重要底层接口。没有柔顺，很多接触任务的数据都采不好；数据采不好，后续 VLA 也很难学到稳定操作。**

### 37GentleHumanoid：安全接触必须进入训练目标

🔗 **项目链接**：https://gentle-humanoid.axell.top

📄 **论文标题**：GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction

🏫 **机构**：斯坦福大学

GentleHumanoid 关注 upper-body compliance for contact-rich human and object interaction。它的目标是让人形机器人在握手、拥抱、辅助坐站、气球操作等任务里保持安全、自然的接触。

![GentleHumanoid 任务展示](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEUHvRt5BUecYUnRQbNOlQDHqRDpYr1XyosHK1VX2lAQIiblYCicPuYcRUibib2C6nuRl0C5f8DqT3yv3Bo34k9o8N02PBJZo2FC1Gg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=36)GentleHumanoid 任务展示

它的核心设计是 reference dynamics：把上肢关节建模为 spring-damper 系统，让参考轨迹随着接触力变化。策略不再跟踪固定目标，而是跟踪会根据接触变形的 reference。

这和外部套一个 impedance controller 不同。GentleHumanoid 把顺应性放进 RL 的奖励和参考状态里，让策略在训练阶段就学会接触。

论文还加入安全力阈值机制，避免接触力无限增长。实验覆盖外力扰动、拥抱人体模特、坐站辅助和气球操作。

我的判断**人形机器人和人接触时，位置控制不是核心，力边界才是核心。**

### 38HAIC：对象也有自己的动力学

🔗 **项目链接**：https://haic-humanoid.github.io/

📄 **论文标题**：HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model

🏫 **机构**：清华大学；香港科技大学（广州）；苏黎世联邦理工；小米机器人实验室

HAIC 关注 Humanoid Agile Object Interaction Control via Dynamics-Aware World Model。它处理的是 underactuated objects，也就是对象本身有独立动力学和非完整约束，不是被机器人末端完全控制的刚性物体。

![HAIC 论文图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEVYcBuV2Ej72NbI9cPfbz2dDpgT75HiaUgNda1Sfz7xHFUpeFeSZicD6JobTwtTLqhW1Rf80icRyV6uPW91IrVBibn1oVk2Bibg4fks/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=37)HAIC 论文图

这类任务很难。比如推车、搬运、球类、箱体互动、受遮挡物体，机器人不能只根据当前几何位置规划手臂动作。对象会受到力、摩擦、惯性、接触约束影响，反过来改变机器人状态。

HAIC 的核心是 dynamics-aware world model，用来预测对象和机器人之间的交互后果，从而支持控制。

我把它放在 CHIP 和 GentleHumanoid 之后，是因为它把“接触”从机器人末端扩展到对象动力学。柔顺控制解决的是机器人怎么接触，HAIC 进一步问：接触之后对象会怎么动。

我的判断**人形机器人做操作任务时，控制对象动力学会比控制手的位置更重要。**

### 39HALO：带负载的人形机器人不是同一个系统

🔗 **项目链接**：https://mwondering.github.io/halo-humanoid/

📄 **论文标题**：Closing Sim-to-Real Gap for Heavy-loaded Humanoid Agile Motion Skills via Differentiable Simulation

🏫 **机构**：浙江大学；中国电信 TeleAI；上海交通大学；Lumos Robotics

HALO 解决 heavy-loaded humanoid agile motion skills。真实机器人执行任务时经常会携带未知负载，而负载会改变机器人动力学。一个空载时表现很好的策略，拿重物后可能直接失效。

![HALO 论文图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEW4xVzEs04kLibaiaicVEIibnRRHDwgrXtnBqVibh36yrUU9AQGlodAQYsJjDDtvOtqPh4aHxJP0ib6ouNPtolaMxZIAd5Lr0DiasT0LI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=38)HALO 论文图

HALO 用 differentiable simulation 做两阶段系统辨识。第一阶段校准名义机器人模型，减少固有 sim-to-real 差异；第二阶段识别 payload mass distribution，处理带负载后的动力学变化。

论文的实验包括真实机器人上的高动态动作和负载情况。它强调，通过显式减少结构化模型误差，可以让 RL 策略在重载条件下更稳定地零样本迁移。

这篇论文的价值在于，它把“负载”从任务条件变成控制系统的一部分。

我的判断**人形机器人一旦进入搬运和工具使用场景，空载控制器就不够了，负载辨识会成为基础能力。**

### 40Heracles：偏离参考轨迹时，继续追踪可能是错的

🔗 **项目链接**：https://heracles-humanoid-control.github.io/

📄 **论文标题**：Heracles: Bridging Precise Tracking and Generative Synthesis for General Humanoid Control

🏫 **机构**：X-Humanoid Heracles 项目组 / 北京人形机器人创新中心

Heracles 试图连接 precise tracking 和 generative synthesis。它的核心问题是：传统 tracking controller 在正常状态下很好，但当机器人受到强扰动、摔倒或远离参考状态时，继续追踪原始参考动作可能会更糟。

![Heracles 恢复动作概览](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEV2xfMJcHkBt8VDZgXxGC5j3jKIpBUgrx0YDPSPYEyIzEhDajogicabMjUohrCib2a3R9scgs61wlgr3e83tyt6PqRiabCHWGRG1k/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=39)Heracles 恢复动作概览

Heracles 在参考动作和低层物理跟踪器之间插入一个生成式中间件。正常时，它尽量保持参考动作；异常时，它根据当前状态生成新的未来关键姿态，让低层控制器执行更适合恢复的动作。

这篇论文最重要的思想是“异常状态下要改写参考”。一个人被推倒后，不会继续执行原来的走路动作，而会做出恢复动作。人形机器人也需要这种能力。

我的判断**通用控制器不能只会正常动作，还必须知道什么时候放弃原参考、重新生成恢复动作。**

### 41SafeFall：失败不可避免，但不能灾难化

🔗 **项目链接**：https://safefall.github.io

📄 **论文标题**：SafeFall: Learning Protective Control for Humanoid Robots

🏫 **机构**：山东大学；BIGAI；清华大学

SafeFall 做的是 protective control for humanoid robots。它的出发点非常现实：双足机器人不可避免会摔倒，而摔倒会损伤传感器、执行器和结构件。

![SafeFall 论文图](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEVBoTjjg07J28iboAykIrDLCpR2t10zwsXG0MziatJibrAQFd81FEJ85mKIyVvntClstNWbo2wW1rbiasb7vdgAKyqGlbuauRSSNKk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=40)SafeFall 论文图

SafeFall 不是让机器人永远不摔，而是在检测到跌倒不可避免时，激活保护策略，减少硬件冲击。系统包含一个轻量 GRU-based fall predictor 和一个 damage mitigation policy。正常控制时它保持 dormant，不干扰 nominal controller。

论文在 Unitree G1 上做真实实验，包括不同方向外力推扰、走路时误踩台阶、高速跑步绊倒等场景，并报告最大关节力、接触力等指标改善。

SafeFall 的重要性在于，它改变了评价思路。

过去我们问机器人成功率多高。

未来还要问：失败时会怎么样？

我的判断**人形机器人真实部署时，安全不是没有失败，而是失败不应造成灾难性损伤。**

### 42Thor：强接触环境里，机器人需要全身反应

🔗 **项目链接**：https://baai-aether.github.io/baai-thor/

📄 **论文标题**：Thor: Towards Human-Level Whole-Body Reactions for Intense Contact-Rich Environments

🏫 **机构**：北京理工大学；北京智源人工智能研究院

这篇中文文件名是“北京理工：雷神”，论文题目对应 Human-Level Whole-Body Reactions for Intense Contact-Rich Environments。它关注的是强接触环境中的全身反应。

![Thor 论文图](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEUkQd3gAib21zrLBW2hnE1aJ0CN6VsYZK06Y1OZZXjYMBEky1k6g0lxwAJzToGaazqAwup7rUZYIhqfj7Cpct859hZXvQH6mNjU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=41)Thor 论文图

机器人在服务、工业、救援场景中可能需要持续和环境发生强接触，比如拉、推、支撑、撞击、搬运。传统方法如果只关注末端或下肢，很难产生类似人的全身协调反应。

论文设计 force-adaptive torso-related reward，并提出 Thor RL architecture，把上肢和下肢解耦但共享全身信息。它希望机器人在强交互任务中通过躯干、上肢、下肢协同产生更大、更稳定的作用力。

我把它放在 GentleHumanoid 和 CHIP 之后看：GentleHumanoid 强调安全柔顺，CHIP 强调可调末端刚度，Thor 强调强接触下的全身发力。

我的判断**接触控制不是只有“软”，还包括什么时候要软、什么时候要硬、什么时候要用全身发力。**

## 七、整体判断：人形机器人正在形成一套“身体系统栈”

如果把这些论文放成一张系统图，我会这样理解：

•数据层：人类动作、视频、遥操作、交互如何变成机器人可执行参考。代表工作包括 GMR、NMR、OmniRetarget、H2O、HumanX、HDMI、GenMimic。

•参考 / 跟踪控制层：参考动作如何进入物理仿真、稳定跟踪并在线修正。代表工作包括 DeepMimic、OmniTrack、BeyondMimic、运动生成与跟踪、Heracles。

•控制层：多动作跟踪、抗扰、恢复、负载适应。代表工作包括 Any2Track、RGMT、OmniXtreme、SONIC、AMS、HALO。

•感知层：视觉和深度如何进入动作闭环。代表工作包括 PHP、Deep Whole-body Parkour、Hiking、足球、DoorMan、VIRAL。

•接触层：力、柔顺、对象动力学如何进入控制。代表工作包括 CHIP、GentleHumanoid、HAIC、Thor。

•安全层：跌倒、异常状态、失败恢复。代表工作包括 SafeFall、Heracles、AMS。

•任务接口层：精细运动能力成熟后，语言、VLA 如何调用身体能力。代表工作包括 SENTINEL、WholeBodyVLA、OmniH2O、GR00T、MetaWorld、BFM-Zero。

•世界模型层：动作执行前如何预测后果、评估策略。代表工作包括 Ego-Vision World Model、DreamDojo。

我认为这张表比“哪篇论文更强”更重要。

因为这些论文不是在同一个榜单上竞争。它们更像是在一起补齐人形机器人的身体系统栈。

GMR/NMR/OmniRetarget/H2O/OmniH2O/HumanX/HDMI/GenMimic 解决动作数据和遥操作接口进入机器人身体的问题；

DeepMimic/OmniTrack/BeyondMimic 解决参考动作进入物理控制的问题；

RGMT/Any2Track/OmniXtreme 解决通用跟踪、动作生成和抗扰；

PHP/Deep Whole-body Parkour/VIRAL/DoorMan 解决视觉闭环；

CHIP/GentleHumanoid/Thor 解决接触和柔顺；

SafeFall/Heracles 解决失败恢复；

SENTINEL/WholeBodyVLA/OmniH2O/GR00T/MetaWorld/BFM-Zero 探索上层智能如何调用已经被封装的身体能力；

Ego-Vision World Model/DreamDojo 解决执行前如何预测身体与世界的后果。

所以这批论文共同指向一个判断：

人形机器人不是一个大模型加一个身体，而是一套从数据、参考、控制、感知、精细接触、安全、任务接口到世界模型评估的完整系统。

## 八、我对未来的几个判断

### 判断一：动作库会继续变大，但动作库不是终点

动作库越来越大是必然的。

未来会有 mocap、AMASS、人类视频、生成视频、真机遥操作、仿真 rollout、多机器人数据等来源。

但 **动作库大不代表机器人能用**。

下一阶段真正稀缺的，不只是更多动作，而是 **更精确的交互数据**：机器人看到了什么、身体状态是什么、手脚和物体怎么接触、用了多大力、哪里发生了滑移或失败、失败后如何恢复。

关键是：

•**动作是否能重定向**；

•**是否保留交互关系**；

•**是否物理一致**；

•**是否能被控制器稳定跟踪**；

•**是否能根据视觉和接触在线调整**；

•**是否包含视觉、本体、接触、力和失败恢复信息**；

•**是否能组合成长任务**。

所以未来比拼的不是谁有更多动作，而是谁能把动作变成 **可训练、可复用、可精细交互的身体能力**。

### 判断二：视觉会从“识别目标”变成“控制闭环”

VIRAL、DoorMan、PHP、Deep Whole-body Parkour、Hiking、足球技能都在说明这一点。

视觉不是任务开始前看一眼目标，而是 **在执行过程中不断参与动作**。

✓**机器人会因为视觉调整身体**

✓**身体动作会改变视觉输入**

✓**视觉误差会改变接触结果**

接触结果又会影响下一步视觉和动作。

这会让 **视觉 sim-to-real** 成为人形机器人落地的核心瓶颈。

### 判断三：柔顺和力控会成为精细操作的数据质量问题

如果底层控制器太硬，遥操作采集接触任务会很难。操作者很难稳定擦白板、推车、抱人、搬箱子。采集出来的数据差，后续上层模型再大也学不好。

所以 **柔顺不是控制小细节，而是决定机器人能不能进入精细操作阶段**。没有稳定的接触和力控，就很难采到高质量交互数据；没有高质量交互数据，VLA 调用运动控制也只能停留在概念上。

### 判断四：失败恢复会成为严肃指标

未来人形机器人论文 **不能只展示成功视频**。

我会更关注：

•失败发生在哪里；

•摔倒时最大冲击力多少；

•能否保护头部和传感器；

•能否从异常姿态恢复；

•是否能重新进入任务；

•是否能避免伤人和损坏环境。

SafeFall 和 Heracles 这类工作会越来越重要。

### 判断五：VLA 调用运动控制不是起点，而是结果

SENTINEL、WholeBodyVLA、OmniH2O、GR00T、MetaWorld、BFM-Zero 都说明一件事：

语言和 VLA 最终必须通过身体接口落地。但这个接口不是凭空出现的。

更合理的顺序应该是：

**第一步**，底层运动控制先解决稳定移动、动作跟踪、抗扰和失败恢复；

**第二步**，控制器进入精细全身交互，能处理手、脚、腰、头部、视线、接触力、负载和对象动力学；

**第三步**，这些能力被封装成技能、latent action、短时 action chunk 或身体 API，VLA 才能稳定调用。

这个接口不能太底层，否则上层模型学习成本太高；也不能太僵硬，否则复杂任务表达不了。

未来可能需要一种中间接口：

•技能 token；

•latent action；

•接触模式；

•柔顺系数；

•目标姿态；

•embodiment tag；

•relative end-effector action；

•kinematic pose interface；

•短时 action chunk；

•恢复策略；

•视觉闭环目标。

这就是我说的“身体 API”。

### 判断六：世界模型会变成策略上线前的试验场

Ego-Vision World Model 和 DreamDojo 让我更确信一点：未来人形机器人不会只靠“直接上真机试”来验证策略。

如果世界模型只会生成好看的未来视频，价值有限。真正有价值的是 **action-conditioned rollout**：给它当前观察、机器人动作和任务上下文，它能不能预测接触后果、物体状态变化、失败概率和长时程漂移。

这类模型一旦足够可信，就会进入三个位置：

•训练时，帮策略理解动作后果；

•部署前，筛掉明显危险或低成功率的候选动作；

•运行中，为 model-based planning 提供短时预测。

所以世界模型不是替代运动控制，也不是替代真机实验。它更像任务执行前的一层 **“试运行环境”**：先让策略在模型里摔几次、偏几次、撞几次，再决定什么值得交给真实机器人。

## 九、最后总结

读完这批论文，我最想留下的一句话是：

人形机器人真正难的，不是把动作做出来，而是让动作进入真实世界的精细交互闭环。等这件事做稳之后，才谈得上被上层模型稳定调用。

这个闭环里有：

•人类动作和机器人身体之间的翻译；

•参考动作和物理可执行性之间的矛盾；

•视觉和身体动作之间的双向影响；

•接触力和柔顺控制；

•更精确的全身交互数据采集；

•负载变化和硬件边界；

•跌倒、失败和恢复；

•成熟之后，语言和 VLA 对身体能力的调用；

•世界模型对策略后果的预测和评估。

如果只看单个 demo，我们容易觉得人形机器人已经很接近“通用”。

但把这 42 篇独立工作放在一起看，会发现真正的通用还差很远。

它不是一个大模型就能解决，也不是一个强控制器就能解决。

它需要一整套 **身体系统**：

•**数据要能来**；

•**动作要能转**；

•**参考要能改**；

•**控制要能稳**；

•**视觉要能闭环**；

•**接触要能精细**；

•**力要能控制**；

•**失败要能恢复**；

•**上层智能要能调用**；

•**世界模型要能预演**。

所以我对人形机器人的新判断是：

未来最重要的竞争，不是单点动作能力，也不是单个大模型能力，而是谁能更快把运动控制推进到精细交互阶段，并把这种能力系统化、接口化、可评估化。

谁能更快把动作数据、物理控制、视觉感知、精细接触、力控、安全恢复和高质量数据采集连成一个可靠闭环，谁就更接近真正可用的人形机器人。VLA 和世界模型会很重要，但它们更像是站在这套身体能力之上的下一层。
