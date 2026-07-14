---
title: RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源
author: Roboparty萝博派对
date: "2026-07-14 13:01:27"
source: "https://mp.weixin.qq.com/s/DL-ypgpyLVnypxMwA5d5pw"
---

# RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源

![Image](https://mmbiz.qpic.cn/mmbiz_gif/a7nfRicgiaaBTh4yEo0XVkpiaF4H4yzZxze3Dn9MNe0E8B4tu9G5MkGWoHRuS4TCDCHOibJ8Nhbl8iamQbnA8iaWnwCTNYPG1LtAIpvJ1Q1E675JQ/640?wx_fmt=gif&from=appmsg#imgIndex=0)![Image](https://mmbiz.qpic.cn/mmbiz_png/a7nfRicgiaaBRyNP7x3iaI9FiaIqP6sTuVcUPuOk40M3X6F41P8Dcf0BFvVC2xQ9Gtiaj2icyyKnbN9BMSuqz7gAouExZEmWibluh9BUmC7lVdVWEw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

RoboParty Lab 官网：https://lab.roboparty.com

**RoboParty Lab 正式成立：**

**让每一个有能力、有技术、有想法的人，被这个世界看到**

在这里，**idea + 高自由度研究环境 + 开放平台 + 3～6 个月**，就可能让一个**有能力、有技术、有想法的人，被这个世界快速看到**。

但在人形机器人研发中，一个好 idea 往往并不是输在想法本身，而是被大量底层问题消耗：本体是否可靠、数据从哪里来、动作如何处理、训练框架是否稳定、策略如何部署到真机、实验结果如何复现、成果如何形成论文或开源影响力。很多年轻工程师和研究者，本来应该把时间用在真正前沿的问题上，却不得不反复从零搭建基础设施。

这也是 RoboParty Lab 想要解决的问题：把人形机器人研发中最耗时、最分散、最难复现的底层能力，提前沉淀成一套可以**被复用、被验证、被继续扩展的研发底座**。

这套底座就是**Party OS**。它连接机器人本体、数据、训练框架、动作工具链、Sim2Real、真实机器人验证、开源发布与技术影响力，让开发者不必反复消耗在重复建设里，而是能更快把一个真正好的 idea，变成可验证、可开源、可发表、可被世界看见的成果。

你只需要带着真正好的问题、想法和能力，进入这个系统，把时间花在最纯粹、最关键的事情上。

不论是做有意义的 research，还是做 solid 的开源产出，RoboParty Lab 都希望让真正有价值的技术工作获得应有的回报。

在这里，一群纯粹的人，基于 Party OS，做一些足够前沿、足够真实、也足够有意义的事情。

Party OS Github地址：

 https://github.com/Roboparty/Party\_OS

![Party OS Roadmap](https://mmbiz.qpic.cn/mmbiz_png/a7nfRicgiaaBQia7DG9ZmErevDYO6JWYWcE7pfibtF8MvicHgwkL0YwJKBZXPmW6v05fDKeX3wRy6ib50ia9GGzHPX4g1MlZVKWicaPU6GGJokQLkibk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

**开源监督学习框架 MimicLite****：**

让通用人形运动跟踪进入小时级迭代

**8 卡、3 小时，全开源。低成本打通高动态运动跟踪和 SONIC 级低延迟遥操。**

Github地址：https://github.com/Roboparty/MimicLite

MimicLite 是一套面向人形机器人通用运动跟踪的开源训练与部署基础设施。它贯通数据、训练、统一评测与真机部署，使研究者能够以更低算力快速迭代跟踪策略，并将来自不同训练框架的策略接入同一套 sim2real 系统。

**小时级训练，保持强劲跟踪性能**

MimicLite 是一套面向人形机器人通用运动跟踪的开源训练与部署基础设施。它使用 8 张 RTX 4090 GPU，在约 3 小时内训练出具有强劲跟踪性能的通用策略。训练成本约为 24 GPU-hours，仅相当于 SONIC 算力的约 1/875，同时实现更好的全局根部跟踪和相当的局部身体跟踪精度。

MimicLite 还可以随并行环境数量、GPU 数量和模型容量继续扩展。实验表明，更大的训练规模能够进一步提高高动态动作的完成度和整体跟踪质量，使新数据、新任务和新控制设定的验证不再依赖漫长的训练周期。

**面向持续迭代的 Tracking Infra**

MimicLite 通过统一的 motion、robot asset 和 policy artifact 接口，减少训练、评测与部署之间的系统差异。any4hdmi 将来自 LAFAN、100STYLE、SONIC 和真实数据等不同来源的动作组织为统一格式；mjhub 则保证机器人模型在训练、运动学计算和 sim2sim 验证中的一致性。

这套设计的重点不是增加零散工具，而是建立一条稳定、可复现的数据到部署链路，让研究者能够集中迭代 observation、reward、termination 和数据分布，而无需为每个数据集或训练后端重新搭建系统。

**从低延迟遥操作到高动态真机动作**

同一套 MimicLite policy 可以直接用于低延迟 Pico/XR 遥操作。系统将人体输入实时转换为参考运动，并根据机器人状态输出低层控制目标。

在真机上，单个策略既能完成行走、转身、侧步、下蹲和单膝或双膝跪地等交互动作，也能跟踪虎跳衔接肩滚、旋转踢和侧手翻等高动态动作，实现灵活遥操作与敏捷运动能力的统一。

**跨 Codebase 的统一评测与 Sim2Real**

**MimicLite 采用模块化 observation interface，将 policy-specific 的输入构造与共享部署 runtime 解耦。接入其他 codebase 训练的策略时，只需实现对应的 observation class，并通过 YAML 定义各项 observation 的顺序和参数，无需修改推理、仿真或机器人接口。**

目前系统已经接入 SONIC、HEFT、TeleopIT、Humanoid-GPT、BFM-Zero 和 TWIST2 等策略。适配后的 policy 可以沿同一条路径完成 matched evaluation、sim2sim 验证与真机部署，使 MimicLite 不仅服务自身训练出的策略，也成为跨 codebase policy 的统一评测与部署层。

全开源**RoboParty RP1**发布后可快速部署，目前已开放 codebase：

https://github.com/Roboparty/Party\_OS

![]()已关注Follow  Replay    Share     Like  Close**观看更多**更多


*退出全屏**切换到竖屏全屏**退出全屏*Roboparty萝博派对已关注Share Video，时长00:09

0/0

00:00/00:09 切换到横屏模式 继续播放进度条，百分之0[Play](javascript:;)00:00/00:0900:09[倍速](javascript:;)*全屏* 倍速播放中 [0.5倍](javascript:;)  [0.75倍](javascript:;)  [1.0倍](javascript:;)  [1.5倍](javascript:;)  [2.0倍](javascript:;)  [超清](javascript:;)  [流畅](javascript:;)  Your browser does not support video tags

继续观看

RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源

观看更多转载,RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源Roboparty萝博派对已关注Share点赞WowAdded to Top Stories[Enter comment](javascript:;)  [Video Details](javascript:;)

高动态新月转体

![]()已关注Follow  Replay    Share     Like  Close**观看更多**更多


*退出全屏**切换到竖屏全屏**退出全屏*Roboparty萝博派对已关注Share Video，时长01:32

0/0

00:00/01:32 切换到横屏模式 继续播放进度条，百分之0[Play](javascript:;)00:00/01:3201:32[倍速](javascript:;)*全屏* 倍速播放中 [0.5倍](javascript:;)  [0.75倍](javascript:;)  [1.0倍](javascript:;)  [1.5倍](javascript:;)  [2.0倍](javascript:;)  [超清](javascript:;)  [流畅](javascript:;)  Your browser does not support video tags

继续观看

RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源

观看更多Original,RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源Roboparty萝博派对已关注Share点赞WowAdded to Top Stories[Enter comment](javascript:;)  [Video Details](javascript:;)

单一policy实现高动态和通用全身摇操作

![]()已关注Follow  Replay    Share     Like  Close**观看更多**更多


*退出全屏**切换到竖屏全屏**退出全屏*Roboparty萝博派对已关注Share Video，时长00:56

0/0

00:00/00:56 切换到横屏模式 继续播放进度条，百分之0[Play](javascript:;)00:00/00:5600:56[倍速](javascript:;)*全屏* 倍速播放中 [0.5倍](javascript:;)  [0.75倍](javascript:;)  [1.0倍](javascript:;)  [1.5倍](javascript:;)  [2.0倍](javascript:;)  [超清](javascript:;)  [流畅](javascript:;)  Your browser does not support video tags

继续观看

RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源

观看更多Original,RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源Roboparty萝博派对已关注Share点赞WowAdded to Top Stories[Enter comment](javascript:;)  [Video Details](javascript:;)

RP1 测试

**无监督强化学习控制开发框架 UFO**

**UFO 是一套面向研发、完全开源的人形机器人无监督强化学习控制（Unsupervised RL Control）开发框架，覆盖训练基础设施、数据管线、算法研究和推理部署全流程。**

Github地址：https://github.com/Roboparty/UFO

框架致力于降低无监督强化学习控制的研发门槛，使研究者能够快速复现 SOTA 方法、探索新的行为表示（representation）、适配不同机器人平台，并实现从训练到真实机器人遥操作部署的一体化开发。

**Fast Training Infrastructure**

利用更轻量级**MJLab 作为backend**，兼容单卡、多卡并行训练，在 8 张 RTX 4090 GPU 上仅需不到 12 小时即可完成BFM-Zero算法训练，摆脱了对单张大显存 GPU 的依赖；在 8 张 H200 上 6-8小时即可完成训练，在性能上也持续优于 BFM-Zero。

![Image](https://mmbiz.qpic.cn/mmbiz_png/a7nfRicgiaaBQfsrKbxFqbeeRrATsB2ORkVWYgSUsS5Xf6B3KeYdpI4x3sHKHYqcuRheO4TdXcjXu2BHeddIWm38W2U815sVH1bLr7WqMVWs8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

**General and Extensible Framework**

统一的 codebase 不再受限于特定机器人，可**无缝适配不同机器人形态，大幅降低新平台的迁移成本**。支持来自不同来源的数据混合训练，并提供灵活的数据调度与配比机制。通过合理的数据分布设计，无监督强化学习不仅能够学习稳定的通用运动，还已展现出侧手翻等高动态动作的学习能力。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/a7nfRicgiaaBRUtLtykQp8xPZqYCRujagiagds9SGtgEAic6Y2S09PPTK0iaiaL1bK6Ybd0re0VEEnQ3RUmxK4oNz6AdiaibaV4DSicyVYeDQZxQbZicI/640?wx_fmt=gif&from=appmsg#imgIndex=4)

**New Representation Integration**

除**集成经典** **BFM-Zero（FB Representation）**外，框架**支持多种行为表示（representation）的无监督学习研究**。

我们已探索**TeCH**（**Te**mporal Distance Modeling via **C**ontrastive Representation Learning for **H**umanoid Whole-Body Control）等新型表示，并取得良好的控制效果，为更通用的无监督控制算法提供统一实验平台。

![]()已关注Follow  Replay    Share     Like  Close**观看更多**更多


*退出全屏**切换到竖屏全屏**退出全屏*Roboparty萝博派对已关注Share Video，时长00:58

0/0

00:00/00:58 切换到横屏模式 继续播放进度条，百分之0[Play](javascript:;)00:00/00:5800:58[倍速](javascript:;)*全屏* 倍速播放中 [0.5倍](javascript:;)  [0.75倍](javascript:;)  [1.0倍](javascript:;)  [1.5倍](javascript:;)  [2.0倍](javascript:;)  [超清](javascript:;)  [流畅](javascript:;)  Your browser does not support video tags

继续观看

RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源

观看更多Original,RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源Roboparty萝博派对已关注Share点赞WowAdded to Top Stories[Enter comment](javascript:;)  [Video Details](javascript:;)

UFO-TeCH(disturbance)

![]()已关注Follow  Replay    Share     Like  Close**观看更多**更多


*退出全屏**切换到竖屏全屏**退出全屏*Roboparty萝博派对已关注Share Video，时长00:47

0/0

00:00/00:47 切换到横屏模式 继续播放进度条，百分之0[Play](javascript:;)00:00/00:4700:47[倍速](javascript:;)*全屏* 倍速播放中 [0.5倍](javascript:;)  [0.75倍](javascript:;)  [1.0倍](javascript:;)  [1.5倍](javascript:;)  [2.0倍](javascript:;)  [超清](javascript:;)  [流畅](javascript:;)  Your browser does not support video tags

继续观看

RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源

观看更多Original,RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源Roboparty萝博派对已关注Share点赞WowAdded to Top Stories[Enter comment](javascript:;)  [Video Details](javascript:;)

UFO-TeCH(motion\_tracking)

**Teleoperation in the Real World**

**首次开源无监督强化学习控制的遥操作代码及完整验证方案，支持真实机器人部署**。机器人能够自然完成深蹲、半蹲、跪地、打滚、跌倒恢复以及抗外力扰动等复杂全身动作，为无监督强化学习在真实场景中的应用提供了完整参考实现。

![]()已关注Follow  Replay    Share     Like  Close**观看更多**更多


*退出全屏**切换到竖屏全屏**退出全屏*Roboparty萝博派对已关注Share Video，时长00:54

0/0

00:00/00:54 切换到横屏模式 继续播放进度条，百分之0[Play](javascript:;)00:00/00:5400:54[倍速](javascript:;)*全屏* 倍速播放中 [0.5倍](javascript:;)  [0.75倍](javascript:;)  [1.0倍](javascript:;)  [1.5倍](javascript:;)  [2.0倍](javascript:;)  [超清](javascript:;)  [流畅](javascript:;)  Your browser does not support video tags

继续观看

RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源

观看更多Original,RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源Roboparty萝博派对已关注Share点赞WowAdded to Top Stories[Enter comment](javascript:;)  [Video Details](javascript:;)

Teleop demo

**Human-to-Humanoid tools**

**30 秒完成 Human-to-Humanoid 动作重映射，推动人形机器人研发工具链开源化。**

Github地址：https://github.com/Roboparty/human-humanoid-tools

Human-to-Humanoid tools（hhtools）是一款高效的开源retarget开源工具，保证用户实现全程web前端操作，并具有以下功能：

**Fast Retarget**

hhtools 依托 Newton IK （Warp可并行）与 Interaction-Mesh（MPC solver） 交互网格双后端架构，实现行业领先的轻量化高速动作迁移，单段地形跑酷、舞蹈、物体交互等全身复杂动作仅需 30 秒即可完成完整重定向运算，同时保障动作时序平滑、无关节突变与滑脚失真问题，大幅压缩人形机器人动作调试周期，适配快速迭代的研发场景，同时可以实现批量并行retarget。

![]()已关注Follow  Replay    Share     Like  Close**观看更多**更多


*退出全屏**切换到竖屏全屏**退出全屏*Roboparty萝博派对已关注Share Video，时长02:06

0/0

00:00/02:06 切换到横屏模式 继续播放进度条，百分之0[Play](javascript:;)00:00/02:0602:06[倍速](javascript:;)*全屏* 倍速播放中 [0.5倍](javascript:;)  [0.75倍](javascript:;)  [1.0倍](javascript:;)  [1.5倍](javascript:;)  [2.0倍](javascript:;)  [超清](javascript:;)  [流畅](javascript:;)  Your browser does not support video tags

继续观看

RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源

观看更多Original,RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源Roboparty萝博派对已关注Share点赞WowAdded to Top Stories[Enter comment](javascript:;)  [Video Details](javascript:;)

**Any Motion**

**兼容并可视化市面上绝大部分开源数据集格式**（自动识别），包括但不限于BVH / GLB / SMPL; 数据集包括但不限于bvh Mocap, AMASS, GVHMR, LAFAN1, OMOMO, PHUMA, Intermimic, Meshmimic。

**Any URDF**

**打破单一机型绑定限制，原生支持市面上所有标准 URDF 格式人形机器人模型**，开发者仅拖入机器人 URDF 和 Mesh 文件夹，无需针对不同机器人开发定制适配代码（自动识别）。

![]()已关注Follow  Replay    Share     Like  Close**观看更多**更多


*退出全屏**切换到竖屏全屏**退出全屏*Roboparty萝博派对已关注Share Video，时长01:31

0/0

00:00/01:31 切换到横屏模式 继续播放进度条，百分之0[Play](javascript:;)00:00/01:3101:31[倍速](javascript:;)*全屏* 倍速播放中 [0.5倍](javascript:;)  [0.75倍](javascript:;)  [1.0倍](javascript:;)  [1.5倍](javascript:;)  [2.0倍](javascript:;)  [超清](javascript:;)  [流畅](javascript:;)  Your browser does not support video tags

继续观看

RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源

观看更多Original,RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源Roboparty萝博派对已关注Share点赞WowAdded to Top Stories[Enter comment](javascript:;)  [Video Details](javascript:;)

**Robot → Robot (R2R)**

区别于仅支持人到机器人的传统工具，**hhtools 实现机器人到机器人的动作互转通道**，可将一款人形机器人的成熟动作库直接迁移至另一款结构差异较大的机器人。依托统一骨骼对齐管线，解决数据集找不到的问题。

![]()已关注Follow  Replay    Share     Like  Close**观看更多**更多


*退出全屏**切换到竖屏全屏**退出全屏*Roboparty萝博派对已关注Share Video，时长01:54

0/0

00:00/01:54 切换到横屏模式 继续播放进度条，百分之0[Play](javascript:;)00:00/01:5401:54[倍速](javascript:;)*全屏* 倍速播放中 [0.5倍](javascript:;)  [0.75倍](javascript:;)  [1.0倍](javascript:;)  [1.5倍](javascript:;)  [2.0倍](javascript:;)  [超清](javascript:;)  [流畅](javascript:;)  Your browser does not support video tags

继续观看

RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源

观看更多Original,RoboParty Lab 正式成立！三项人形机器人训练、控制与数据工具链开源Roboparty萝博派对已关注Share点赞WowAdded to Top Stories[Enter comment](javascript:;)  [Video Details](javascript:;)

**Dataset Analysis & Visualize**

如果想要找到1000条数据集里面水平运动速度在3m/s的数据，可通过hhtools内置一体化运动数据分析与 3D 可视化模块，支持关节轨迹曲线、重心变化、接触点热力图等多维度数据解析，辅助开发者快速清洗、筛选高质量训练数据，一站式完成数据预览、质检、分类导出，搭建轻量化机器人动作数据处理工作台。

**RoboParty Lab：面向下一代人形机器人开发者的开放实验室**

RoboParty Lab 将延续 RoboParty 的 Branding 与开源基因，聚焦人形机器人前沿学术与工程突破。

当前，RPLab 将重点围绕四个核心方向打造**Party OS**：

**Humanoid Locomotion**

**人形机器人通用基础运动控制**

为人形机器人提供足够强大的数据采集、生成和处理的Infra&tools；Sim2real&Real2sim infra；BFM (Behavior Foundation Model) 基础通用运动模型。

**Humanoid Perceptive Interaction**

**人形机器人感知交互系统**

让机器人在复杂环境中更稳定、更自然、更高效地完成行走、奔跑、跳跃、攀爬等基础运动能力；场景交互（Human Scene Interaction, HSI); 物体交互（Human Object Interaction, HOI)。

**Humanoid Whole-Body Manipulation**

**人形机器人全身协调控制与操作**

搭配足够强大的BFM运控基座、为人形机器人研发具备scale能力的VLA、World Model大模型能力。

**Agentic Humanoid**

**人形机器人智能体系统**

利用 Agent + Skills 系统级架构实现低成本高智能。

围绕Party OS，RoboParty Lab 将持续开放内部研发中沉淀的工具链、实验项目与工程文档，包括但不限于 motion generation, motion retarget、codebase、sim2real、camera infra、state estimator、VLA infra、world model infra以及面向开发者的实验性工具。

我们希望 RP Lab 不只是一个展示页面，而是一套持续生长的开放技术系统。

开发者可以在这里看到 RoboParty 正在做什么，也可以真正参与其中：使用工具、提交 issue、贡献代码、适配新的机器人结构、复现新的动作能力，甚至和我们一起定义下一代人形机器人的研发方式。

**不只是工具，**

**而是开源人形机器人 Infra 的开始**

MimicLite codebase 和 hhtools 不是 RoboParty Lab 的全部，它是第一个开放样本。真正重要的不是发布某一个工具，而是逐步建立一种机制：

把内部研发过程中高频使用的工具链、工程经验、实验流程和技术模块，持续沉淀为外部开发者也可以理解、使用和参与的开放基础设施。

这也是 RoboParty 对“开源”的理解。开源不只是把代码放到 GitHub 上。开源是一种组织方式：让问题被看见，让路径被复现，让能力被共享，让更多聪明的年轻人可以站在彼此的肩膀上，继续往前走。

开源也是一种加速方式：当更多开发者、研究者、工程师和学生能够参与真实问题，工具会更快被验证，系统会更快被修正，新的想法会更快进入真实机器人。

从 ROBOTO Origin 到 RoboParty Lab，RoboParty 正在从“开源一台人形机器人”，进一步走向“建设开源人形机器人基础设施”。

ROBOTO Origin 让开发者看到，一台人形机器人如何从 0 到跑。

RoboParty Lab 则希望让开发者看到，一家高迭代机器人团队如何把本体、控制、数据、模型、工具链与社区协作，持续沉淀为可复用的开放系统。

**为全球极客，打开一片技术净土**

我们欢迎全球开发者关注 RoboParty Lab，体验 hhtools，提交反馈，贡献代码，适配新的机器人结构，创造新的动作 Demo，也和我们一起探索下一代人形机器人基础设施的可能性。

人形机器人的未来，不应该只被少数封闭系统定义。

它应该属于那些真正敢想、敢做、敢开源的人。

**RoboParty Lab 正式开启。**

**让每一个有能力、有技术、有想法的人，被这个世界看到。**

**欢迎加入 RoboParty Lab**

**招聘网页：**

https://lab.roboparty.com/join-us

**招聘邮箱**：

hr@roboparty.com

**RoboParty will save the world!**

**END**

**关注我们**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/a7nfRicgiaaBRpOorsx4AL8T5DKuIoFGCESqrHUo5yOsq9z2SJMnhWElhlHoLCrkBogypSLibsmbszXTXbIyI49JzUsG2ulpAia8W7TWpupKYicc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)
