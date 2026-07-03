---
title: "我把最近Loco-Manip工作重新梳理了一遍：人形机器人怎样与物理世界接触，数据、策略、力控和VLA各自解决什么"
author: 具身智能研究室
date: "2026-07-03 08:50:00"
source: "https://mp.weixin.qq.com/s/UjShbwl8p1h9ukymfiRNaw"
---

# 我把最近Loco-Manip工作重新梳理了一遍：人形机器人怎样与物理世界接触，数据、策略、力控和VLA各自解决什么

读者交流入口

添加作者备注「具身求职帮帮群 / 运动控制算法交流群」。

|  |  |
| --- | --- |
| 作者微信 作者微信 | Haley 商务合作 Haley 商务合作 |

这篇只讨论人形机器人里的 Loco-Manip，也就是移动和操作同时发生的任务。这里的接触不只是一只手碰到物体，还包括脚底支撑、身体重心、物体受力、负载摆动、触觉反馈，以及上层模型怎样调用全身动作。

我整理了一条链路：

****接触数据怎么来？**接触怎么进入策略？生成式方法怎么补长尾数据？接触发生之后怎么稳住？VLA 和世界模型怎么调用接触能力？**

## 数据从哪里来

在现在data\_driven的时代，策略能学到什么，很大程度取决于数据里有没有真实的接触结构。

这里真正缺的是**带物体状态、场景约束、接触时序、本体可执行性的交互数据**。普通动作数量再多，如果只剩人体姿态，策略后面也很难凭空学出稳定接触。

OmniRetarget 处理的是重定向里的交互保持。传统重定向容易只保留人体姿态，丢掉人和物体、人和场景之间的关系。OmniRetarget 关注脚底支撑、手物接触、身体和场景约束，让机器人碰到该碰的物体、踩到该踩的位置。

OmniRetarget： https://omniretarget.github.io

**HumanX：**https://wyhuai.github.io/human-x/

**HDMI：**https://hdmi-humanoid.github.io

**SUGAR：**https://tianshuwu.github.io/sugar-humanoid/

HumanX、HDMI、SUGAR 都从人类视频里拿交互经验。HumanX 把视频转成可学习的人形交互技能；HDMI 从 RGB 视频中恢复人和物体轨迹；SUGAR 先抽取运动轨迹和接触先验，再通过技能优化和蒸馏得到可部署策略。三者共同面对一个问题：**视频里有丰富接触，却没有天然的机器人动作标签。**

![图2：HumanX 从人类视频生成交互数据，再训练人形交互技能。](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXtOLsr3XHe1frBRSy6ciaZnVPJ5QkXt6lX7GN4qZDILWxEqNpSVIvbCBcQ64NLQ1YS0qvWjxJA1XiaTDDuKJoCKtSVX1Qibbbu1U/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

图2：这类方法更值得看的，是人类视频如何被加工成机器人可训练的交互数据。

Human-as-Humanoid 和 HumanoidUMI 处理的是另一类入口。Human-as-Humanoid 把 Ego-Exo 人类视频转成人形动作标签；HumanoidUMI 用轻量设备采集机器人无关示范，再把关键点、腕部视角、夹爪动作接到人形全身控制器上。它们降低采集成本，也把跨本体误差带进了控制链路。

Human-as-Humanoid：https://zgc-embodyai.github.io/Human-as-Humanoid/

**HumanoidUMI：**https://arxiv.org/abs/2606.27239

**HumanoidMimicGen：**https://humanoidmimicgen.github.io/

**VLK：**https://vision-language-kinematics.github.io

HumanoidMimicGen 和 VLK 更偏向于数据制造。HumanoidMimicGen 通过全身规划生成Loco-Manip示范；VLK 在重建场景中合成视觉、语言、运动轨迹三元组，让策略同时看到第一视角图像、任务指令和全身轨迹。

人形 Loco-Manip 真正稀缺的是带着物理约束的接触数据，动作片段本身反而没那么稀缺。

人能做到的动作机器人未必能做；视频里看起来稳定的接触，真机上可能缺少摩擦、支撑或可行关节姿态。

## 接触怎么进入策略

这一组的核心分歧在于：**接触被写成什么信号。**SceneBot 写成逐身体部位接触标签，OmniContact 写成接触流，CoorDex 写成身体和手之间的协同接口，HALOMI 写成头手目标和主动感知，WT-UMI 写成触觉和力监督，CEER、Pro-HOI、CWI 则更关注全身控制接口。

SceneBot 是一个很清楚的起点。它把**逐身体部位接触标签**放进策略输入，同时从人类动作里反推**场景交互图**，再重建可训练的地形和物体。参考动作因此多了一层物理含义：左脚该踩台阶、双手该持续接触箱子、骨盆可能要和椅面形成支撑。

**SceneBot：**https://ericcsr.github.io/scenebot/

**OmniContact：**https://omnicontact.github.io/

**CoorDex：**https://skevinci.github.io/coordex/

**HALOMI：**https://halomi-humanoid.github.io

**WT-UMI：**https://wt-umi.github.io/WTUMI/

CEER：https://robotproject8.github.io/ceer\_page/

Pro-HOI：https://arxiv.org/abs/2603.01126

CWI：https://arxiv.org/abs/2606.27676

![图1：SceneBot 将接触标签、参考动作和场景交互放进统一运动跟踪框架。](https://mmbiz.qpic.cn/mmbiz_png/icibRpSZ9SJEU6grZeicwmV4Bg1vCwA7S4X0rJlwkaQdIh18YQMnkzq3t01B8h2TzNTRwbyibBvVyAFBrIoXFaKgYIib6F6dM5ed1O1hsiaGPI8Dk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

图1：重点看接触标签和场景交互图如何进入训练链路。

接口拆解

1**OmniContact**：接触流接口。长程 Loco-Manip 会经历接触建立、保持、转移和断开。**接触流把技能衔接里最容易丢掉的接触时序显式写了出来。**

2**CoorDex**：身体-手协同接口。边走边抓、边退边开冰箱、拿起物体后再转身，失败经常来自身体支撑没有跟上。CoorDex 用协调残差策略连接身体先验和手部先验，核心是**让手上的接触质量和身体支撑同时成立**。

3**HALOMI**：主动感知接口。它从人类示范里收集头部和手部目标，让机器人学会看哪里、靠近哪里、手该到哪里。头部视角和手部目标是接触建立的前置条件。

4**WT-UMI**：触觉和力接口。它用全身触觉接口采集触觉图像、接触力和末端位姿，再用力监督规划接触相关动作。**接触质量、受力大小、接触分布会直接改变后续动作。**

5**CEER / Pro-HOI / CWI**：全身控制接口。CEER 用**末端执行器-根部接口**，Pro-HOI 用**根节点轨迹**，CWI 用**复合式全身模仿**。它们共同指向一个判断：**接触属于全身控制接口的一部分，不能只压在手部局部控制里。**

这条路线还没有统一答案。接触标签、接触流、触觉力、根节点轨迹、身体-手潜变量都能描述接触，但依赖的数据和控制接口不同。

## 生成式路线怎么补数据

只靠真实示范和遥操作，接触数据很难扩到足够大。生成视频、扩散模型、3D 资产、高斯溅射世界和仿真闭环频繁出现，原因就在这里。

生成式路线补的是长尾接触场景。判断它有没有价值，不能只看视频像不像，得看它能不能提供**可恢复、可跟踪、可训练**的接触轨迹。

GenHOI 用生成视频驱动人形物体交互，重点落在从生成视频里恢复交互轨迹，再让机器人零样本执行。Imagine2Real 也利用视频生成先验，但更强调稀疏关键点和零样本人形物体交互，减少对精细几何建模和复杂形态重定向的依赖。

**GenHOI：** https://arxiv.org/abs/2606.12995

**Imagine2Real：** https://arxiv.org/abs/2605.22272

**GRAIL：** https://research.nvidia.com/labs/dair/grail/

**Humanoid-DART：** https://arxiv.org/abs/2606.26855

GRAIL 把 3D 资产和视频先验放在一起，用数字资产生成 Loco-Manip 数据，降低真实场景重建和真机遥操作成本。Humanoid-DART 用扩散模型生成轨迹，再让强化学习策略跟踪这些目标轨迹。这里扩散模型承担的角色很明确：**扩大可以训练的目标空间。**

![图3：GRAIL 用 3D 资产和视频先验生成多样化人形 Loco-Manip 数据。](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEXtibgl7CG70Ipz1VZ3dFUZuqKJm2hvw4R8kNG0F3WXlGHXYVKtYQNgibw1URDclOQLhZHDSSjK8GZiclVFkrxalaEBVjtlQeJOs4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

图3：生成式路线的关键不在画面，而在这些资产和先验能不能转成可训练的接触过程。

**LEGS：** https://legsvla.github.io/

**OASIS：** https://arxiv.org/abs/2606.08548

**SIMPLE：** https://psi-lab.ai/SIMPLE

LEGS、OASIS、SIMPLE 属于基础设施侧。LEGS 用高斯溅射重建真实场景外观，再无遥操作生成 VLA 训练数据；OASIS 从仿真资产、仿真遥操作和域随机化走向真实人形部署；SIMPLE 把接触动力学、视觉渲染、任务资产和数据采集放在一起。

生成式路线真正要证明的，是生成出来的接触时序、物体运动和身体支撑能不能被机器人执行。

这里必须做物理验证。手在视频里看起来抓住了物体，脚底是否打滑、物体是否被正确支撑、重心是否越界，仍然要靠物理一致性检查和控制验证。

## 接触发生之后，模型应该怎么稳住

很多真机失败并不来自机器人完全不会动。更常见的是接触已经建立，但持续过程出问题：手上的力变了，脚下支撑变了，物体位置变了，身体还在跟踪旧轨迹，任务链断掉。

这一类论文处理的是接触后的稳定性：**力怎么适应、柔顺性怎么调、负载怎么稳、强接触下身体怎么反应。**位置跟踪只是开始，持续接触才是真机难点。

FALCON 关注力自适应人形移动操作。它把下肢稳定和上肢末端任务分工处理，并通过 3D 力课程让策略逐步适应外力。HMC 走异构元控制路线，在位置控制、阻抗控制、力位混合控制之间动态选择。这个思路偏工程，也更接近真实接触任务。

**FALCON：** https://lecar-lab.github.io/falcon-humanoid

**HMC：** https://loco-hmc.github.io

**WoCoCo：** https://arxiv.org/abs/2406.06005

![图4：FALCON 展示推车、搬运和开门等力自适应人形移动操作任务。](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEUO28JgP4yNhqnHXfeNAhXcQCVly5ubZaguY7IVcy4oUrjEfnNhb8JIPibHibtI3icKLBNwL4ANh9DhymnmPXzn8AM2gZwiaThleDg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

图4：这里可以看到，真实任务里的接触会持续受力，也会持续逼着机器人调整身体。

WoCoCo 通过顺序接触学习全身控制。它把任务拆成多个接触阶段，降低长程强化学习的探索难度。这个方法和 OmniContact 的接触流有相近的问题意识：接触有顺序、有阶段、有切换条件，不能只按单帧事件处理。

GentleHumanoid 和 CHIP 都把重点放在柔顺性。GentleHumanoid 将上肢柔顺接入全身跟踪策略，CHIP 通过事后扰动学习自适应柔顺性。开门、擦拭、推车、协作搬运里，过硬的位置跟踪往往会把接触变成冲击。

**GentleHumanoid：** https://arxiv.org/abs/2511.04679

**CHIP：** https://nvlabs.github.io/CHIP/

**HOIST：** https://arxiv.org/abs/2606.00252

**Thor：** https://baai-aether.github.io/baai-thor/

HOIST 处理悬挂负载，难点是欠驱动物体会摆动，机器人只能通过全身动作间接影响负载。Thor 面向强接触环境，关注受力后的全身反应。把这几篇放在一起看，它们都在补同一块能力：**接触建立以后，机器人要继续稳定地和世界交换力。**

位置跟踪只能说明机器人到了那里，力和柔顺决定机器人接触以后还能不能继续工作。

力控制、阻抗、柔顺、负载稳定、强接触反应都和硬件、任务、物体属性高度相关，落到新机器人和新物体时还要重新验证。

## VLA 和世界模型怎么调用

VLA 和世界模型如果只输出目标点或粗动作，接触问题还是会被丢回底层。关键在于：上层模型能不能调用一个**带接触结构的全身动作接口**。

OpenHLM 讨论全身原生人形 VLA，关心 VLA 能不能直接面对人形完整动作空间。WholeBodyVLA 走统一潜变量路线，让移动和操作在同一个潜在空间里学习。它们都在尝试避免上肢和下肢割裂。

**OpenHLM：** https://openhlm-project.github.io/

**WholeBodyVLA：** https://opendrivelab.com/WholeBodyVLA

**ROVE：** https://xpeng-robotics.github.io/rove

ROVE 处理人形 VLA 后训练里的人工接管问题。人工接管数据不总是专家示范，接管里常有犹豫、低效和错误；直接模仿会把这些问题一起学进去。

MotionWAM、HAIC、WOLF-VLA 把问题推向世界模型、动力学感知和最优控制。MotionWAM 用世界动作模型和全身 motion token 做实时人形 Loco-Manip；HAIC 表示动态占据、碰撞边界和接触可供性；WOLF-VLA 用全身最优控制生成动态一致的数据，再训练面向 VLA 的人形策略。

**MotionWAM：** https://arxiv.org/abs/2606.09215

**HAIC：** https://haic-humanoid.github.io/

**WOLF-VLA：** https://arxiv.org/abs/2606.25591

![图5：MotionWAM 将世界动作模型接到实时人形全身移动操作上。](https://mmbiz.qpic.cn/sz_mmbiz_png/icibRpSZ9SJEUtlg02YxAvUOKiaudTN7JhGgSCIaiaDZNw5uGFBEHseAia1D7VYBaiaKonqAKpXmKFUkd5WVsyicuN3ic1v4cNicIuudP9ARnAws6GyA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

图5：上层模型最终仍然要落到全身动作接口，接触能力如果没有进入接口，预测得再好也很难上真机。

这几篇共同说明，VLA 和世界模型不能绕开接触。它们负责理解任务、预测后果、调度行为，但最后仍然要落到身体上。**如果底层动作接口没有接触结构，上层模型很容易生成看起来合理、真机上不稳定的动作。**

## 还没想清楚的地方

统一接触接口、生成式物理一致性、触觉和力数据规模化、VLA 是否真的学到接触因果，这几个问题还没有被彻底解决。

开放问题

01**统一接口**：接触标签、接触流、触觉力、根节点轨迹、全身 motion token 还没有收敛成统一接口。不同论文都在定义自己的中间表示，长期会带来系统割裂。

02**生成式验证**：生成式视频里的接触时序和物体动力学还缺少稳定验证方法。视觉上能过关，不等于力学上能执行，这会影响 GenHOI、Imagine2Real、GRAIL、Humanoid-DART 这类路线能走多远。

03**触觉和力数据**：WT-UMI、FALCON、HMC、GentleHumanoid、CHIP 都说明力和柔顺很重要，但规模化采集和跨硬件复用还不轻松。

04**VLA 接触因果**：OpenHLM、WholeBodyVLA、ROVE、MotionWAM、HAIC、WOLF-VLA 都在往全身任务模型走，但如果底层控制器处理不了力，最后还是落不到真机。

05**接口互通**：接触数据、接触表示、力反馈、全身控制、上层任务模型需要互相接上。各说各话只会让系统越来越复杂，真正能复用的部分不多。
