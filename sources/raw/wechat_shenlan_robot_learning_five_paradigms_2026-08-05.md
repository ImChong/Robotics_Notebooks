---
title: 机器人学习算法五大体系详解：模仿、强化、多模态、持续学习……
author: 深蓝具身智能
date: "2026-08-05 10:56:00"
source: "https://mp.weixin.qq.com/s/r2zUtQfwH_r0WHrnY4CHuA"
---

# 机器人学习算法五大体系详解：模仿、强化、多模态、持续学习……

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/kaugqJpv9nuCktylvYoMKHYNAVojoRUpfyf1py08JvUnkfPXArzj4t5bMiaS6RBCXHHGhf8xlyw8icHrJcjEyYoA/640?wx_fmt=gif&from=appmsg#imgIndex=0)

![伯克利HIL-SERL：强化学习基于视觉Franka机器人精确灵巧操纵策略,1-2.5 小时训练任务实现100% 成功率- PNP机器人 具身智能|具身方案|机器人|灵巧手](https://mmbiz.qpic.cn/mmbiz_gif/uwFbeBKoFGfk3O1tnZqvibZQDDWibvjqpn329VBMkPZusTiaaRElian1BETMyzCNbgeIISBe5qM88Pd4Tc5vxw8LiauFboUfRJEGuYdwF9iaAQg04/640?wx_fmt=gif&from=appmsg#imgIndex=1)

机器人“自主学习”是个伪命题，大部分能力还是人给的？

——从模仿学习到持续学习

机器人获取技能，到底该走哪条路？

> 教一个孩子系鞋带，本质上就是一套完整的具身智能训练流程：
>
> 你放慢动作让他看（模仿学习），用语言纠正他的力度（强化学习），他观察别人系不同鞋带（视频学习），听懂“穿过那个洞”的指令（多模态学习），最后把这项技能稳固在大脑里，同时不忘扣纽扣（持续学习）。

五条路线的起点不同，但都在回答同一个问题：面对没有被完整写进程序的现实环境，机器人究竟依靠什么信息来改进行为？

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcAfzDFD8sgxfJKClhRwfU57FWur7WLnopQSD8EY2EdvstGNibhJqoKNZjoG6s0Og1CM7x7rtTRD7F6eiaKQEXDvn84VW8VZWJPA/640?wx_fmt=png&from=appmsg#imgIndex=2)

▲图1 | 机器人学习算法全景：从示范、反馈、多模态数据与连续经历理解五类范式 ©【深蓝具身智能】编译

我们先从最直观的一类方法：从人类示范中学习开始。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4YWHX9iazKkdkgh363zh9GFAfZia4RWWoYhutUeS8g43MnicLMfe9kUAZg/640?wx_fmt=jpeg&from=appmsg#imgIndex=3)

## 模仿学习：把“正确示范”变成可执行策略

## 核心思路：先让人类完成任务，记录机器人在不同情况下应采取的动作，再让模型学习这种从观测到动作的对应关系。

## 只要能够取得质量较高的示范数据，它就能为机器人提供一个可用的策略起点。

不过，学习“下一步该怎么动”并不等于学会了完整操作。对于需要持续协调的任务，如果模型每个时刻都独立预测一个动作，前一步的小误差就可能影响后续决策。

于是，一种自然的改进是：不把动作看成彼此割裂的单步指令，而是把短时间内连续发生的一段动作一起建模。

> ALOHA 项目是这一思路在精细双臂操作中的一个例子。

它所采用的 ACT（Action Chunking with Transformers）会根据多视角图像和关节状态，一次预测一段连续动作，而不是只给出下一时刻的单一指令。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGfYqxxM4TagkWnEsHiaK13sib8T9jw70ublk00Hu2qeg9taljqNOgqicueticAicN3xtZ5xaEiajv9tkHcMSwuuMvOBkkEOibXNVveY68/640?wx_fmt=png&from=appmsg#imgIndex=4)

▲图2 | ALOHA 项目中的 ACT 策略：模型结合多视角图像与关节状态，生成一段连续的机器人动作 ©【深蓝具身智能】编译

这种设计的重点，不是让机器人机械地重复某个姿势，而是学习一段动作内部的协调关系。

在 ALOHA 所设定的穿入扎带、打开半透明调料杯、插入电池等任务中，系统以较少示范数据展示了相应操作能力。

![]()已关注Follow  Replay    Share     Like  Close**观看更多**更多


*退出全屏**切换到竖屏全屏**退出全屏*深蓝具身智能已关注Share Video，时长00:23

0/0

00:00/00:23 切换到横屏模式 继续播放进度条，百分之0Play00:00/00:2300:23倍速*全屏* 倍速播放中0.5倍0.75倍1.0倍1.5倍2.0倍超清流畅 Your browser does not support video tags

继续观看

机器人学习算法五大体系详解：模仿、强化、多模态、持续学习……

观看更多转载,机器人学习算法五大体系详解：模仿、强化、多模态、持续学习……深蓝具身智能已关注Share点赞WowAdded to Top StoriesEnter comment Video Details

但这一结果依赖于特定硬件、数据采集流程和任务条件，不能直接外推到其他机器人或场景。

### 难点不在“学到第一条轨迹”，而在偏离后的恢复

模仿学习的一个关键风险是分布偏移。

训练数据多来自专家顺利完成任务时经过的状态；但机器人部署后，微小误差可能使它到达训练集没有覆盖的状态。

此时，策略输出的下一步动作未必能够把系统带回正确轨迹，误差可能在时序任务中累积。

这也是后续方法需要重点处理的问题。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGeMzNsMaxAMdMRbtqxvGF5Eia0o0TQHicicgiamZCghg8I0y0nEv5a7yMwVheSJbicicS3UO8psrBJsw3GLuwK2dia3g5JAUWkU9Vj5xg/640?wx_fmt=png&from=appmsg#imgIndex=5)

▲图3 | 模仿学习中的误差累积风险，以及 DAgger 的交互式数据聚合思路 ©【深蓝具身智能】编译

> DAgger（Dataset Aggregation）提供了一种较有代表性的改进思路

让当前策略在环境中运行，并让专家为策略实际访问到的状态补充正确动作标注，再把这些数据并入训练集。

它的价值在于把“机器人可能犯的错误”纳入数据收集过程。

不过，DAgger 并不保证在所有任务中都能消除误差；专家干预成本、任务安全性和状态覆盖范围仍然会影响实际效果。

既然专家示范存在覆盖范围和采集成本的限制，那么，有没有一种方法可以让机器人自己去摸索“好”与“坏”的边界呢？

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibqgVaodXH45G6Pdbk9xSEsUtlicqgxKkAiaK0P8QzGwuLiatibYiaIagQoOg/640?wx_fmt=jpeg&from=appmsg#imgIndex=6)

## 强化学习：用反馈而非逐步示范优化行为

核心思路：强化学习不要求专家为每一步提供目标动作，而是让机器人在环境中反复交互，并根据奖励信号更新策略。

奖励可以编码任务完成、进度、能耗或违反约束等信息；算法的目标是让策略在长期交互中获得更高的累计回报。

它尤其适合那些“怎样做”不容易写成固定示范、但“做得好不好”能够定义的任务。

![Image](https://mmbiz.qpic.cn/mmbiz_gif/uwFbeBKoFGcHnY41BupTpKeBNdibsnJSssxo5nhI5Scvf8hleBCAVibH2hLDacdiceWJESXmv5HFCTft6nrTjpG9ticT0xUARTEbSK3xWqZGbQs/640?wx_fmt=gif&from=appmsg#imgIndex=7)

但在真实机器人上进行大量试错，通常伴随着较高成本和安全风险：机器人可能损坏自身、物体或周边设备。因此，研究与开发中常先在虚拟的仿真环境里训练策略，再在真实系统上验证迁移效果。

在仿真环境里训练最大的痛点是速度。为了让机器人学得更快，Isaac Gym 系统将物理模拟与神经网络训练都放到 GPU 上执行，减少了数据在不同计算设备之间的传递。

这样，机器人可以在大量并行环境中同时收集经验；在该系统的特定基准设置下，这种设计带来了数量级的训练加速。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGdlFib2ABVwecx7PbHLVG1iaKkLiczVic8YxGRcUDoUoj1IofcokibcQLoGEvxbk9Lk3baBCHBXaKuLpAupkJmZXdkwCR4AZV8TNmxs/640?wx_fmt=png&from=appmsg#imgIndex=8)

▲图4 | Isaac Gym 的 GPU 端到端训练管线：学习框架、环境逻辑、张量 API 与物理模拟器之间的数据流 ©【深蓝具身智能】编译

### Sim2Real：仿真中学到的策略，如何适应真实系统？

即使提高仿真保真度，真实系统在视觉渲染、动力学、传感与执行等环节仍可能与仿真不同。这样的“仿真—真实差距”可能使在仿真中有效的策略在真实系统上出现性能下降；这正是 Sim2Real 研究要处理的问题。

域随机化（Domain Randomization）是常见的应对方法之一。它

不追求构建单一且完全精确的仿真场景，而是在训练中有意识地改变光照、纹理、物体位置或相机视角等条件，让模型在更多变化中学习较稳定的特征。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcwVTnhiaySjThiavPlmOwkzSFicoGKicdVXO3HSHDzic9Vh3P3RYQTkbvGgfpsDvNPT9SmsSHx11TzicEknA1icIMxx3eyJdpE6zIlwQ/640?wx_fmt=png&from=appmsg#imgIndex=9)

▲图5 | [域随机化](https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247507074&idx=1&sn=530cd11ccaa86662d3f912fc31a25794&scene=21#wechat_redirect)：训练时随机改变仿真场景的视觉渲染（左），以扩大策略训练阶段覆盖的视觉变化范围；右图为真实测试场景 ©【深蓝具身智能】编译

这样做的目标，是降低策略对某种固定纹理或光照条件的依赖，更多关注与任务相关的物体位置和形状。

在四足机器人研究中，也有工作将仿真训练的神经网络策略部署到 ANYmal 等真实平台，用于高动态运动和跌倒恢复等任务。不过，仿真训练并不能替代真实系统中的最终验证。

无论是专家示范还是仿真试错，获取数据的成本依然高昂。相比之下，互联网上每天都在产生海量的人类操作视频。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibTvia4qLjYyoM2Do58jX9J71HickLLA3NxCQp6fPljkgY26WIeaoeeYVQ/640?wx_fmt=jpeg&from=appmsg#imgIndex=10)

## 从互联网视频中学习：把人类视频变成“辅助先验”

## 真实机器人数据的采集会受到设备、时间、安全和人工标注成本的限制。与之相比，互联网包含了大量人类完成操作任务的视频。

## 核心思路：从视频中学习（Learning from Video，LfV），是用这些视频补充传统机器人数据，从中提取任务时序、物体交互或物理行为方面的可迁移先验。

需要区分的是：人类视频通常不是带有机器人控制信号的数据集。

视频往往缺少可直接执行的机器人动作标签，拍摄视角也可能与机器人相机不同；人手与机械手在自由度、接触方式和可达空间上也不相同。

LfV 的关键不在于“看一段视频就直接复制动作”，而在于如何弥合这些分布差异，把视频中的信息与机器人数据或模型结合起来。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGcls1RpicOcvfohYYVeT8oxStXp5JFQE2E1ksuUzmYoPGLeKRtqyL5ME3VPicr3KacvMxHeuVqUfy4w242bn7JGouUyJ2bq7jdw8/640?wx_fmt=png&from=appmsg#imgIndex=11)

▲图6 | VideoDex 将人类视频作为动作先验：检测人手后，通过手部姿态和相机轨迹重定向到机器人，用于预训练策略网络 ©【深蓝具身智能】编译

> VideoDex

它尝试从人类视频中提取视觉和动作先验，将人手姿态和相机轨迹重定向到机器人，以此预训练策略网络；随后仍需收集少量机器人真实示范来完成后续训练。这类视频数据目前不能完全替代真实机器人上的交互，但可作为有价值的补充来源。

视频可以提供操作过程的视觉线索，但它很少直接告诉机器人任务的语义目标。

要让机器人理解人类的语言指令，还需要把视觉、语言和动作放到同一个学习过程中。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4yZibwaOWpB3DrcxuiafpXicx2ibHiaHAZFr7ptU6ud2hsxgCXvV0JGHtTDw/640?wx_fmt=jpeg&from=appmsg#imgIndex=12)

## 多模态学习：让图像、语言与动作进入同一学习过程

视觉—语言—动作模型（Vision-Language-Action，VLA）关注的是如何将机器人所见的图像、用户给出的自然语言指令，以及动作输出连接起来。

与仅根据图像输出动作的策略相比，VLA 试图引入语言层面的任务语义；与只处理文本或图片的视觉语言模型相比，它又需要把输出落实为可执行的机器人动作。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGfCzfwul7Djcp0xKBYZcL05aToDS5ELgRWD13g1GADef75FldGWNlMQYShwicwPx1lzfj88f4OCQyIIreBOtLHWFxBPjhbsWicibo/640?wx_fmt=png&from=appmsg#imgIndex=13)

▲图7 | RT-2 将机器人动作表示为文本Token，并与互联网视觉—语言数据共同微调，从而在推理时解码出闭环控制动作 ©【深蓝具身智能】编译

> RT-2 模型

它将机器人的动作表示成文本Token（Token），使机器人轨迹数据能够与视觉—语言数据放进同一个训练过程。这种统一表示的目的，并不是让机器人“像语言模型一样说话”，而是让模型在理解图像和指令时，也能输出与之对应的控制动作。

在 RT-2 的评测中，研究团队考察了新物体泛化、对训练数据中未出现指令的理解，以及基础语义推理等问题。这些设置说明，视觉语言模型中的部分语义知识可以与机器人控制数据结合。

不过，由于这些评测并未覆盖开放环境下的长程规划、系统级可靠性与安全保障，我们尚不能据此推断大模型已经解决了机器人控制的所有问题。

进一步说，如果语言知识可以共享，不同机器人积累的操作经验是否也可以共享？

> Open X-Embodiment

将来自多种机器人平台的数据放入共同训练框架，并在其评测中观察到跨机器人数据带来的正迁移。硬件结构、动作空间和传感器差异仍是现实约束，但这种尝试为跨平台数据复用提供了一个可验证的方向。

至此，无论是单一任务的示范，还是跨越模态与平台的通用模型，解决的主要都是“如何学会新能力”。

但在真实世界中，学习并不是一次性的，这也带来了机器人学习的重要挑战之一：

如何在获得新能力的同时，尽量不丢失旧能力？

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4hKdH0P2rRHX1TlxUqlAx7X6m2hcl7XttttyRW05mhbEa1msX7zEzvw/640?wx_fmt=jpeg&from=appmsg#imgIndex=14)

## 持续学习：新任务增加后，旧能力如何不被遗忘

持续学习的关注点与前四类方法不同。

它讨论的是：当数据分布或学习目标随时间变化，且机器人无法一次获得全部训练数据时，系统能否继续学习新技能，同时尽量保持已经学会的能力。

这就引出了持续学习的核心难题——灾难性遗忘：如果模型只针对新任务进行参数更新，旧任务上的表现可能随之下降。

为了缓解这一矛盾，研究者们梳理了多种应对策略，包括限制重要参数改变的正则化方法、动态扩展网络容量的架构方法，以及让模型重新复习旧经验的记忆回放技术。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGfauGkPZqg8umb2UicNrNdpuWic3Aj2gTia4YjGKzlI36ATduFyeiaZBKOib0K1Xn3hATMBML50YOhee6EyGbBicdufaz0CHbia6m81AA/640?wx_fmt=png&from=appmsg#imgIndex=15)

▲图8 | 持续学习综述归纳的四类常见策略：正则化、经验回放、生成式回放与架构方法；重叠区域表示部分方法可同时具备多类特征 ©【深蓝具身智能】编译

不过，很多持续学习方法仍主要在仿真环境或静态数据集上评测。真正部署到机器人上，还必须考虑存储、计算、在线采样成本和安全等工程条件。

因此，持续学习更适合被理解为一个重要的研究目标和系统评估框架，而不是可以不加条件直接部署的通用能力。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX7OGyOG8gzibCyLX91kbhEcWl0mnLk5Zb5uVRabIn51LEKNicYT8OlZZQ/640?wx_fmt=png&from=appmsg#imgIndex=16)

## 从选择算法，走向设计学习组合

回看这五类范式，可以发现它们解决的是机器人学习过程中的不同问题：模仿学习提供示范，强化学习提供反馈，LfV 扩展数据来源，VLA 将视觉、语言和动作连接起来，持续学习则关注能力在时间维度上的保持与适应。

在实际系统中，这些方法往往需要配合使用。例如：

示范数据可以为策略提供初始能力，强化学习可用于进一步优化局部表现，视觉—语言模型则为任务理解提供语义信息。

具体如何组合，仍取决于任务定义、数据条件、硬件能力与安全要求。

因此，与其寻找一套能够解决所有问题的学习方案，不如回到更实际的问题：在给定任务、数据和安全边界下，什么学习信号最可靠，什么验证方式最能反映真实能力？

这也是具身智能从研究走向应用时需要持续回答的问题。

编辑｜阿豹

审编｜具身君

##

Ref

1、Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. RSS 2023.

2、Ross, S., Gordon, G., & Bagnell, J. A. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011.

3、Sutton, R. S., & Barto, A. G. Reinforcement Learning: An Introduction (2nd ed.). MIT Press, 2018.

4、Makoviychuk, V., Wawrzyniak, L., Guo, Y., et al. Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning. 2021.

5、Tobin, J., Fong, R., Ray, A., et al. Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS 2017.

6、Zhao, W., Queralta, J. P., & Westerlund, T. Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: A Survey. 2020.

7、Robot Learning from Randomized Simulations: A Review. Frontiers in Robotics and AI, 2022.

8、Hwangbo, J., Lee, J., Dosovitskiy, A., et al. Learning Agile and Dynamic Motor Skills for Legged Robots. Science Robotics, 2019.

9、McCarthy, R., Tan, D. C. H., Schmidt, D., et al. Towards Generalist Robot Learning from Internet Video: A Survey. JAIR, 2025.

10、Shaw, K., Bahl, S., & Pathak, D. VideoDex: Learning Dexterity from Internet Videos. CoRL 2023.

11、Zitkovich, B., Yu, T., Xu, S., et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. CoRL 2023.

12、Open X-Embodiment Collaboration. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. CoRL 2023.

13、Lesort, T., Lomonaco, V., Stoian, A., et al. Continual Learning for Robotics: Definition, Framework, Learning Strategies, Opportunities and Challenges. Information Fusion, 2020.

14、Underactuated Robotics：Direct Collocation and Trajectory Optimization

15、Model Predictive Control：Theory, Computation, and Design

 ****推荐阅读**
[![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcibJS8986MfCcVATGOkcK6lNQfiaTORbuhSFoATTmZ5kA6nV8l8REia7nm4A4OxC1yOePBqrzWHQQd0ALicYANgOoNmRbibChjcAuQ/640?wx_fmt=png&from=appmsg#imgIndex=17)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=3824573915845640194&scene=126#wechat_redirect)[![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGcRfEtsGjVkl7cXB7QYAAib4wOMhdRcvsQicHnmiaxqoibw9LUCGGcPGSYnUPeUlZEoiaBlQezclFhp5yZQ6yLcLAjYeI67pJmvhOMw/640?wx_fmt=jpeg&from=appmsg#imgIndex=18)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=944555238&lang=zh_CN#wechat_redirect)

**![Image](https://mmbiz.qpic.cn/mmbiz_png/qKE443uRvLo6ic3ZPUttmFZ2AefQ4wjHSlQluSDkaxL9icWicpPYYmpo1Wa37Scjhh4AS5VwYJtmlTf5cKMiaIXg5g/640?&random=0.17349735674179656&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&wx_fmt=other#imgIndex=19)**

**【深蓝具身智能】****的原创内容均由作者团队倾注个人心血制作而成，希望各位遵守原创规则珍惜作者们的劳动成果；未经授权禁止任何机构或个人抓取本账号内容，进行洗稿/训练，否则侵权必究⚠️⚠️**


![Image](https://mmbiz.qpic.cn/mmbiz_png/Nabxc8rdYriaKqxCUjcZ8sSCnSNlWpqdI1kyXXQjXbtv95xvACqQoqL2ibbKXt9PB0FLPibKiawGsTcQrnKDGWVw2Q/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1#imgIndex=20)

点击❤收藏并推荐本文**
