---
title: "最大误区：Sim-to-Real 不是训完之后的事情！从辨识到适应，这些工作贯穿全程"
author: 深蓝具身智能
date: "2026-07-28 10:56:00"
source: "https://mp.weixin.qq.com/s/6rbLz_6nQz9z6kma9K4BFQ"
---

# 最大误区：Sim-to-Real 不是训完之后的事情！从辨识到适应，这些工作贯穿全程

![Image](https://mmbiz.qpic.cn/mmbiz_gif/uwFbeBKoFGcOjfZUY5cjPiaWfKaJBVfynP3CSCl3x4XuzM2PnIMrg9gpz94oEa1jWIKupJutlBmE0r7aptSpxbJcyHs3JF41gN6Qib3R9Jybk/640?wx_fmt=gif&from=appmsg#imgIndex=0)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGesyQ222PSmiaH6F54b23JYLf8jRxE14KF8L0RiaDmaqEvRjEfN35AroWtmZ9Z7gusjCsiaRDGjN98ibAg5CX4ttqYDYOtTIU5c36E/640?wx_fmt=png&from=appmsg#imgIndex=1)

系统不再是”训完部署就结束”，而是在运行过程中持续校准。

——Sim2Real

将仿真中训练的控制策略迁移至真实机器人，涉及模型精度、观测接口、控制分配和域随机化等多个层面的工程决策。但在诸多技术细节之上，一个更为根本的判断往往被忽视：

**Sim-to-Real 并非训练完成之后再考虑的事情，而是从系统辨识阶段就应当启动、并在机器人持续运行中不断校准的完整链路。**

如果将迁移视为训练结束后的「一个独立步骤」，那么辨识、训练、部署三个环节便「彼此割裂」：

仿真模型在策略训练后不再更新，部署时遇到的参数漂移没有反馈回路，前馈补偿依赖的可能是数月前标定的旧值。

这种线性流程的局限在实机测试中会逐步显现，围绕这一判断：

“

本文先探讨如何在仿真里训练出好策略（系统辨识、训练策略、域随机化与课程学习）；随后进入实机部署阶段，仿真策略到实机之间还差什么，以及怎么补上；最后讨论分层安全防护为何必须独立于策略存在。

也就是说，系统不再是“训完部署就结束”，而是在运行过程中持续校准。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGeMcRWtAgoC9qwtfQUONd8xdUicyYlgFFibLyt7WicVYnloaNy2A4cDgIoricfriaicb7VibiaJRHIgicSaJQFb4flNnJvAWibDuJMTY8TTI/640?wx_fmt=png&from=appmsg#imgIndex=2)

▲图1 | Sim2Real 包含实机数据采集、系统辨识、策略训练和部署保护等环节，是一套完整的闭环工程，在运行过程中持续校准。©【深蓝具身智能】编译。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4YWHX9iazKkdkgh363zh9GFAfZia4RWWoYhutUeS8g43MnicLMfe9kUAZg/640?wx_fmt=jpeg&from=appmsg#imgIndex=3)

为什么仿真策略在实机上容易失效？

Sim-to-Real 迁移的首要工作，并非直接调整策略或扩大随机化范围，而是先将这个Gap进行分解，判断其来源：哪些应当通过辨识与前馈直接消除，哪些只能依靠策略的鲁棒性来覆盖。

这一分解直接决定了后续所有工程环节的路径选择：

- 参数误差：

机器人质量、连杆惯量和关节摩擦力等仿真参数与真实硬件存在偏差。

- 难建模的动态与环境差异：

真实机构存在齿轮回差、皮带和结构件柔性，执行器延迟与电机温升也会改变输出特性；地面柔顺性等接触差异同样难以在仿真中完整复现。

- 观测误差：

真实传感器存在噪声、偏置与量化误差，导致策略获得的状态估计与真实状态存在偏差。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGesyQ222PSmiaH6F54b23JYLf8jRxE14KF8L0RiaDmaqEvRjEfN35AroWtmZ9Z7gusjCsiaRDGjN98ibAg5CX4ttqYDYOtTIU5c36E/640?wx_fmt=png&from=appmsg#imgIndex=4)

▲图2 | 面对实机性能下降，首先需要判断误差类型，再选择相应的处理方法。©【深蓝具身智能】编译。

这三类误差需要采用不同的处理方式。

在实际操作中，一种常见的误区是：一旦实机表现不佳，便立刻在仿真中大幅放宽质量、摩擦和时延的随机范围。

这样做虽然能让策略覆盖更多极端情况，但如果范围严重脱离真实硬件，策略就会变得过于保守，进而影响运动质量与能效。

相应的工程处理方式是：

- 对于可建模的误差，优先进行校准；
- 对于难以完整建模的动态和环境差异，利用域随机化进行覆盖；
- 对于随时间变化的参数与工况，采用在线适应。

安全机制则作为独立的部署保障，用于限制测试和运行风险。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibqgVaodXH45G6Pdbk9xSEsUtlicqgxKkAiaK0P8QzGwuLiatibYiaIagQoOg/640?wx_fmt=jpeg&from=appmsg#imgIndex=5)

## 系统辨识：建立可靠的物理基准

在实际开发中，研究者通常会将机器人的 3D 模型（如 URDF 文件）导入仿真器并开始训练。但厂商提供的模型参数未必经过完整的实机校准，部分惯量、摩擦或阻尼参数也可能仍为默认值。

例如，如果仿真中的关节摩擦力设置得较低，而真机减速器的实际摩擦力较大，那么仿真与实机对应的动力学响应将存在明显差异。

因此，Sim2Real 通常会先通过系统辨识建立可靠的物理基准。

其基本流程是：

> 让真实机器人执行一组激励动作，记录关节位置、速度和力矩数据；
>
> 随后在仿真中输入相同的控制信号，比较仿真轨迹与实机轨迹的偏差；
>
> 最后通过优化算法调整摩擦、惯量等参数，使两者的响应尽可能一致。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGdQTWiapajK2hy2T2rKsAlPJod1klc1gUxNdqbXOpMrQoAFVKxaey5SlgRyXrIvSjLHlSTjMsgyw0oMlEJNw8WbgGQ9q6Ku7QX0/640?wx_fmt=png&from=appmsg#imgIndex=6)

▲图3 | 系统辨识的基本闭环：用实机轨迹校准仿真参数，逐步缩小两者的响应差异。©【深蓝具身智能】编译。

系统辨识需要重点关注两个方面。

第一，辨识参数并非越多越好。

如果机器人仅在平地上慢速行走，一些与高频加速度相关的参数便无法被充分激发，此时拟合这些参数容易导致过拟合。因此，辨识所用的激励轨迹需要包含足够丰富的速度和加速度变化。

第二，辨识的目标并不是建立一个固定不变的精确模型，而是为强化学习提供合理的物理基准。

后续的域随机化应围绕这一基准，覆盖真实的硬件公差与测量误差，避免在错误的默认模型上扩大随机范围。

2018 年的 Minitaur 四足机器人研究采用了这一思路。研究人员首先通过系统辨识建立电机与延迟模型，再在此基础上引入随机化，最终将仿真中的跑跳步态迁移至实机。

在这一流程中，模型校准负责降低系统性偏差，随机化负责覆盖剩余的不确定性。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGePE3963Et5KECCfgFDMfoGr1FIssA7kSdw60ywBfAPAVtcMlqmCRTlZD9e1tCOnO8InPhdEib4bUYkGy1tSBnSiaML3yLkvZia8U/640?wx_fmt=png&from=appmsg#imgIndex=7)

▲图4 | Minitaur 论文比较了真实机器人、经系统辨识的仿真模型与未辨识模型的响应。校准后的仿真轨迹更接近实机，为后续随机化与策略迁移提供了更准确的模型基准。©【深蓝具身智能】编译。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibTvia4qLjYyoM2Do58jX9J71HickLLA3NxCQp6fPljkgY26WIeaoeeYVQ/640?wx_fmt=jpeg&from=appmsg#imgIndex=8)

## 观测、动作与奖励设计需要面向实机部署

完成系统辨识后，仿真器具备了与实机较为一致的物理响应。但这只是迁移的前提，策略能否在实机上有效运行，还取决于训练阶段的设计是否面向实机部署的约束。

观测空间是否包含了实机无法获取的特权信息？动作接口与底层控制的职责如何划分？奖励函数是否考虑了硬件安全？

这些问题如果在仿真训练中被忽视，策略即便在仿真中得分很高，也未必能在实机上稳定运行。

### 观测空间：训练信息必须与实机条件一致

在仿真环境中，系统可以获取精确的全局速度、地形高度图和摩擦系数等特权信息。但在真实机器人上，通常只能依赖带有噪声的 IMU（惯性测量单元）和关节编码器。

如果在训练时直接使用了实机无法获取的数据，策略在部署时便会失效。

因此，能够在复杂地形下实现“盲行”的 ANYmal，通常只依赖本体感受数据（如关节位置、速度和历史动作）。

2020 年发表在 Science Robotics 上的研究表明，策略仅通过一段历史动作和状态序列，便能实现从仿真到泥地、雪地等真实地形的零样本（Zero-shot）迁移。

策略并未直接识别雪地类别，而是通过历史数据中的物理交互线索进行判断。例如，足端打滑或被障碍物阻挡，都会在连续的关节反馈序列中形成可识别的特征。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcXGicBLwe3PvRKJnwEA0ibpmcw7RnR599q9R92edMxbw8CW9LvQpmiaNE67J52Y2icZxA4w37syiaIuFoKq2bsKK8jP4QICSm5L8MA/640?wx_fmt=png&from=appmsg#imgIndex=9)

▲图5 | ANYmal 的训练框架将“训练时可见的特权信息”和“部署时能获得的本体感知”分开处理：教师策略先利用完整仿真信息学习，再把能力迁移给只依赖真实可用观测的学生策略。©【深蓝具身智能】编译。

### 动作接口：策略与底层控制的合理分工

强化学习策略可以直接输出关节力矩，也可以输出目标关节位置。两者在工程实践中各有应用场景。

在许多四足机器人系统中，输出目标关节位置是一种常见选择。策略以相对较低的频率生成关节目标，底层 PD 控制器则在更高频的闭环中跟踪这些目标，并处理局部的快速扰动。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGf2H68JUep5tNmpUJOgTX3iboGY2ArPokBPQFjxQtLmG0atFX1VUGzOStnk0Kic5icI5eWRTGRFvEP8dRenAhpv2V1689zhgzcVwA/640?wx_fmt=png&from=appmsg#imgIndex=10)

▲图6 | 策略以较低频率生成关节目标，底层 PD 控制器以更高频率进行闭环跟踪，两者承担不同时间尺度的控制任务。©【深蓝具身智能】编译。

这种架构具有明确的工程分工：

强化学习处理全局、非线性的决策问题，经典反馈控制处理局部且需要快速响应的执行细节。

### 奖励函数：兼顾任务目标与物理可行性

在仿真中，如果仅设置速度跟踪奖励，机器人可能会为了追求速度而产生高频抖动或剧烈的足端冲击。这种策略虽然在仿真中得分较高，但部署到实机后，极易导致硬件过热或机械损伤。

因此，奖励函数不能只包含速度跟踪，还需要加入姿态稳定、动作平滑度、力矩限制和能耗等惩罚项，从而将物理可行性、硬件负载和安全约束纳入优化目标。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4yZibwaOWpB3DrcxuiafpXicx2ibHiaHAZFr7ptU6ud2hsxgCXvV0JGHtTDw/640?wx_fmt=jpeg&from=appmsg#imgIndex=11)

## 域随机化与课程学习：范围和训练顺序同样重要

在基础模型上验证策略后，下一步是提升策略的泛化能力。

- 域随机化（Domain Randomization）

通过在训练中引入质量、摩擦力、延迟等参数的随机变化，避免策略对单一物理模型产生过拟合。

这种随机范围的设定需要依据实际硬件公差与测量误差，而非随意扩大。范围过大容易导致策略陷入过于保守的次优解。

- 课程学习（Curriculum Learning）关注另一个问题：策略应当按照怎样的顺序面对不同难度的训练任务。

如果一开始就让未经训练的策略面对极具挑战性的复杂地形，机器人往往无法收集到有效的正向奖励。

课程学习的思路是：先在平坦地形和微小扰动下建立基本运动能力，随后再逐步增加地形起伏与外力干扰。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGcT7REIqJDB34l8FKZHKloEL61pXHVU2jzWfKYGYvTlYF43HUFq6OOMMZn1Uib3RyljxXyeYSqibPs9eTDjfs0uRTVsibCcFmHcFA/640?wx_fmt=png&from=appmsg#imgIndex=12)

▲图7 | 课程学习根据策略能力逐步提高任务难度，避免训练初期直接面对过于复杂的环境。©【深蓝具身智能】编译。

苏黎世联邦理工学院（ETH Zurich）团队在训练 ANYmal 时，采用了基于游戏启发的课程学习机制，通过大规模并行仿真积累训练经验，并将策略部署至真实机器人。实验结果显示，训练难度调度会直接影响策略的收敛过程。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGdRqHodXicvJFMYvfMEdZdhGY4OR19Fiaicl7lWoWEkzNG5ub0qV9SV7RPvKEXyAf32jCfBpxv2mYHAzbsBPZaSEqHQe1e4sPUINw/640?wx_fmt=png&from=appmsg#imgIndex=13)

▲图8 | “Learning to Walk in Minutes”先在较简单的平地场景中建立基本运动能力，再把策略放进大量并行、难度不同的地形中继续训练，让课程难度随表现自动推进。（图片来源：Rudin et al.，发表于 CoRL 2021，论文集出版于 2022 年）©【深蓝具身智能】编译。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4hKdH0P2rRHX1TlxUqlAx7X6m2hcl7XttttyRW05mhbEa1msX7zEzvw/640?wx_fmt=jpeg&from=appmsg#imgIndex=14)

## 实机部署中的确定性补偿与在线适应

尽管策略在仿真中经历了充分的域随机化训练，实机部署时仍会面临未知的动态变化。

此时，可以将误差进一步划分为可预测的确定性误差和随时间变化的动态误差。

### 摩擦前馈：直接补偿已知的确定性误差

如果在系统辨识阶段已经建立了关节摩擦力模型，便可以在底层控制中引入摩擦补偿力矩。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGcKJQBYIaRLPQn3zoWGfeZoHhcuvGOg1vK0KlkukMNafdymOBCnZY1aHIHDhribgTnq9XrvKyOR3Rqs4R3ibyznVyjCX8YM5Tywk/640?wx_fmt=png&from=appmsg#imgIndex=15)

图9 | 已经辨识清楚的摩擦可以直接进入底层前馈支路，根据关节速度提前补偿，不必等跟踪误差出现后再纠正。©【深蓝具身智能】编译。

对于能够通过动力学公式明确计算、且参数已经得到较可靠辨识的物理量，前馈补偿可以在控制环中显式处理，从而减少策略承担的补偿任务，也更便于分析和调试。

### 在线适应：从交互历史中估计当前动力学状态

在真实环境中，地形摩擦系数、机器人负载以及电池电压都会随时间发生变化，这些动态因素很难在静态模型中完全覆盖。

针对这一问题，RMA（Rapid Motor Adaptation）提出了一种有效的在线适应架构：

首先在仿真中利用多种环境参数（如摩擦系数、负载）训练基础策略；随后训练一个“适应模块”，通过近期的状态与动作历史序列，实时推断当前环境与动力学对应的隐变量。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGfO962Ee1erOfic3quLOzWAhYrCDU8y0nBRY1lgpgibjEgaDnV0G044sWaeU2BcZ9csdNvaLzvfGzIToSS1OpaFgjFibrEuMiabiczA/640?wx_fmt=png&from=appmsg#imgIndex=16)

图10 | RMA 先借助仿真中的环境参数训练基础策略，再让适应模块从近期交互历史中推断环境表征；实机部署时不需要直接测得摩擦系数或负载。©【深蓝具身智能】编译。

在实机部署时，机器人无需直接测量真实的摩擦系数或负载，适应模块能够根据运动反馈实时调整基础策略的输出，从而实现对未知环境的快速适应。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGduJHfrKktHmMeJSa92734JvQibacwgfb8zdSBIlHjccBHUIKiaLAz9rMBdDvnn22m6IK3DFAI728LZqwrG8HmyGjYj3gOeDQojs/640?wx_fmt=jpeg&from=appmsg#imgIndex=17)

▲图11 | RMA 在沙地、泥地、草地和障碍物等多种真实条件中测试四足机器人的快速适应能力，验证了基于近期交互历史估计环境状态的方法。©【深蓝具身智能】编译。

### 从稳定行走到复杂敏捷运动

随着技术发展，四足机器人的任务已从基础的平稳行走扩展到跨越沟壑、攀爬高台等复杂敏捷动作。在这一阶段，Sim2Real 面临的挑战不再局限于底层动力学参数的迁移。

- 在 Robot Parkour（机器人跑酷）研究中

系统首先分别训练攀爬、跨越、低姿穿越等单项专家技能，随后通过知识蒸馏，将这些技能整合到一个基于第一视角深度相机的统一视觉策略中。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/uwFbeBKoFGcfy1ibgUmKiaA7Ps0jzPh09EQXvg5ZLCqlZF0TRIiaqHKKAKMibtYZAcicCQRtygjY11icTR1bQhACuwiaLwBPeJcjAmdBBQL1XG5osE/640?wx_fmt=jpeg#imgIndex=18)

▲图12 | Robot Parkour 将多种障碍技能带到真实机器人上，迁移问题也由单纯的动力学差异扩展到了感知噪声与技能切换。©【深蓝具身智能】编译。

此时的 Sim2Real 迁移，需要额外处理真实的深度相机噪声、光照变化，以及在物理极限边缘的动态技能切换。

这显著提升了系统集成的难度，但也进一步推动了具身智能在复杂环境中的应用。

- 与 Robot Parkour 不同，ANYmal Parkour 采用了另一种架构

系统保留了行走、跳跃、攀爬和低姿穿越等独立的专门技能，由高层导航策略根据视觉感知结果，动态选择并衔接合适的底层技能，而非将所有动作融合为单一控制器。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGf8w2FWwNYlpE0O7Hk13OffbCyfR4Ij5lb0PBicmJUKXOiaw8KJK9gibOYtx1iab4icZrFMRUpxOyt1yHnzDXjfKMibGIibJZJzhf700k/640?wx_fmt=jpeg&from=appmsg#imgIndex=19)

▲图13 | ANYmal Parkour 展示了攀爬、跳跃和低姿穿越等真实技能。系统保留多种专门运动技能，再由高层导航策略根据感知结果选择并衔接合适的技能。©【深蓝具身智能】编译。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX7OGyOG8gzibCyLX91kbhEcWl0mnLk5Zb5uVRabIn51LEKNicYT8OlZZQ/640?wx_fmt=png&from=appmsg#imgIndex=20)

## 实机部署必须建立分层安全机制

在仿真中发生跌倒只需重置环境；但在真实部署中，跌倒可能导致昂贵的硬件损坏或引发安全事故。

因此，在实际工程中通常会部署多层安全防护体系，例如：

物理急停开关、驱动器电流限制、机械止挡、软件力矩限幅、软关节限位以及跌倒检测。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcKyxZ7sicuEIaa11BicBZBIRSAKVInCDicnBhMujjC2sjPvB05qeBvf8foxyla46aSlN7zeQ4WkaGsibUBzRYeIqCN6YJRTAPPqDk/640?wx_fmt=png&from=appmsg#imgIndex=21)

▲图14 | 实机部署需要将物理急停、驱动器保护、状态监测和策略侧约束组合为分层防御体系，避免依赖单一安全机制。©【深蓝具身智能】编译。

分层防护不依赖单一的安全机制。考虑到软件异常和瞬时电流过载等风险，系统需要在算法层、控制板层、驱动器层和机械结构层分别设置保护措施。

缺乏安全保障的策略，即使在仿真中表现良好，也不具备实机部署条件。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX5ne3MfNYQBbic4xIYsEJDKpCRqQXk6gllicSqc7QiabhaIEuCXA1I4xsg/640?wx_fmt=png&from=appmsg#imgIndex=22)

## Sim2Real 的工程落地，本质是一套误差分层处理的系统工程

回顾从仿真到现实的迁移过程，可以看出 Sim2Real 并非单一的算法，而是一套严密的系统工程：

> 系统辨识提供相对准确的物理基准；
>
> 观测与奖励设计保证训练目标符合实机约束；
>
> 域随机化与课程学习逐步提升策略的泛化范围；
>
> 底层前馈控制直接补偿可以明确建模的误差；
>
> 在线适应处理真实环境中持续变化的动态因素；
>
> 分层安全体系降低实机测试与部署风险。

因此在工程落地中，Sim2Real 不应只被视为训练后的一个迁移步骤，而应是从系统辨识到部署后持续校准的完整闭环链路。

---

**本文内容来源：**深蓝学院联合英国纽卡斯尔大学正教授、智身科技CTO-潘为，打造的《四足机器人：从动力学建模到强化学习》课程。

通用的四足强化学习技能：

- 训练时不挑硬件：CPU能跑，GPU能跑，Windows能跑，macOS能跑，Linux当然也能跑。
- 部署时不挑生态：训出来的策略，宇树能用，小米能用，智元能用，国外的也能用，不绑定任何一家硬件平台。
- 仿真到真机不翻车：碎石、斜坡、台阶，仿真里什么样，真机上就什么样。

⬇️阅读原文：理解四足机器人底层物理本质，再用RL控制它！

编辑｜阿豹

审编｜具身君

 ****推荐阅读**
[![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcibJS8986MfCcVATGOkcK6lNQfiaTORbuhSFoATTmZ5kA6nV8l8REia7nm4A4OxC1yOePBqrzWHQQd0ALicYANgOoNmRbibChjcAuQ/640?wx_fmt=png&from=appmsg#imgIndex=23)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=3824573915845640194&scene=126#wechat_redirect)[![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGcRfEtsGjVkl7cXB7QYAAib4wOMhdRcvsQicHnmiaxqoibw9LUCGGcPGSYnUPeUlZEoiaBlQezclFhp5yZQ6yLcLAjYeI67pJmvhOMw/640?wx_fmt=jpeg&from=appmsg#imgIndex=24)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=944555238&lang=zh_CN#wechat_redirect)

**![Image](https://mmbiz.qpic.cn/mmbiz_png/qKE443uRvLo6ic3ZPUttmFZ2AefQ4wjHSlQluSDkaxL9icWicpPYYmpo1Wa37Scjhh4AS5VwYJtmlTf5cKMiaIXg5g/640?&random=0.17349735674179656&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&wx_fmt=other#imgIndex=25)**

**【深蓝具身智能】****的原创内容均由作者团队倾注个人心血制作而成，希望各位遵守原创规则珍惜作者们的劳动成果；未经授权禁止任何机构或个人抓取本账号内容，进行洗稿/训练，否则侵权必究⚠️⚠️**


![Image](https://mmbiz.qpic.cn/mmbiz_png/Nabxc8rdYriaKqxCUjcZ8sSCnSNlWpqdI1kyXXQjXbtv95xvACqQoqL2ibbKXt9PB0FLPibKiawGsTcQrnKDGWVw2Q/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1#imgIndex=26)

点击❤收藏并推荐本文**
