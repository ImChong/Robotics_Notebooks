---
title: 2026年：斯坦福宋舒然团队14篇工作全盘点
author: 深蓝AI
date: "2026-08-18 17:32:00"
source: "https://mp.weixin.qq.com/s/vcewu3wKIcrsidzfGr2-yg"
---

# 2026年：斯坦福宋舒然团队14篇工作全盘点

**![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpBGZJuSuGB6mD5Eq39KVKic4rrPPcTwUSKGWrNic9BSspeptuNOD3Yw3rictMY9Y5dFUntzROFluWo2iaxiaDybF0Byo3WpcHrIiaiavc/640?wx_fmt=jpeg#imgIndex=0)「机器人真正“触达”物理世界」**

2026年，具身智能正从单一的"视觉-动作"映射，走向跨实体泛化与真实世界接触式操作。

斯坦福大学 Robotics and Embodied AI Lab（REALab）的宋舒然教授团队，今年交出了一份系统性答卷：

14 篇代表性工作，从基础模型的统一表征与扩散策略微调，到操作层的多感官融合、极简柔顺控制、数据采集接口与单次示范泛化，技术脉络清晰，一路向下扎根。

这些工作都在回答同一个问题：

当机器人物理形态千差万别、真实世界的物理接触又极其复杂时，该怎么打破硬件与数据的壁垒，让智能体在更广阔空间里自我进化？

本文将这 14 篇分为"机器人基础模型与策略微调""多模态感官融合与顺应控制""数据采集接口与跨具身操作"三个方向，带你一览 REALab 如何一步步拓展机器人精细操作的边界。

**欢迎关注【深蓝AI】**将持续分享人工智能领域前沿动态👇***深蓝AI*****1****—****机器人基础模型与策略微调**
**1. Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design**![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpDFm7L3HbFxO8kIrmppM4xpiaRh7iafHJxHmsrLE2TGxK7QggNtbicECZRXTZqvh5ibASkmCZreicaEA9OKib6ZqzaP5jTj59sstn0XA/640?wx_fmt=jpeg&from=appmsg#imgIndex=1)

图 1 | 该图展示了 Transformer Transformer 模型的整体架构与工作流程，包含针对实体（Embodiment）与动力学（Dynamics）的 RoboTokens 输入、位置编码、AdaLN 扩散步骤以及动作-机器人优化（Motion-to-Robot Optimization）与跨实体控制策略（Cross-Embodiment Control Policy）两个核心应用分支©【深蓝 AI】编译

传统的机器人设计与控制策略多采用解耦方式，难以实现面向多样化任务与运动轨迹的机器人形态与控制的联合协同设计。

本文引入了 Transformer Transformer，这是一种基于 RoboTokens 训练的扩散模型。它实现了对机器人实体、状态和动作的统一标记化，并通过动力学自我引导，将值预测用于引导实体扩散过程，从而生成高价值的机器人设计方案。

在实际测试中，该模型实现了对未见奖励和轨迹的零样本优化，优化后的 ALOHA 机械臂轨迹追踪误差降低了 70% 以上。


**2. DF-ExpEnse: Diffusion Filtered Exploration for Sample Efficient Finetuning（ICML 26）**![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpBeTBpOUJvpOnlxn4h1cpz3YoQvsFM3R1pR8adbloggT1zpoTCmyuoDTibxuibOKWhNjKliaTovWsNCsQtPn1YJ2SSSYG7zuNPcMw/640?wx_fmt=png&from=appmsg#imgIndex=2)

图 2 | 该图展示了 DF-ExpEnse 框架的整体架构与核心流程，通过动作空间滤波、样本候选生成、以及结合批评家集成与价值估计的不确定性导向探索，实现高效的样本微调©【深蓝 AI】编译

利用预训练生成式控制策略进行在线微调时，盲目的动作扰动往往难以高效触及高质量状态空间，导致微调过程耗时且样本利用率不足。

DF-ExpEnse 巧妙利用了生成式控制策略的多模态建模能力构建候选动作集，并引入批评家集成（ensemble of critics），精准识别在“动作质量”与“高探索兴趣”之间实现最佳平衡的动作。

与默认微调方案相比，DF-ExpEnse 能够显著且持续地提升样本效率，大幅减少了在线交互样本量。


**3. From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning**![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpDy8iazmAqJu3wdmPnr9qeMDCfnPYv3Ousnku9Nw09o9ZoAwvjuBmS8B7oiccyUW6PTh4w7jzWia37ly0X0Zjy7PP22T8Ajwkw89g/640?wx_fmt=png&from=appmsg#imgIndex=3)

图 3 | 该图展示了分布收缩强化学习（DICE-RL）方法的框架流程，通过将预训练的生成式行为克隆策略精炼为“专业”策略，在成功动作模式周围收缩动作分布，并利用残余离线强化学习实现高效且稳定的微调。©【深蓝 AI】编译

传统的行为克隆难以在长周期复杂任务中达到极高成功率，而直接应用强化学习微调又面临探索效率低下和灾难性遗忘的问题。

DICE-RL 框架将强化学习视为一种“分布收缩”算子。它在预训练策略的基础上，结合选择性行为正则化与价值引导的动作选择，在抑制策略漂移的同时，通过在线反馈显著放大高成功率的行为模式。

在复杂长周期操作技能评估中，DICE-RL 展现出极高的样本效率和收敛速度。但若初始预训练策略未能涵盖必要的探索子空间，强化学习阶段仍可能陷入局部最优。


**4. Are Foundation Models the Route to Full-Stack Transfer in Robotics?**![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpBSnEZrLXOFKI2ynToeptRzBlzWn9TOeiakcUKwJIrZq3OichK0BkWPes78I43MwxPibsERDPgib7w377EPCDShfXUAGGrynVDO1ico/640?wx_fmt=png&from=appmsg#imgIndex=4)

图 4 | 该图展示了三种不同类别视觉-语言-动作模型（VLA）的典型实现架构与推理流程，包括基于直接映射的 OpenVLA、基于 FAST 的 π0-FAST 以及基于去噪流匹配的 π0 模型©【深蓝 AI】编译

机器人的技能迁移长期局限于特定的抽象层级，难以兼顾高层语义理解与底层精细控制，跨形态泛化一直是核心挑战。

本文系统剖析了基础模型与 Transformer 架构对迁移学习的影响，指出大模型通过统一的表征空间，打破了语言理解与电机控制的割裂，使机器人向“全栈式迁移”迈出了关键一步。

基础模型将毫无疑问地成为推动机器人全栈式迁移的核心路线。然而，当前模型缺乏深层物理动力学建模，且高度依赖海量交互数据，在面对高动态物理环境时其实际效果仍受显著制约。


**5. Gated Memory Policy: In-Context Memorization and Adaptation**![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpAmXwJZcpO5xtfMJsosVRVwMIrTV9R3rs3aKM53JbuHzU9bArHyxB0NiaDr11OLvebiaCnDEVF6vwTyros2fYwmPFV6D0uO1HqE0/640?wx_fmt=jpeg&from=appmsg#imgIndex=5)

图 5 | 该图展示了Gated Memory Policy (GMP) 网络的架构与门控注意力模块设计，包含二进制内存门控、带噪历史动作以及推理时的历史 Token 缓存机制。©【深蓝 AI】编译

现有的视觉运动策略在盲目延长观测历史时，往往由于分布偏移和过拟合导致性能显著下降，机器人难以自适应地决定“何时需要记忆”。

Gated Memory Policy (GMP) 引入了学习型的内存门控机制，精准判断任务阶段并仅在必要时激活历史上下文。同时，通过在动作中注入扩散噪声，提升了策略在长序列交互中的抗干扰能力。

在非马尔可夫操作测试中，GMP 相较于长历史基线方法平均成功率提升了 30.1%。

***深蓝AI*****2****—****多模态感官融合与顺应控制**
**6. Multisensory Continual Learning: Adapting Pretrained Visuomotor Policies to Force**![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpC4qBnjeoiac1G85eULtScfSBoibbJNjgzAG4KYueEqjNYzNPsaOgWVF9uD8B3DXAZRFXd7ibB7T5kLyTIzkHaGx6iaicylGEXG2Pls/640?wx_fmt=png&from=appmsg#imgIndex=6)

图 6 | 该图展示了MuSe多感官持续学习架构的框架流程，通过模态专用编码器对图像、本体感觉、文本和力/力矩（F/T）历史进行编码，结合联合序列模型实现动作预测、未来视频帧生成以及自适应顺应控制的虚拟目标设定©【深蓝 AI】编译

在接触密集型任务中仅靠视觉很难捕捉细微的交互状态，但多感官数据稀缺，为每种传感器组合从头预训练专属策略几乎不可能。

MuSe 架构通过多阶段融合、多感官未来预测以及经验回放，将有限的多感官数据（如力/力矩信号）有机整合到预训练的纯视觉策略中，避免了对原有视觉能力的灾难性遗忘。

在真实的机器人微调任务中，MuSe 表现出极强的力控适应能力，甚至反哺了原有的视觉预训练性能。但该框架目前主要依赖特定硬件配置的传感数据，面对传感器安装位置变化时仍需额外校准。


**7. Minimalist Compliance Control（RSS 26）**![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpC5yZz1xyjVe8cTWTbXdkiaENY7h5PtC5mxTDyNXrQ1Z4W8FBAqhEEiccDFFcCMQefYmEg65H7cpZPJ7bb1AXicP3jYSKfIKfUWWc/640?wx_fmt=png&from=appmsg#imgIndex=7)

图 7 | 该图展示了Minimalist Compliance Control（极简柔顺控制）的核心框架与应用概览：(A) 通过电机电流或电压变化及雅可比矩阵估计外力，无需力传感器或学习模型；(B) 具备跨实体（Embodiment-Agnostic）通用性，适用于ARX机械臂、G1人形机器人及LEAP灵巧手；(C) 可作为即插即用模块无缝集成至VLM策略、模仿学习策略及基于模型的策略中，实现白板绘画、煎蛋放置、球体旋转等精细物理交互任务。©【深蓝 AI】编译

柔顺控制是机器人实现安全物理交互的核心，但长期以来严重受限于对昂贵力/力矩传感器的依赖，基于强化学习的免传感器方案又面临显著的 Sim-to-Real 鸿沟。

极简柔顺控制（Minimalist Compliance Control）仅利用现代电机现成的电流或电压信号与雅可比矩阵估计外部作用力，无需力传感器或学习过程，即可直接引入任务空间导纳控制器中。

该方法具备跨实体通用性，在机械臂、人形机器人等平台上均实现了鲁棒的柔顺交互。


**8. In-the-Wild Compliant Manipulation with UMI-FT**![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpACcWpWVbBxktGh0mAHpA587kzj4upBt2pv1RRBUkNvKmDhkvLLGoJJ8eoD9xQFISFOwHYCP6RibCBd7IbVmphpwSMicFRf1oX5k/640?wx_fmt=png&from=appmsg#imgIndex=8)

图 8 | 该图展示了 UMI-FT 的自适应顺应策略架构（Adaptive Compliance Policy Structure），通过融合 RGB、深度图像、左右手指 CoinFT 触觉传感器（或六维力/力矩传感器）以及本体感觉输入，利用特征提取与自注意力机制融合，最终生成动作控制指令。©【深蓝 AI】编译

精确的力调制在操作任务中至关重要，但商业六维力传感器成本高、体积大，严重限制了具备力感知能力的机器人策略在野外环境的规模化采集。

UMI-FT 是一个专为野外环境设计的手持式数据采集平台，将紧凑的力/力矩传感器直接安装在手指上。基于该平台采集的多模态数据，研究团队训练了一种自适应顺应策略，精准预测位置目标与抓取力。

在白板擦拭、灯泡插拔等任务中，UMI-FT 赋能的策略显著优于传统纯视觉基线。

***深蓝AI*****3****—****数据采集接口与跨具身操作**
**9. ModPack: An Extensible Teleoperation Interface for Bimanual Mobile Manipulation**![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpBhdweeaibPNLhMypv62dyoTAV5zUvOh9QQPcToEvA3XGAZicCt5NuW5xicrkoxJpLYTXQiaCNxRPzSlP0zqH4JEhd07rnCTqjbpeM/640?wx_fmt=jpeg&from=appmsg#imgIndex=9)

图 9 | 该图展示了ModPack可扩展双臂移动操作遥操作接口的核心架构与模块组成，包括支持插拔模块的背包核心、主动感知模块（如Apple Vision Pro、iPhone）以及6-DoF与7-DoF主手机械臂配置©【深蓝 AI】编译

现有的机器人遥操作系统大多高度定制于特定的硬件平台，在面对多样化的机器人构型时，需要重新搭建通信与计算架构，研发成本极高。

ModPack 提出了一个模块化、可扩展的遥操作框架。其核心是一个自包含的可穿戴“背包”单元，集成板载计算与电源，并支持即插即用的主动感知模块（如 Apple Vision Pro）与各型主手机械臂。

该系统在多种机器人平台上显著降低了异构系统适配的复杂度。不过，作为便携式设备，背包的重量在长时间连续作业中可能会对操作员造成体力负担。


**10. Behavior Prompting Policy: Demonstrations as Prompts for Manipulation**![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpC1v3U9FACjBSqqImUAkCictD8WH8jVH4RbvPgYQ9wlxAmqq3e7tT9Wh53QtbvyGaJQve48zdllv41jSmmPmqgQ7cly1ZgqIvgc/640?wx_fmt=jpeg&from=appmsg#imgIndex=10)

图 10 | 该图展示了行为提示策略（Behavior Prompting Policy, BPP）的整体架构与流程，包含注意力池化模块将观察与本体感觉编码为提示嵌入，并通过基于Transformer的扩散模型生成执行动作。©【深蓝 AI】编译

传统的机器人策略依赖特定任务的重新训练或在线微调，难以仅凭单次人类示范在测试时灵活适应并执行开放世界中的新指令。

行为提示策略（BPP）借鉴了上下文学习思想，将人类单次行为示范作为提示，与当前的机器人观察无缝对齐，并输入至基于 Transformer 的扩散模型中，直接生成连贯的执行动作。

借助于 iPhUMI 接口采集的多样化数据，BPP 能够让机器人仅凭单次示范成功泛化至未知任务。


**11. HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations（RSS 26）**![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpDIHibZSObFnrElQ96jB4wI1258p9I2FjJEBKorD7yEQm1C3iahTnnFbWWpviazQ5niaom3SiaOboW1jXicuM2Yg9bGvhMdACdlRfdPc/640?wx_fmt=png&from=appmsg#imgIndex=11)

图 11 | 该图展示了HoMMI系统的总体框架，包含HoMMI数据采集、跨具身手眼策略（采用具身无关的视觉表征与放松的头部动作表征及扩散Transformer）以及机器人部署与全身控制器流程©【深蓝 AI】编译

全向移动操作的数据采集受限于特定硬件，而利用无机器人的第一人称人类示范又会显著加剧观测与动作空间中的人机具身差异。

HoMMI 框架通过专门的跨具身手眼策略显式弥合差异：采用具身无关的视觉表征消除视角差异，设计放松的头部动作表征适应运动学不匹配，并引入基于扩散 Transformer 的全身控制器协调运动。

该框架采集的无机器人示范数据有效训练出了具备强大泛化能力的鲁棒策略。


**12. One Demo Is Worth a Thousand Trajectories: Action-View Augmentation for Visuomotor Policies**![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAU43OWAJMrqvv5o1gic1FQD1alkOHJY32G3y5H2G4pgsXR7wg0XIxnx7cdcPdpXQPG3CcwErUesdCJyTmNTZXGsd8Mquu71P5U/640?wx_fmt=png&from=appmsg#imgIndex=12)

图 12 | 该图展示了Action-View Augmentation方法的整体概览，通过单次扫描与演示重建3D场景点云及Fisheye 3DGS，进而生成包含新视角与新障碍物的1000条增强训练轨迹与多视角鱼眼图像©【深蓝 AI】编译

机器人初始配置的细微改动以及未曾见过的障碍物，极易导致视觉运动控制策略面临超出分布的观测结果，从而引发执行失败。

该框架通过单次真实世界手眼演示，结合适配广角鱼眼相机的 Gaussian Splatting 重建 3D 场景，并利用轨迹优化技术生成包含新视角与新障碍物的 1000 条增强训练轨迹与多视角图像。

这种“Action-View Augmentation”显著提升了各项操作任务的成功率。


**13. Geometry-Aware 4D Video Generation for Robot Maniplation（ICLR 26）**![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpCwV5xCMwwSlNhicQkapd8HsBIXqeKPNwax5Ktzy4TN7IdfTN3cpp1HX3uYV924U9QgRg3Aq2aY0H4FnJibjDcJ3df7k4pm3z1ug/640?wx_fmt=jpeg&from=appmsg#imgIndex=13)

图 13 | 该图展示了几何感知 4D 视频生成（Geometry-aware 4D Video Generation）的整体框架，模型输入来自两个相机视角的 RGB-D 观测，并预测参考视角下的未来 4D 点图，实现几何一致的 4D 视频生成©【深蓝 AI】编译

现有的视频生成模型在建模动态场景时，要在多个相机视角之间保持生成视频的时间连贯性与几何一致性，仍然是一个重大挑战。

该模型在训练中引入跨视角点图对齐监督，强制生成视频满足多视角 3D 几何一致性。在无需相机位姿输入的情况下，模型能从新颖视点生成空间与时间上严格对齐的未来视频序列。

预测出的 4D 视频可结合位姿追踪器恢复末端执行器轨迹，进而训练出能泛化至新视点的策略。


**14. DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation（ICML 26）**![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpB4lxkbUoiaPVKleNrHDGK8ENve8R1gRhdMibqmUABzHiahqwDb0ibYs5RTc4LbOsbtI3HvUDSdQ9C62IOAt1IuOXs7auVC3I32VI8/640?wx_fmt=jpeg&from=appmsg#imgIndex=14)

图 14 | 该图展示了 DexMachina 的方法框架，包括从手物示范出发提取任务奖励、运动奖励与接触奖励，以及通过虚拟对象控制器（VOC）实现的自动课程学习流程©【深蓝 AI】编译

学习灵巧操作策略以追踪人类手物示范是一项极具挑战的任务，其难点在于庞大的动作空间、时空不连续性以及人机手部形态的具身鸿沟。

DexMachina 提出了一种基于课程学习的算法，其核心是使用强度衰减的虚拟对象控制器（VOC）。物体首先被自动驱动至目标状态，随后策略在运动和接触的引导下逐渐学会接管控制权。

在包含多样化任务和灵巧手的仿真基准测试中，DexMachina 显著优于基线方法。不过，该方法目前主要在仿真环境中验证，虚拟对象控制器的平滑过渡在真实物理世界中的鲁棒性仍有待检验。

***深蓝AI*****4****—****总结**

从上述 14 篇工作可以看出，REALab 在 2026 年的研究重点已经全面走向了多模态融合、跨具身泛化与物理顺应性的深水区。

无论是通过 Transformer Transformer 实现形态与动作的协同设计，还是借助 ModPack 和 UMI-FT 降低真实世界数据的采集门槛，团队都在试图解决机器人走向非结构化环境时的核心痛点。

这些工作表明，未来的具身智能不仅需要更聪明的“大脑”（如统一的基础模型），还需要更具适应性的“小脑”（如极简柔顺控制）和更灵活的数据获取方式。这为整个机器人学习领域提供了极具参考价值的技术路线。

编辑｜阿豹

审核｜阿蓝

**往期推荐** Recommend [![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpAhYq85CXTeKEXodjfiaIUHfDfa8hBib0502WDIBrslJKic68cZC5IiaicIGdXxcCzrBWkfDnacLHu51TWqpBIg0BezPN8zyV3CHEJc/640?wx_fmt=jpeg&from=appmsg#imgIndex=0)](https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247602525&idx=1&sn=179072d10ad35c9c441927d095c3e381&scene=21#wechat_redirect)**近五年谁在 Science Robotics 上发文最多？盘点全球顶尖机器人实验室**[![图片](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAKPXr0kicZddyXdPOg1Jm7tKusPLcWicG0ALpMqpjSHZxxsu45C13rzA4XZ2leKiaxG64fPqc9zIRIj8CR43YYibVy9ic8aRib3LUd8/640?wx_fmt=png&from=appmsg#imgIndex=0)](https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247602190&idx=1&sn=a9e9a29449a395f8c08f54f4c78fed06&scene=21#wechat_redirect)**3D目标检测经典算法全盘点：单目、双目、激光雷达****欢迎关注【深蓝AI】**持续分享人工智能领域前沿动态👇![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/943LxrS8cpCFreRWsn2fgjfEz7fB26oBpbfOsHK7zRA7xsBRS9mpSIvgQwOETOeicmb4PgKiby0nOGDo9ObI0JrvBflh4oibEdgwTEykKOSQ1w/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=16)
