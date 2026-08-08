---
title: 人形机器人运动控制：强化学习与策略训练体系详解
author: 深蓝具身智能
date: "2026-08-08 10:56:00"
source: "https://mp.weixin.qq.com/s/mxesB0pGI_NLSkSf-cZYug"
---

# 人形机器人运动控制：强化学习与策略训练体系详解

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/kaugqJpv9nuCktylvYoMKHYNAVojoRUpfyf1py08JvUnkfPXArzj4t5bMiaS6RBCXHHGhf8xlyw8icHrJcjEyYoA/640?wx_fmt=gif&from=appmsg#imgIndex=0)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGdia7DC9nVspMMaaFYhXoWmEB8wQrvID5cko9ewU9gkD1ngHMBliatPIT7yXxFW9ejUw2JoyeolYeT8V2uUfhu63z6oG5nGkIwPY/640?wx_fmt=png&from=appmsg#imgIndex=1)

人形机器人运动控制没有一劳永逸的解析公式，而是依靠策略不断迭代逼近

——强化学习策略训练体系

人形机器人的智能运动能力生成，可以大致分为：传统动力学建模控制、数据驱动强化学习控制，这两条技术路径。

- 传统控制依赖人工建模、轨迹预设和参数调试，适配复杂非结构化场景的能力存在明显上限。

这一问题放在数学层面解释的话，是因为绝大多数数学问题都没有解析解（真实物理系统无法被公式 100% 精确描述），只有数值解，而且是近似数值解（通过迭代计算不断逼近真实结果）。

> 机器人运动伴随非线性、接触冲击等复杂物理效应，不存在完美闭式公式，精细建模也只是理想化简化，仅能逼近真实结果。

- 强化学习策略训练体系就比较适合求解“近似数值解”

该体系由强化学习基础框架、Actor-Critic网络架构、PPO算法、奖励函数机制、Teacher-Student蒸馏技术五个部分组成，各部分功能独立、但彼此存在互补，共同完成机器人运动策略的训练、优化、轻量化与真机部署。

本文将从各部分的核心概念、应用实例、在人形机器人运动控制中的核心实现过程等方面进行详细梳理。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4YWHX9iazKkdkgh363zh9GFAfZia4RWWoYhutUeS8g43MnicLMfe9kUAZg/640?wx_fmt=jpeg&from=appmsg#imgIndex=2)

强化学习（RL）基础框架

- ## **核心概念**

## **强化学习是一套机器自主迭代学习的交互框架。**

## **机器人作为独立智能体，持续采集自身运动状态、外部环境状态，根据当前状态输出对应的控制动作，执行动作后接收环境反馈的单一标量奖励值，依据奖励数值的高低，持续修正自身的动作选择逻辑，反复迭代直至形成稳定、合规的运动策略。**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGfO5oiavqGI66vMfEsk4v4WiapZrnmWnEcWjuED46ibRSabSD8pkg9wwcDaemd70hGY1fvBaf7tnN8x50pnC8Y6TtqHic6ic7ibEibBBo/640?wx_fmt=png&from=appmsg#imgIndex=3)

▲图| 玩红白机游戏是强化学习最早的试炼场©【深蓝具身智能】编译

整个过程无需人工预设完整运动轨迹，仅通过状态、动作、反馈的循环交互，完成运动技能的自主习得。

- 举个例子

比如，车辆新手训练过程中，驾驶员不依赖预设行车路线，反复尝试油门开度、方向盘角度、刹车时机等操作。

每一次操作后，根据车辆是否平稳、是否偏离车道、是否减速到位等实际结果判断操作优劣，不断调整操作习惯，经过多次试错后形成稳定、正确的驾驶操作逻辑。

- 在人形机器人运动控制中，该框架是所有数据驱动运动训练的基础载体

传统模型控制需要工程师针对行走、蹲起、平衡恢复等每一类动作，单独完成动力学建模、轨迹规划与控制器参数调试，工作量大且场景泛化性差。

强化学习框架仅需定义机器人状态空间、动作空间、环境约束与奖励规则，即可让机器人自主学习平地行走、崎岖路面适配、外力扰动恢复、全身肢体协同等各类运动技能。

所有后续网络架构、优化算法、模型优化技术，均需要依托该交互框架运行。

- 核心过程

强化学习的核心数学建模为马尔可夫决策过程，通过五元组完成标准化定义，五元组表达式为。

其中  为机器人全局状态集合，包含各关节旋转角度、关节运动速度、IMU采集的躯干姿态与角速度、足部接触状态、机器人质心位置与速度等所有可观测运动参数；

  为机器人动作集合，对应控制器下发的关节力矩指令或目标关节位置指令；

  为环境状态转移概率，表征机器人在当前状态执行指定动作后，切换至下一状态的概率分布；

  为单步即时奖励，是环境对单次动作的量化评价指标；

  为折扣因子，取值区间固定在0到1之间，用于降低远期奖励对当前动作决策的影响权重，保证策略训练的稳定性。

框架的核心优化目标，是最大化智能体长期累积折扣回报的数学期望，具体公式为：

公式中 代表机器人运动策略函数，核心功能为输入当前机器人全局状态，输出对应动作的概率分布，是机器人动作选择的核心依据。

强化学习作为整套策略训练体系的顶层基础框架，具备支撑性。演员评论家网络架构、近端策略优化算法、奖励函数、师生蒸馏技术，通常无法独立完成运动策略训练，必须嵌入强化学习的状态交互、动作执行、奖励反馈、迭代更新循环中运行。该框架定义了整体训练流程的运行逻辑，其余所有模块均为框架内的功能细分组件。![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibqgVaodXH45G6Pdbk9xSEsUtlicqgxKkAiaK0P8QzGwuLiatibYiaIagQoOg/640?wx_fmt=jpeg&from=appmsg#imgIndex=4)

## Actor-Critic（演员-评论家）网络架构

## **该架构是深度强化学习的标准化网络结构，由两个功能完全独立、数据相互联动的神经网络组成。**

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGfJbS8vueBIjpbibNN7bibB3HYSmPsVvrx4vj5dCjH0KGt5yj6CqM3oHIibwmZV8f3HljoTliaB7G3mtD6gZT1ibm3A83DNVEjUV6Rw/640?wx_fmt=png&from=appmsg#imgIndex=5)

▲图| Actor-Critic（演员-评论家）网络架构©【深蓝具身智能】编译

- 网络架构

Actor网络负责执行决策功能，接收机器人实时状态，直接输出对应的控制动作；

Critic网络负责价值评估功能，接收相同的机器人状态，计算当前状态下机器人可获得的长期累积回报数值，为Actor网络的参数更新提供量化依据。

两个网络分工合作，完成决策与评估的闭环。

- 举个例子

比如场地竞速场景中，参赛车手负责完成方向盘调整、油门控制、刹车控制等实操动作，对应Actor网络的决策功能；

专属裁判实时观测车手的行驶路线、车速控制、车身姿态，判断当前操作是否有利于完成比赛，给出量化评价并指导车手调整操作，对应Critic网络的评估功能。

- 在人形机器人运动控制中，该架构实现了运动决策与价值评估的功能拆分，解决了单一网络无法同时完成动作输出与策略优化的问题。

Actor网络直接对接机器人底层执行器，输出的关节力矩、位置指令可直接驱动机器人完成运动，保障控制指令的实时输出。

Critic网络通过拟合状态价值函数，定量地判断每一个状态、每一组动作的长期收益，量化区分优质动作与劣质动作，为网络参数迭代提供梯度支撑，让机器人的运动优化具备明确的迭代方向。

目前所有人形机器人强化学习运动策略，大多基于该架构搭建。

- 核心过程

该架构的计算核心为优势函数，标准定义公式为。

公式中  为动作价值函数，表征机器人在当前状态下执行指定动作后，能够获得的整体长期回报；

  为状态价值函数，表征当前状态下所有可选动作的平均长期回报。

- 优势函数计算结果为正值，代表当前执行动作优于平均动作水平，网络需要保留并强化该动作逻辑；
- 计算结果为负值，代表当前动作存在缺陷，网络需要修正动作参数。

Actor-Critic网络是强化学习框架的核心载体，是近端策略优化算法（PPO）目前主要的运行基础（包含它的各种变种）。

PPO算法的参数更新规则，针对Actor网络参数设计，且需要依托Critic网络输出的优势函数完成计算。

奖励函数输出的单步即时奖励，会持续输入Critic网络，用于精确拟合状态价值函数，直接决定优势函数的计算精度。同时，AMP等主流机器人模仿学习算法，底层网络结构大多沿用Actor-Critic架构，该模块的稳定性一定程度上决定了整体训练效果。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibTvia4qLjYyoM2Do58jX9J71HickLLA3NxCQp6fPljkgY26WIeaoeeYVQ/640?wx_fmt=jpeg&from=appmsg#imgIndex=6)

## PPO（近端策略优化）算法

## **PPO是强化学习体系中专用的策略更新算法，核心作用是约束神经网络单次参数迭代的更新幅度。**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGc2AkwHy6ibY8e5ia9qbI4gFQ6ypHe9pCic7NZYYPnLVlJqPUyIPrlbReQQSQM3Qh2L3WDQOT56b6GHnPYTpkrSEWUWy29VDmvbq0/640?wx_fmt=png&from=appmsg#imgIndex=7)

▲图| PPO示意图©【深蓝具身智能】编译

- 核心概念

基础策略梯度算法不存在更新边界约束，网络参数单次迭代变化量不可控，容易导致已经训练收敛的稳定运动策略突然失效，直观说就是“学了新的忘了旧的，邯郸学步”。

PPO通过设置固定裁剪阈值，限制新旧策略的输出分布差异，保证每一次参数更新均为小幅优化，不会出现策略突变问题。

人的学习也是这样，颠覆性的推翻过去习得的知识技能是很少见的。往往是累计增量修改来学习。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGd8kBfUbjxRgPeVgER1az5pYBfCL3YO5EjILwv6NSljm7l6tY41Ko4jG7KGgdAYZicrJa2prXe13rzhtJy4ZQwz0MhOvoeygMRU/640?wx_fmt=png&from=appmsg#imgIndex=8)

▲图| PPO的精髓其实类似于ReLU激活函数，直接消减到超过限制的波动©【深蓝具身智能】编译

- 人形机器人运动策略训练的主流优化方案

人形机器人的运动维度高、自由度多、动力学约束复杂，微小的参数波动就会导致步态失衡、机身倾倒。

PPO的约束机制可以保障高维运动策略的训练稳定性，让机器人在迭代过程中持续优化步态平滑度、平衡稳定性和环境适配性，不会出现前期习得的运动能力在后期迭代中丢失的问题。

一般主流人形机器人步态训练、全身运动模仿学习，会采用PPO作为核心迭代算法。

- 核心过程

PPO的核心为裁剪目标函数，完整计算公式为：

公式中 为新旧策略概率比值，用于量化两次迭代的策略差异；

   为当前时刻的优势函数，由Critic网络计算得出；

  为人工预设的裁剪阈值，常规取值为0.2，用于限定策略更新的最大幅度。

该公式通过双重取值约束，保留优质策略的优化空间，抑制劣质突变更新，保障训练稳态。

- 与其他模块的相互关系：

PPO算法运行于Actor-Critic网络架构之上，仅针对Actor策略网络进行参数更新，不直接作用于Critic网络。算法迭代所需的优势函数，由Critic网络基于奖励函数输出的回报值计算得到。

PPO是强化学习框架内部的核心优化规则，决定整个训练循环的迭代方式。同时，AMP模仿学习算法的训练迭代逻辑，基于PPO算法搭建，是动作能力泛化训练的核心支撑。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4yZibwaOWpB3DrcxuiafpXicx2ibHiaHAZFr7ptU6ud2hsxgCXvV0JGHtTDw/640?wx_fmt=jpeg&from=appmsg#imgIndex=9)

## 奖励函数（Reward Function）

- ## **核心概念**

## **奖励函数是一套固定的量化计算规则，是机器人运动行为的评价标准。机器人每完成一次动作、完成一轮状态迭代后，奖励函数会根据预设规则计算出一个标量数值，正向数值代表动作符合预期、具备优化价值，负向数值代表动作存在缺陷、需要修正。**

## **所有网络参数迭代、策略优化，以该数值作为核心依据。**

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGcq42c93Pnh2K1Z0iaYv9iaZjSbJicQ8I8YsgytBb9jicbE0ibTvZGLEL5u2icFm5OicGYicRkQTIRemp9b5amAtibqfjjfeH90YB5LHcmM/640?wx_fmt=jpeg#imgIndex=10)

▲图| 奖励函数示意图©【深蓝具身智能】编译

奖励函数直接定义机器人的运动优化目标，决定最终习得的运动形态。工程师通过配置不同的奖励权重与约束规则，引导机器人完成指定运动任务。

通过正向奖励激励机器人完成前进、转向、姿态保持等目标行为，通过负向奖励惩罚机身倾斜、关节超限、剧烈冲击、原地抖动、摔倒等无效、危险行为。

奖励函数不存在通用模板，每一种运动任务、每一类机器人机型，都需要独立调试权重参数，参数配置偏差会直接导致训练出畸形步态、无效运动策略。

- 核心过程

工程落地中，人形机器人运动训练的奖励函数均采用多维度加权求和形式，具体公式为：

公式中 为任务奖励项，用于约束机器人完成核心任务，包含前进速度误差、目标位置偏差、转向角度精度等参数；

  为平衡奖励项，约束机器人躯干倾角、质心偏移量、支撑状态稳定性；

  为平滑奖励项，抑制关节指令突变、肢体剧烈抖动，保证运动连贯性；

  为惩罚项，对关节超限位、机身倾倒、足部剧烈碰撞等危险行为输出负奖励；

 、、、 为各维度权重系数，由工程师根据训练需求人工调试。

奖励函数是整个强化学习训练体系的评价核心，是Critic网络拟合价值函数的数据来源。

奖励函数输出的单步回报值，决定了优势函数的计算精度，进而影响PPO算法的参数更新效果。若奖励函数设计不合理，Actor-Critic网络无法拟合价值模型，PPO迭代会出现无效优化，最终导致整个强化学习训练流程失效，无法产出可用的运动策略。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4hKdH0P2rRHX1TlxUqlAx7X6m2hcl7XttttyRW05mhbEa1msX7zEzvw/640?wx_fmt=jpeg&from=appmsg#imgIndex=11)

## 师生知识蒸馏（Teacher-Student）

- ## **核心概念**

## **师生蒸馏是模型轻量化的后置处理技术，分为教师网络与学生网络两个主体。**

## **教师网络结构复杂、参数量大，在仿真环境中完成完整训练，具备成熟、稳定、全面的运动能力。**

## **学生网络结构精简、参数量小、推理速度快，不参与原始训练，仅通过学习拟合教师网络的动作输出规律，复刻教师网络的运动策略。**

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGd2yEUGCqHMnwber8BDfVQia5xrDz8DgbQsgLhJtqgOtibZHxeuL4rTgpFd69gxMA5JWYDKiaibkH83diaVaIEeR9n0TNxicsFQCqmaQ/640?wx_fmt=jpeg#imgIndex=12)

▲图| 师生蒸馏的形象化描述©【深蓝具身智能】编译

- 举个例子

资深操作人员具备成熟、全面的实操经验，可完成各类复杂操作，对应教师网络。

新手人员通过全程模仿资深人员的操作动作、操作逻辑，快速掌握核心技能，无需从零试错学习，对应学生网络。

最终新手人员以更低的学习成本、更简洁的操作方式完成同等工作。

- 该技术解决了仿真训练大模型无法真机部署的工程痛点

仿真环境中训练的强化学习策略，为保障运动精度和泛化能力，网络参数量大、推理延迟高、算力消耗高。

人形机器人机载控制器算力有限，无法承载大模型的实时推理运算。师生蒸馏通过参数拟合与知识迁移，将大模型的运动能力迁移至轻量化小模型，在基本保留原有运动精度、稳定性、自适应能力的前提下，降低模型推理时延、减少算力消耗，适配真机硬件运行条件，是强化学习策略从仿真训练走向真机落地的必经步骤。

- 核心过程

师生蒸馏的核心训练损失函数为最小化师生网络输出差异，公式为：

公式中  为教师网络的动作输出分布；

  为学生网络的动作输出分布，损失函数通过最小化两者差值，实现运动策略的精准迁移。

蒸馏训练过程中，会同步叠加任务奖励约束，保证学生网络不仅复刻教师网络的输出，同时满足机器人运动任务的各项指标要求。

师生蒸馏是强化学习训练流程的后置独立模块，不参与前期迭代训练。

蒸馏的对象，是经过强化学习完整框架、Actor-Critic网络、PPO算法、奖励函数优化后收敛完成的成熟策略模型。

该模块不改变原有训练逻辑，仅完成模型轻量化迁移，是强化学习策略实现Sim2Real落地的核心配套技术。

- **实践项目展示 | 基于教师-学生蒸馏的全身运动跟踪**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/uwFbeBKoFGeNdQnnB5mndzVPnK5muTibQebdy1OzW8SsoAMCSumokcfmEicdTXkdKGtUQfSBRhJic4OvLYsv3AqSb0NrJYAMASsLJianOUsGLQg/640?wx_fmt=gif&from=appmsg#imgIndex=13)

▲图源| 深蓝学院《人形机器人运动控制》课程

《人形机器人运动控制》课程：部分实践项目展示

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGfmOMPNfNb9DX0gaHhx3EjyUPlu5ZN0Ok4Lc1U8bh5BDib4swzibm1YOcINao20St4PfV4OIicacvUXMB4x5p62Tcx4PsWUBTWlBI/640?wx_fmt=png&from=appmsg#imgIndex=14)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX7OGyOG8gzibCyLX91kbhEcWl0mnLk5Zb5uVRabIn51LEKNicYT8OlZZQ/640?wx_fmt=png&from=appmsg#imgIndex=15)

## 人形机器人强化学习策略训练的技术闭环

## **人形机器人强化学习策略训练体系的五个核心模块，**互相耦合形成了一套完整的“训练-优化-落地”技术闭环，各模块功能互补，不存在功能重叠，也无法单独脱离体系独立实现运动策略生成。

强化学习基础框架是整个体系的顶层运行逻辑，搭建了智能体与环境的交互迭代循环，为所有后续模块提供运行载体；

Actor-Critic网络架构是体系的硬件载体，实现决策与评估的功能拆分，是策略迭代的网络基础；

奖励函数是体系的评价标准，定义所有运动优化的目标与方向，决定机器人最终的运动能力形态；

PPO算法是体系的优化规则，在保障训练稳定性的前提下，完成网络参数的精准迭代更新。

Actor-Critic、奖励函数、PPO三者协同，完成仿真环境内机器人运动策略的从零训练、迭代收敛与性能优化。

师生知识蒸馏独立于前端训练流程，属于体系的后置工程落地模块。在前端模块协同训练得到高精度大模型策略后，该模块完成模型轻量化压缩，解决真机算力不匹配的问题，实现仿真策略到实体机器人运动能力的无损迁移。

整套体系运行数据流相对是固定的，机器人采集状态输入Actor网络生成动作、执行动作完成环境交互、奖励函数计算单步回报、Critic网络拟合价值函数并计算优势函数、PPO算法依据优势函数更新Actor网络参数、循环迭代至策略收敛、师生蒸馏轻量化模型、真机部署运行。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGdsm8tLlYUZnWOlvd5yCI9NVdKlD2hd1bKvsjnxgArU0V0eMczEJtnFLqSib3Bew2oLKmZghx6w6TmA6BicIUITHE7bL3ATelWWY/640?wx_fmt=png&from=appmsg#imgIndex=16)

该强化学习体系与人形机器人传统模型控制体系形成互补关系。

- 传统WBC、MPC控制依托动力学建模实现精确的轨迹跟踪，稳定性高、安全性强，但泛化能力弱。
- 强化学习体系无需精细人工建模，依托数据迭代实现自适应运动，泛化能力强、适配复杂场景。

当前工程落地的主流方案为混合控制架构，底层依托传统控制保障运动安全与基础稳定性，上层依托强化学习策略生成自适应运动指令，结合两类技术的优势，实现人形机器人高稳定、高泛化的智能运动控制。

编辑｜咖啡鱼

审编｜具身君

 ****推荐阅读**
[![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcibJS8986MfCcVATGOkcK6lNQfiaTORbuhSFoATTmZ5kA6nV8l8REia7nm4A4OxC1yOePBqrzWHQQd0ALicYANgOoNmRbibChjcAuQ/640?wx_fmt=png&from=appmsg#imgIndex=17)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=3824573915845640194&scene=126#wechat_redirect)[![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGcRfEtsGjVkl7cXB7QYAAib4wOMhdRcvsQicHnmiaxqoibw9LUCGGcPGSYnUPeUlZEoiaBlQezclFhp5yZQ6yLcLAjYeI67pJmvhOMw/640?wx_fmt=jpeg&from=appmsg#imgIndex=18)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=944555238&lang=zh_CN#wechat_redirect)

**![Image](https://mmbiz.qpic.cn/mmbiz_png/qKE443uRvLo6ic3ZPUttmFZ2AefQ4wjHSlQluSDkaxL9icWicpPYYmpo1Wa37Scjhh4AS5VwYJtmlTf5cKMiaIXg5g/640?&random=0.17349735674179656&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&wx_fmt=other#imgIndex=19)**

**【深蓝具身智能】****的原创内容均由作者团队倾注个人心血制作而成，希望各位遵守原创规则珍惜作者们的劳动成果；未经授权禁止任何机构或个人抓取本账号内容，进行洗稿/训练，否则侵权必究⚠️⚠️**


![Image](https://mmbiz.qpic.cn/mmbiz_png/Nabxc8rdYriaKqxCUjcZ8sSCnSNlWpqdI1kyXXQjXbtv95xvACqQoqL2ibbKXt9PB0FLPibKiawGsTcQrnKDGWVw2Q/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1#imgIndex=20)

点击❤收藏并推荐本文**
