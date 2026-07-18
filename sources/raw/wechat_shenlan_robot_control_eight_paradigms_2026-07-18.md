---
title: 机器人控制算法八大体系详解：从 PID 到强化学习
author: 深蓝具身智能
date: "2026-07-18 10:56:00"
source: "https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g"
---

# 机器人控制算法八大体系详解：从 PID 到强化学习

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/kaugqJpv9nuCktylvYoMKHYNAVojoRUpfyf1py08JvUnkfPXArzj4t5bMiaS6RBCXHHGhf8xlyw8icHrJcjEyYoA/640?wx_fmt=gif&from=appmsg#imgIndex=0)

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGcPV8eQcN9UW8ALicjBWCUrcwOFn8GOk0pgzZy4S4081qPvMAm2EIaVBHQGYU5CTIhZpXjZjm0rr7ibwKmtoXZSKttbLtOVDXYyU/640?wx_fmt=jpeg#imgIndex=1)

高水平的机器人控制从来不是某一类算法的“独角戏”

——三条主线厘清机器人控制算法全貌

机器人控制算法种类繁多，散见于不同理论分支：PID、MPC、鲁棒控制、自适应控制、强化学习……

这些方法之间究竟是什么关系？面对具体项目时该从哪类入手？

本文试图提供一个结构化的认知框架：

> 从机器人控制系统的分层架构出发；
>
> 先锚定控制算法在整个系统中的位置；
>
> 再展开对算法层内部方法的分类剖析。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGdBibCpGENKHS0rc6CL4Ux41begibVXialXcC8KjT8CHicib4INmGee1kKfEArs3P78ecGLfIWXXpiaCDRkp9PPn6lFgFagZ8owhVG20/640?wx_fmt=png&from=appmsg#imgIndex=2)

▲图| 控制算法所在位置©【深蓝具身智能】编译

机器人控制系统遵循分层闭环架构，从上至下依次为：任务规划层、控制算法层、伺服执行层。

控制算法层作为核心，可依据建模方式、抗扰能力、数据依赖度分为八大类：

> 经典线性反馈控制
>
> 基于模型的非线性动力学控制
>
> 鲁棒控制
>
> 自适应控制
>
> 位置/力混合控制
>
> 滚动优化与迭代学习控制
>
> 机器学习驱动控制
>
> 强化学习智能控制

需要特别指出：这八类算法并非简单的平行并列关系，而是遵循了一条从经典到现代、从解析到数据的演进逻辑：前四类属于显式建模控制，后四类分别面向接触作业、约束优化、数据补偿与自主习得。

在逐一展开之前，我们先统一以下核心术语：

- **系统状态**：机器人当前可量化的运动信息，包括关节角度、运动速度、加速度、末端位姿等；
- **误差**：目标期望状态与机器人实际状态的差值，是所有反馈控制的核心输入；
- **扰动**：外部碰撞、负载变化、摩擦漂移等影响机器人运动的非预期变量；
- **稳态误差**：系统动态收敛后，始终残留的固定偏差，无法通过比例调节消除；
- **非线性耦合**：多关节机器人中，单个关节运动对其他关节产生的惯性、离心干扰；
- **前馈控制**：基于模型提前计算补偿量，在误差产生前干预系统，无滞后；
- **反馈控制**：基于实时误差修正输出，抵消已产生的偏差，存在小幅滞后。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4YWHX9iazKkdkgh363zh9GFAfZia4RWWoYhutUeS8g43MnicLMfe9kUAZg/640?wx_fmt=jpeg&from=appmsg#imgIndex=3)

# **经典线性反馈控制（伺服底层基础）**

该类别是所有机器人的底层核心控制单元，针对**线性、弱扰动、单输入单输出**系统设计，主要用于电机、舵机、单关节伺服闭环。

无需复杂动力学建模，是工业机器人的标配底层算法。

伺服，就是精确控制的意思。

举个例子：人控制自己的胳膊手是精确控制，想到哪儿就动到哪儿，很精确；洗衣机甩干就不是伺服，是乱甩。

## **核心术语解析**

- **线性系统**：输入与输出呈正比例关系，无耦合、无滞后、无参数漂移的控制系统；
- **闭环反馈**：通过传感器采集实际输出，与目标值对比修正输入的控制逻辑；
- **极点**：决定控制系统收敛速度、震荡程度的核心特征量，极点位于复平面左半区间时系统稳定；
- **状态空间**：用矩阵形式描述多变量系统输入、状态、输出关系的数学模型。

## **代表性算法及原理**

**代表算法一：PID控制（比例-积分-微分控制）**是最通用的无模型线性控制算法。

它由三个核心调节环节组成：

- **比例环节（P）**根据当前误差大小实时输出驱动力，误差越大出力越大，快速拉近实际值与目标值；
- **积分环节（I）**累计历史残留误差，逐步叠加补偿量，专门消除稳态误差；
- **微分环节（D）**根据误差的变化速率预判趋势，误差快速增大时提前反向制动，抑制超调与震荡。

怎么理解「P」「I」「D」？

举个例子：开车时，P是根据当前偏离车道的幅度打方向，I是修正长期跑偏的惯性偏差，D是预判车辆偏移速度提前回正方向。

**代表算法二：LQR线性二次调节器，**基于状态空间的多变量最优控制算法。

核心原理是同时最小化两个核心指标：机器人轨迹跟踪误差、电机控制能耗。

通过矩阵求解一组固定反馈增益，在控制精度与动力消耗之间取得全局最优平衡。

配套的**龙伯格观测器**可通过部分传感器数据，估算系统无法直接测量的内部状态。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcicYgyxXEdr2MCibKCwZrJ0cqBkkw61BbJ6tkNSuJ2MZohfRyNEia7M8hDlzKGgad5LJ8DWVLrhEZolaBTibSD94b8UUcLZkfcPIQ/640?wx_fmt=png&from=appmsg#imgIndex=4)

▲图| LQR的基本公式，描述状态偏差随时间的变化关系，状态偏差随时间的变化关系。©【深蓝具身智能】编译

****代表算法三：**极点配置控制，**是人为指定控制系统极点的位置。

主动定义系统的收敛快慢、震荡幅度，将不稳定的原生系统修正为指定动态特性的稳定系统，多用于高精度伺服系统的动态调校。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGeBoE4iaOviaHAiabbvXll9qYP6Jgt3vibzGGp6UXVPhicj92NcN9j64wel31ws4dBnCPnzpvmIPuDvIRKL7rPpNCzyXxOSBvVBo48M/640?wx_fmt=png&from=appmsg#imgIndex=5)

▲图| 极点配置控制的“古早”研究©【深蓝具身智能】编译

当控制对象从单关节扩展到多自由度机械臂时，系统呈现强非线性与动力学耦合特征。

由此引入基于模型的非线性动力学控制。![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibqgVaodXH45G6Pdbk9xSEsUtlicqgxKkAiaK0P8QzGwuLiatibYiaIagQoOg/640?wx_fmt=jpeg&from=appmsg#imgIndex=6)

# **基于模型的非线性动力学控制（多关节机器人）**

针对多自由度机械臂、人形机器人等强非线性、强耦合载体设计，核心依赖机器人精确的**动力学模型。**

通过模型抵消系统固有非线性项，是中高端工业机械臂的主流中层控制算法。

## **核心术语解析**

- **动力学模型**：描述机器人关节力矩与运动状态（位置、速度、加速度）映射关系的数学方程，包含惯性、重力、离心力、科氏力四项核心参数；
- **反馈线性化**：通过坐标变换与状态反馈，将非线性系统等效转化为标准线性系统的数学方法；
- **模型失配**：实际机器人参数与理论动力学模型参数存在偏差的现象，是基于模型控制的主要误差来源；
- **前馈补偿**：基于目标轨迹提前计算力矩，在运动前抵消固有干扰，无反馈滞后。

## **代表性算法及原理**

******代表算法一：****CTC计算力矩控制**是机械臂最经典的非线性控制算法。

CTC控制本质上是“前馈+反馈”的复合控制。它利用机器人动力学模型（涉及惯性、科里奥利力、重力等）计算出一个前馈力矩，抵消系统的非线性和耦合效应，将复杂的非线性系统“线性化”为一个解耦的、类似弹簧阻尼的线性系统；再叠加反馈控制（如PID）来消除轨迹误差、应对模型偏差和外界扰动。

缺点是高度依赖精准动力学参数，模型失配时性能大幅下降。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGeB4FcjiatlLibbibu5edaYQRzibOTiadPxU4znEhrQsw7Jb2UicYanzRTvUvJlicm2kSGdrgsic9tremEx8l0QvBYEPEwIQic1ROShwuuM/640?wx_fmt=png&from=appmsg#imgIndex=7)

▲图| CTC控制过程©【深蓝具身智能】编译

****代表算法二：**逆动力学控制（IDC）**是CTC的基础形态。

根据目标轨迹的期望位置、速度、加速度，通过动力学方程逆解，直接求解关节所需的理论力矩作为前馈量，再叠加少量反馈修正偏差，侧重前馈驱动，闭环修正能力弱于CTC。

****代表算法三：**全局反馈线性化控制，**属于通用非线性控制方法，不仅限于机器人。

通过数学变换完全消除系统所有非线性特征，将任意非线性系统转化为标准线性系统，后续可直接复用所有线性控制算法，CTC是该算法在机器人领域的专属工程实现。

上述基于模型的方法在模型精确时性能优越，但实际工况中扰动与参数偏差不可避免。

由此发展出两类不依赖精准模型的应对策略：鲁棒控制、自适应控制，前者侧重被动抵抗，后者侧重主动辨识。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibTvia4qLjYyoM2Do58jX9J71HickLLA3NxCQp6fPljkgY26WIeaoeeYVQ/640?wx_fmt=jpeg&from=appmsg#imgIndex=8)

# **鲁棒控制（抗扰动、模型失配场景）**

鲁棒控制不依赖精准系统模型，核心目标是在存在外部扰动、模型参数偏差的情况下，强制保证系统稳定、误差可控。广泛应用于户外移动机器人、重载机械臂等复杂工况。

鲁棒，想必大家非常熟悉了，是robust健壮的意思，也可以理解为健壮性控制：你踢一脚正在步行的机器人，它不会摔倒马上回正，就是健壮性。

## **核心术语解析**

- **鲁棒性**：控制系统抵抗外部扰动、内部参数偏差的能力；
- **滑模面**：人为设计的误差收敛理想轨迹，是滑模控制的核心约束目标；
- **抖振**：滑模控制高频切换输出产生的微小高频震荡，是工程中主要抑制目标；
- **扰动传递增益**：外部扰动转化为跟踪误差的放大倍数，H∞控制的核心优化指标。

## **代表性算法及原理**

******代表算法一：****SMC滑模控制**是工业最常用的鲁棒控制算法。

直观原理是预先设计一条误差收敛的滑模面，控制器通过高频切换驱动力，强制机器人的运动状态始终贴合滑模面滑动收敛至零误差。

该算法对外部碰撞、模型误差完全不敏感，鲁棒性极强；工程中通过高阶滑模、终端滑模改良原生算法的抖振问题。

滑模控制（Sliding Mode Control, SMC）本身是一种处理系统不确定性和外部干扰的鲁棒控制方法——让系统状态像“滑滑梯”一样，被强制引导到一条预设的稳定轨道（滑动面）上，然后沿着这条轨道自动滑向目标点。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGf7OpnAibcO7vicP3DlXphPU6kIc6fQwBLiaJ8OicCYuD4H9EY91UDqmGjb2WicdbpkhMkva4DbsnOrnhtia5xZq46AFYmXI7j3VUbfU/640?wx_fmt=png&from=appmsg#imgIndex=9)

▲图| mathwork对机器人滑模控制仿真的官方实例©【深蓝具身智能】编译

********代表算法二：******H∞控制**是最优鲁棒控制算法。

核心原理是将所有外部扰动、模型误差定义为系统输入，通过优化最小化“扰动→跟踪误差”的最大传递增益，保证机器人在最坏的扰动工况下，误差仍被严格限制在安全范围。

适用于精密手术机器人、航空机器人。

******代表算**************法三：******μ综合控制**是H∞控制的进阶版本。

专门针对多维度不确定参数系统。通过量化机器人多关节耦合带来的参数不确定性，计算系统稳定裕度，比H∞更适配多自由度复杂机器人的鲁棒优化。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4yZibwaOWpB3DrcxuiafpXicx2ibHiaHAZFr7ptU6ud2hsxgCXvV0JGHtTDw/640?wx_fmt=jpeg&from=appmsg#imgIndex=10)

# **自适应控制（参数时变、负载动态变化场景）**

自适应控制区别于鲁棒控制：鲁棒控制是“被动抵抗干扰”，自适应控制是“主动辨识参数并修正模型”，专门解决机器人负载变化、零件磨损、工况漂移等时变参数问题。

## **核心术语解析**

- **时变参数**：随时间、工况变化的系统参数，如抓取负载重量、关节摩擦系数；
- **在线辨识**：机器人运动过程中实时采集数据，迭代更新参数的过程；
- **参考模型**：预设的理想无误差动力学模型，作为自适应修正的对标基准；
- **参数收敛**：辨识得到的未知参数逐步逼近真实物理参数的过程。

## **代表性算法及原理**

******代表算************法一：MRAC****模型参考自适应****控制**属于经典自适应算法。

首先预设一个动态性能最优的参考模型，实时对比机器人实际运动输出与参考模型输出的差值；

然后通过自适应律在线更新控制器参数，逐步缩小两者差距；

最终让机器人的运动特性完全匹配理想模型，适配变负载抓取场景。

**代表算法二：自适应计算力矩控制****（A-CTC）**属于传统CTC的改进版本。

针对CTC依赖精准参数的缺陷，通过在线辨识实时更新机械臂质量、惯量等未知参数，动态修正动力学补偿项，解决抓取不同重量工件时的控制精度衰减问题。

**代表算法三：RLS递归最小二乘辨识控****制**属于参数辨识核心算法。

实时采集机器人关节运动数据，通过迭代运算最小化预测值与实际值的误差，持续更新动力学参数估计值，辨识结果直接用于前馈力矩补偿，是所有自适应控制的基础辨识工具。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4hKdH0P2rRHX1TlxUqlAx7X6m2hcl7XttttyRW05mhbEa1msX7zEzvw/640?wx_fmt=jpeg&from=appmsg#imgIndex=11)

# **位置/力混合控制（环境接触作业专用）**

前文所述算法均为**位置控制**，一般仅适用于无接触自由运动场景。

当机器人执行打磨、装配、人机交互等接触任务时，纯位置控制会产生巨大接触冲击力，力控算法专门解决机器人与环境的柔顺交互问题。

## **核心术语解析**

- **末端阻抗**：机器人末端对外力表现出的等效刚度与阻尼特性，决定接触时的柔顺程度；
- **任务坐标系解耦**：将机器人工作空间拆分为多个独立方向，分别配置控制模式；
- **六维力传感器**：可检测末端三维力、三维力矩的专用传感器，是力控的硬件基础；
- **柔顺运动**：机器人受外力时主动偏移位置，降低接触力的运动特性。

## **代表性算法及原理**

**代表算法一：阻抗控制****是**主流的核心力控算法。

不直接控制末端位置或接触力，而是调节机器人末端的等效刚度和阻尼：

高刚度时优先保证定位精度，低刚度时末端具备柔顺性，碰到障碍物自动退让缓冲冲击。

类比调节弹簧硬度，打磨时调软弹簧实现柔顺接触，钻孔时调硬弹簧保证定位精准。

**代表算法二：****导纳控制**是与阻抗控制对偶的算法。

以六维力传感器采集的外力作为输入，根据外力大小计算末端位置修正量，外力越大偏移越多，适配大负载机械臂、人机协作机器人。

导纳控制是机器人控制中的一种力控制策略，属于阻抗控制的变体。

其核心思想是通过机器人末端的运动来响应外部施加的力（如人机交互或环境接触），模拟一个“虚拟弹簧-阻尼”系统。

具体来说，导纳控制会测量外部力，然后调整机器人的位置、速度或加速度，使其表现得像惯性-阻尼-弹簧系统一样顺从（例如推它会退让）。

与刚度控制不同，它更强调力输入、运动输出的关系，广泛用于协作机器人、医疗机器人等需要安全柔顺交互的场景。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGdOW2mvkFYjL8otGibI2POBaTpKBGxawdL3icl1cvuYzNC5j9gRicQygbBX9haHtBaJIvLzQdmJStYiapTR4ZgJ0OgSGbiaKSJW6vXo/640?wx_fmt=png&from=appmsg#imgIndex=12)

▲图| 阻抗和导纳控制原理图©【深蓝具身智能】编译

****代表算法三：**位置/力混合控制。**

基于任务坐标系解耦原理，拆分机器人运动方向：

- 无约束的自由方向采用位置闭环控制；
- 受环境约束的接触方向采用力闭环控制，两类控制互不干扰。

典型应用例如插销装配中，XY平面控制位置对准，Z轴控制顶紧力完成插接。

**代表算法四：直接力反馈控制，**属于最简力控算法。

以预设目标接触力为参考值，通过力传感器实时采集实际接触力，直接做闭环反馈，精确地跟踪目标压力，适用于精密按压、恒力打磨场景。

部分应用场景对控制算法提出了超出单步反馈的优化需求：要么存在电机限位、避障等硬约束（如人形机器人行走），要么轨迹高度重复（如流水线搬运）。

滚动优化与迭代学习分别应对这两类问题。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX7OGyOG8gzibCyLX91kbhEcWl0mnLk5Zb5uVRabIn51LEKNicYT8OlZZQ/640?wx_fmt=png&from=appmsg#imgIndex=13)

# **滚动优化与迭代学习控制（约束/重复轨迹场景）**

该类别属于优化型控制，结合系统约束与历史运动数据，突破传统单点控制的局限，分别适配带物理约束的动态轨迹、重复式工业轨迹两类场景。

## **核心术语解析**

- **滚动时域**：MPC中每次优化仅针对未来一小段时间窗口，实时更新优化序列；
- **硬件约束**：机器人电机最大转速、力矩、运动行程等物理限制；
- **迭代批次**：ILC中机器人重复执行同一轨迹的单次完整运动周期；
- **批次误差**：单次重复轨迹全程的累计跟踪偏差。

## **代表性算法及原理**

**代表算法一：MPC模****型预测控制，**是通用滚动优化算法。

原理是机器人每个时刻，基于动力学模型预测未来一小段时间窗口内的运动轨迹，结合电机限位、避障等硬件约束求解最优控制序列，仅执行当前第一步指令，下一时刻重新预测、重新优化。

优势是天然适配多约束系统，广泛用于人形机器人步态控制、AGV轨迹跟踪。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGfzxODRjtIjz5IrxwPEnqiae9DXicxRvc2AbPfNCCDSh4fAgE8icsYL1tggYxy3nDD2Nua03MEn0mFlKrogr8zbQ2NR7eP0H1jyxE/640?wx_fmt=jpeg#imgIndex=14)

▲图| MPC控制原理图©【深蓝具身智能】编译

**代表算法二：ILC迭代学习控制**针对重复轨迹的专属优化算法。

原理是机器人执行同一重复轨迹时，记录上一批次运动的全程误差，将误差转化为前馈补偿量叠加到当前批次指令中，迭代次数越多，轨迹跟踪精度越高。

专门用于流水线搬运、固定路径打磨等批量重复工况。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGfB5liajyNIx2IfNSuc5ZT1cUOLAr9qGSuNOUFNokfVYXibSrnA57XU9Qia6OtPESHQPfibmPAmFCdpfpawGER8aw5wClSm8dWAZc0/640?wx_fmt=png&from=appmsg#imgIndex=15)

▲图| 迭代学习控制原理图，来自网络©【深蓝具身智能】编译

当系统建模残差（如复杂摩擦、柔性形变）难以用解析式描述时，数据驱动方法提供了新的解决路径。

机器学习控制即属此类。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX5ne3MfNYQBbic4xIYsEJDKpCRqQXk6gllicSqc7QiabhaIEuCXA1I4xsg/640?wx_fmt=png&from=appmsg#imgIndex=16)

# **机器学习驱动控制（数据驱动建模补偿）**

机器学习控制属于**离线/在线数据拟合类算法**，核心是通过标注数据集训练模型，拟合传统解析模型无法描述的复杂非线性、摩擦残差。

早些年作为传统控制器的补偿模块，现在大有成为主流的趋势，依赖标注数据，无需机器人与环境实时试错。

## **核心术语解析**

- **监督学习**：基于输入-输出标注数据集训练模型，学习已知规律；
- **非参数模型**：无需预设网络结构，由数据自主生成模型结构（如高斯过程）；
- **时序依赖**：机器人连续运动中，当前状态受历史运动数据影响的特性；
- **建模残差**：理论动力学模型与实际机器人运动特性的差值。

## **代表性算法及原理**

**代表算法一：神经网络补偿控制。**

利用神经网络的万能拟合特性，离线采集机器人轨迹、力矩、摩擦数据训练网络，精准拟合建模残差；设备运行时，神经网络输出补偿力矩叠加到传统控制器中，抵消解析模型无法描述的非线性误差。

细分包括适配常规时序的全连接网络、捕捉历史依赖的LSTM、长距离耦合的Transformer网络。

**代表算法二：高斯过程（GP）控制，是经典非参数机器学习算法。**

**无需预设模型结构，基于少量机器人运动数据建立概率动力学模型，不仅能预测下一时刻运动状态，还能输出预测不确定度，可用于安全约束下的高精度控制，适配手术机器人等低数据场景。**

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGeIYxZ2lKicKL9CXPyyKGAW5jnG0SaT0GEo4cUrD2dEiaxXvX6kM3lLuApic2REg2T2w6HDLjSYuyQibp0ABQgEJ4qhygTTNs2Mfbs/640?wx_fmt=png&from=appmsg#imgIndex=17)

▲图| GP控制，来自网络©【深蓝具身智能】编译

**代表算法三：模糊逻辑控制，传统机器学习分支，模仿人类经验推理。**

将人工操作经验转化为“If-Then”模糊规则，无需精准动力学方程，通过模糊推理输出控制量，适合结构复杂、难以建模的非标机器人。

**代表算法四：无监督聚类故障补偿控制。**

基于无监督学习对机器人运行工况聚类，自动识别摩擦漂移、零件磨损等工况变化，实时切换对应的控制补偿参数，实现数据驱动的被动自适应调节。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGevHej3PwCLyfh7e01VgiaaTt5uiaQVicsxwVuic1RyR2nzo1VyhtXKD4biaQjm03LGABUSFycQKic2glm1SoHEUBYl94ibIacjDzBWOI/640?wx_fmt=png&from=appmsg#imgIndex=18)

▲图| 各种常见聚类效果，来自网络©【深蓝具身智能】编译

机器学习中的监督学习依赖已有标注数据拟合映射关系；

强化学习则面向无标注的未知环境，机器人通过持续的交互试错自主优化控制策略。

两类方法在数据需求与学习范式上存在本质区别。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGePDaQXrfxrQPQ0bLUUQmTaZ1XPdklJwXrKUwQ1WrJrXkCF5CLMarjuTvy8r3D1iaibtxZzPq3Ld27KqjSHRd5sg1cv9e0aU9MibY/640?wx_fmt=png&from=appmsg#imgIndex=19)

# **强化学习智能控制（交互试错自主优化）**

强化学习（RL）是**环境交互型无标签学习算法**，区别于监督机器学习，无需标注数据集，机器人通过与环境持续试错交互，依靠奖励函数自主优化控制策略。

适用于无事先准备的模型、未知环境、复杂多任务场景。

## **9.1 核心术语解析**

- **MDP马尔可夫决策过程**：强化学习的数学基础，由状态、动作、奖励、环境转移概率组成；

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcqribTNHNZ42VFWFltibeloFmZZic8RyiaMiayWgGDKC3bvV6eMP9aTc8Znwla06IdnQWZu1VGNQzuWxtQofiaQqxibzmibiakiazYVujjo/640?wx_fmt=png&from=appmsg#imgIndex=20)

▲图| 马尔可夫决策过程©【深蓝具身智能】编译

- **智能体（Agent）**：执行动作的机器人本体；
- **奖励函数**：量化机器人动作优劣的评价指标，误差小、无碰撞则奖励高；
- **策略网络**：输入机器人状态，直接输出控制动作/力矩的神经网络；
- **值函数**：评估当前状态下所有动作长期累积奖励的量化函数；
- **模仿学习**：以人类示教或传统控制器轨迹预训练策略，再通过RL微调的复合学习方式。

## **核心分支与代表性算法**

**代表算法一：基于值函数的强化学习（离散控制）。**

核心通过值函数评估动作价值，输出离散控制量，适配低维简单机器人，主要包含以下几种方式：

- **Q-Learning**

构建Q表存储“状态-动作”的长期奖励，机器人试错更新Q表，查表获取最优动作，用于单关节、简易AGV；

- **DQN深度Q网络**

用神经网络替代Q表，解决高维状态存储爆炸问题，通过经验回放、目标网络稳定训练，适配高维机械臂离散动作控制；

- **Double/Dueling DQN**

修正DQN过估计偏差，拆分状态价值与动作优势，提升训练稳定性。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGfNf94FMV8QCYL4NlDfMhkeqBU6TmgxBfkRnmDmsgN8OFljQcVTtaQnyLFdSWUXV7WDkCseibvb1ugNB4cq4XYlNiafMLJiaCDPPI/640?wx_fmt=png&from=appmsg#imgIndex=21)

▲图| 最简单的Q-learning©【深蓝具身智能】编译

**代表算法二：策略梯度强化学习（连续控制主流）。**

直接优化策略网络，输出连续力矩、速度等控制量，是现代机器人RL的核心。主要包含：

- **PG策略梯度**

沿奖励提升方向直接更新策略网络权重，天然适配连续控制；

- **PPO近端策略优化**

是工业最常用RL算法，限制策略单次更新幅度，避免破坏已学习的稳定控制逻辑，调参简单、训练稳定，用于机械臂装配、人形步态优化；

- **TRPO信赖域策略优化**

严格约束策略更新范围，保证训练过程系统绝对稳定，计算复杂度高于PPO；

- **A2C/A3C演员-评论家算法**

采用双网络架构，Actor输出控制动作，Critic评价动作优劣，多线程并行训练加速收敛。

**代表算法三：基于模型的强化学习（Model-based RL）**

先学习环境/机器人动力学模型，再在虚拟模型中试错，减少真实硬件交互损耗：

- **PILCO**

结合高斯过程学习概率动力学模型，虚拟仿真优化策略，降低实体机器人磨损；

- **Dreamer世界模型RL**

训练神经网络世界模型，机器人通过“想象”预测未来状态，绝大部分学习在虚拟梦境完成，适配昂贵人形、手术机器人。

**代表算法四：分层强化学习（HRL）**

解决复杂多任务学习难、收敛慢的问题：

- 上层策略拆分多阶段子任务（移动-抓取-放置）；
- 底层子策略完成具体运动控制。

典型算法为Option-Critic、分层PPO，用于机器人复杂操作任务。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGf1zcX33bPYibd1gxrs9BUZibJdtiaglxdDNuwHT5iaJbwkXAMfM6TGtujPNupsiavAib82Ld3tDqyKkaMm8cZKHuF7W9OmIDZMr7Uw0/640?wx_fmt=png&from=appmsg#imgIndex=22)

▲图| 分层强化学习，来自网络©【深蓝具身智能】编译

**代表算法五：各类****衍生****的****强化学习鲁棒RL**

训练时注入扰动与模型误差，让策略天生具备抗干扰能力；**自适应RL**负载变化时在线微调策略网络，无需重新全量训练；

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcYbUIN46WRctJR4ozaFQfaFibOwebf3w4zGU93NU0tDy7rMsuO8lBakYTlS8O0okKQd5vrbElXtMdtUozMPEe2VVVJGcicWprow/640?wx_fmt=png&from=appmsg#imgIndex=23)

▲图| 一种衍生的强化学习，来自网络©【深蓝具身智能】编译

**代表算法六：模仿学习（BC/GAIL）**

通过示教轨迹预训练初始策略，再用RL微调，解决纯RL从零试错效率低的问题。

需要指出的是，上述分类侧重于算法机理层面的区分。在实际工业级机器人控制系统中，多类算法常以叠加或级联形式融合使用。

一个典型的融合示例是：

工业机械臂常采用“CTC动力学前馈+滑模鲁棒修正+ILC迭代优化”的复合架构，人形机器人则结合“MPC约束控制+RL步态优化”。

理解每类算法的原理与边界，是为融合设计奠定基础的关键前提。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGcBveibicjpj9A6fibMpHqcmMBUcXHpm3BOv6asZiaP4TpQZ85j5gmMjKlma0PzHgNuG1Jt9PNofdsJypaSFpiboyyKV9nV5jicicmJ4I/640?wx_fmt=png&from=appmsg#imgIndex=24)

行文至此，我们从底层到顶层构建了完整的机器人控制算法体系，八大类别形成清楚的技术递进关系：

经典线性控制奠定伺服基础，非线性动力学控制解决多关节耦合问题，鲁棒与自适应控制应对扰动和参数变化，力控实现环境柔顺交互，优化类算法适配约束与重复轨迹，机器学习完成数据驱动补偿，强化学习实现未知环境自主学习。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGej4botwoEAGYuGAhia301F8CynzqFVYbQvc9jo0oIUylAB4jedyVuJyIpGRfN6GCWonzricb3Hcl6d0bhKuAgkcHkACtTJmwKwc/640?wx_fmt=png&from=appmsg#imgIndex=25)

当然，这一演进并非意味着传统控制理论的失效。

恰恰相反，当前机器学习与强化学习方法在机器人领域的成功应用，均建立在底层伺服控制稳定可靠、中层动力学模型准确有效的基础之上。

没有PID对电流环的精确闭环，任何上层学习算法都无法转化为实际的力矩输出；没有CTC对非线性耦合的有效抵消，RL策略将面对一个极度不稳定的被控对象，训练收敛无从谈起。

对于具身智能系统的开发者而言，理解这一技术体系的层次结构与演进逻辑，意义更在于建立从底层物理约束到上层智能决策的系统性设计思维。

高水平的机器人控制从来不是某一类算法的“独角戏”，而是前馈与反馈、模型与数据、优化与学习的多层次协同。

**归根结底，控制不是选择题，而是一道系统集成题。希望这篇文章能够帮助大家建立清晰的认知框架。**

编辑｜咖啡鱼

审编｜具身君

 ****推荐阅读**
[![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcibJS8986MfCcVATGOkcK6lNQfiaTORbuhSFoATTmZ5kA6nV8l8REia7nm4A4OxC1yOePBqrzWHQQd0ALicYANgOoNmRbibChjcAuQ/640?wx_fmt=png&from=appmsg#imgIndex=26)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=3824573915845640194&scene=126#wechat_redirect)[![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGcRfEtsGjVkl7cXB7QYAAib4wOMhdRcvsQicHnmiaxqoibw9LUCGGcPGSYnUPeUlZEoiaBlQezclFhp5yZQ6yLcLAjYeI67pJmvhOMw/640?wx_fmt=jpeg&from=appmsg#imgIndex=27)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=944555238&lang=zh_CN#wechat_redirect)

**![Image](https://mmbiz.qpic.cn/mmbiz_png/qKE443uRvLo6ic3ZPUttmFZ2AefQ4wjHSlQluSDkaxL9icWicpPYYmpo1Wa37Scjhh4AS5VwYJtmlTf5cKMiaIXg5g/640?&random=0.17349735674179656&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&wx_fmt=other#imgIndex=28)**

**【深蓝具身智能】****的原创内容均由作者团队倾注个人心血制作而成，希望各位遵守原创规则珍惜作者们的劳动成果；未经授权禁止任何机构或个人抓取本账号内容，进行洗稿/训练，否则侵权必究⚠️⚠️**


![Image](https://mmbiz.qpic.cn/mmbiz_png/Nabxc8rdYriaKqxCUjcZ8sSCnSNlWpqdI1kyXXQjXbtv95xvACqQoqL2ibbKXt9PB0FLPibKiawGsTcQrnKDGWVw2Q/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1#imgIndex=29)

点击❤收藏并推荐本文**
