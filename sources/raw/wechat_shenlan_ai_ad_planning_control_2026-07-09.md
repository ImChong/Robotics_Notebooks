---
title: 自动驾驶核心算法盘点｜规划与控制篇
author: 深蓝AI
date: "2026-07-09 17:32:00"
source: "https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247602818&idx=1&sn=e6a0f914dcdd7878d5f4993d247fb85c"
---

# 自动驾驶核心算法盘点｜规划与控制篇

![](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAKPXr0kicZddyXdPOg1Jm7tKusPLcWicG0ALpMqpjSHZxxsu45C13rzA4XZ2leKiaxG64fPqc9zIRIj8CR43YYibVy9ic8aRib3LUd8/640?wx_fmt=png&from=appmsg#imgIndex=0)

在之前的[《自动驾驶核心算法盘点》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzY4NjA5NTgyMQ==&action=getalbum&album_id=4596755873481310212#wechat_redirect)系列中，我们已经深入探讨了 2D 与 3D 目标检测算法。

本文是该系列的第3篇，聚焦规划与控制。

在这篇推文中，我们将顺着自动驾驶系统的架构，为您盘点工业界与学术界最具代表性、引用量极高的经典规划与控制算法。

我们将尽量减少复杂的数学公式，用客观通俗的语言，带您看懂从全局搜索到局部轨迹，再到车辆底层控制的技术演进脉络。

**欢迎关注【深蓝AI】**将持续分享人工智能领域前沿动态👇***深蓝AI*****1****—******规控在自动驾驶系统中的定位****

## 在深入算法之前，我们先从宏观视角看看规控模块在自动驾驶全栈架构中的位置。

经典的自动驾驶软件架构通常被划分为感知、定位、规划和控制四大核心模块。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpBH8h6SjM7YGiaKIfZBT3Q2cK3k1nvnichS2yemnXdufuzhLgO2MeQ80k2aTeFskH4QLFtOBF9ZYrEFTs3XzU5hhdKglBk1pNpWk/640?wx_fmt=png&from=appmsg#imgIndex=1)

图1 | 自动驾驶系统高层软件架构概览，展示了感知、定位、规划、控制四大核心模块的数据流向。规划模块承上启下，接收感知与地图信息后输出轨迹，再由控制模块转化为底层执行指令。©【深蓝 AI】编译

规划模块承上启下，它接收来自感知模块的环境信息（动态障碍物、静态障碍物）和高精地图的全局路线，输出一条包含位置和速度信息的时空轨迹。随后，控制模块接收这条轨迹，计算出具体的方向盘转角、油门和刹车指令，最终驱动车辆行驶。

在规划内部，通常又会进行分层：

1. 全局路由（Routing）：类似我们在导航软件中搜索目的地，给出一条道路级别的长距离路线。
2. 行为决策：决定当前应该跟车、变道、超车还是避让。
3. 运动规划：在行为决策的指导下，生成一条平滑、安全、符合车辆运动学约束的局部行驶轨迹。

接下来，我们将重点盘点运动规划与底层控制领域的经典算法。

***深蓝AI*****2****—****规划篇：从栅格搜索到时空优化**

在运动规划领域，如何在一个充满障碍物的复杂空间中找到一条可行的路径是核心命题。


**1. Hybrid A\*：把车辆运动学真正纳入搜索**

提到路径搜索，很多人第一时间会想到经典的 A\* 算法。

A\* 在游戏和二维网格地图中表现优异，但直接应用于自动驾驶却存在致命缺陷：传统的 A\*算法只搜索二维平面上的点，它不知道汽车是不能横向平移的。这导致 A\* 搜出的路径往往包含锐角折线，真实车辆根本无法行驶。

为了解决这个问题，斯坦福大学的 Dmitri Dolgov 等人在 2008 年的 DARPA 城市挑战赛中提出了 Hybrid A\*算法。该算法至今在学术界已有逾千次引用，并广泛应用于自动驾驶的泊车和低速复杂场景。

![](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpADVnViajjp1DllVJamD8J8FS1GKTXibW0cQfSfSCweFpzIzhiawPwG0NibmPVftp6eOp0L8wQx5hwVXsW9XWKYnkibJHpZEEcrefP0/640?wx_fmt=png&from=appmsg#imgIndex=2)

图2 | Hybrid A 与传统 A 的状态扩展对比示意。左侧为传统 A\*，节点状态仅包含二维坐标，每次扩展只能移动到相邻网格；右侧为 Hybrid A\*，节点状态扩展为包含航向角的，扩展方向受车辆运动学约束，可前进或倒退，从而生成可被真实车辆执行的路径。©【深蓝 AI】编译

核心思路：Hybrid A\* 的突破在于，它将搜索空间从二维的坐标点扩展到了包含车辆航向角的连续状态空间。在每次向外探索新节点时，算法不再是简单地向周围网格移动，而是根据真实的车辆运动学模型（如最大前轮转角限制），模拟出车辆前行或倒车的一小段真实轨迹。

同时，为了保证搜索效率，它依然保留了网格化的思想来剪枝（剔除次优路径），并设计了结合障碍物距离和非完整约束的启发式函数（Heuristic），引导搜索快速向终点收敛。

![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpDwy8ra5Sxg3BGjibhLB9qPgBCMxiaD1OCNNOWcUJFmBuaVpJgK9lllAUdIAOpmzl5yevFC4aIhgR2MVzGtkPITKDmBL4ZT3vUFU/640?wx_fmt=jpeg&from=appmsg#imgIndex=3)

图3 | Hybrid A 在复杂停车场环境中的路径搜索可视化。绿色轨迹为算法规划出的完整行驶路径，右侧黄色扇形为当前节点的候选扩展方向。可以看到，算法能够在密集障碍物之间规划出包含多次转向和倒车的平滑路径。©【深蓝 AI】编译\*

适用场景：由于能完美处理车辆的转弯半径和倒车约束，Hybrid A\* 成为了自动泊车（APA）、狭窄空间掉头、非结构化道路避障的绝对主力算法。


**2. Lattice Planner 与 Frenet 坐标系：结构化道路的经典主线**

Hybrid A\* 在低速复杂场景很强，但在高速公路或城市主干道上，车辆通常是沿着车道线行驶的，完全在自由空间中盲搜效率太低。这时，Frenet 坐标系和基于它的 Lattice Planner（状态网格规划器） 成为了工业界的主流选择。

2010 年，Moritz Werling 等人在 IEEE ICRA 发表了《Optimal trajectory generation for dynamic street scenarios in a Frenet frame》，这篇高被引论文奠定了结构化道路轨迹规划的理论基础。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpB5CPsibpMk5aseAOMUYw97EgricLKe5GuibnDgs81a5HYnezBcSVaSb7EEaHITzicLWqia6icOWC41uenfCUaDr32IhbMbEtal0hwYk/640?wx_fmt=png&from=appmsg#imgIndex=4)

图4 | Frenet 坐标系示意图。以道路参考线（通常为车道中心线）为基准，将车辆的位置从笛卡尔坐标转换为沿参考线的纵向距离与垂直偏移量。这一变换将弯曲的道路在数学上"拉直"，大幅简化了轨迹规划的建模难度。©【深蓝 AI】编译

核心思路：Frenet 坐标系的巧妙之处在于，它不使用传统的直角坐标系，而是沿着道路的参考线（通常是车道中心线），将坐标分解为：

- 纵向距离：沿着参考线走过的距离。
- 横向偏移：偏离参考线的垂直距离。

这样一来，弯曲的道路在数学上就被"拉直"了。Lattice Planner 会在这个坐标系下，分别独立生成横向和纵向的候选轨迹集合（像网格一样铺开）。例如，横向上采样不同的偏移量来变道或避障，纵向上采样不同的速度来加速或减速。

随后，算法会将横纵向轨迹组合成大量的候选轨迹族，并通过一个代价函数（Cost Function）对它们进行打分。打分指标通常包括：是否碰撞、距离中心线多远、加速度是否过大（舒适性）、到达终点的时间等。得分最低的轨迹就是最终的输出。

适用场景：非常适合车道保持、高速变道、跟车、绕行静态障碍物等结构化道路场景。


**3. EM Planner（Apollo）：工业级路径-速度解耦框架**

在量产自动驾驶系统中，如何平衡规划的复杂度和实时性是一个巨大的工程挑战。百度 Apollo 开源平台在 2018 年发表的《Baidu Apollo EM Motion Planner》提供了一个极具代表性的工业级解决方案。

根据论文摘要，该系统在 2017 年之后已部署于数十辆自动驾驶测试车，累计完成了 3380 小时、约 68000 公里的闭环自动驾驶测试。

![](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpCFO3BIzfkur9UD0efeiaSibRhmwY1XMRqayp4Z7IJJEEnBQntoaUqrkAxsqs0xKOUBTcdhNVQ2NYE5ZC3pz5Wd0tZgTMAuewO0I/640?wx_fmt=png&from=appmsg#imgIndex=5)

图5 | EM Planner 系统整体架构图。顶层为参考线生成器（Reference Line Generator），对每条候选车道分别建立 Frenet 坐标系；随后在各车道内依次执行 SL 投影（E-step）、路径规划（M-step）、ST 投影（E-step）和速度规划（M-step）四个交替步骤；最终由参考线轨迹决策器（Reference Line Trajectory Decider）选出最优车道的轨迹输出。©【深蓝 AI】编译

核心思路：EM Planner 的核心思想是路径与速度的解耦。在复杂的动态交通中，同时优化三维时空的计算量呈指数级爆炸。EM Planner 将其拆分为两个迭代的步骤（类似机器学习中的 EM 期望最大化算法）：

1. 路径规划：先假设环境是静态的，在 Frenet 坐标系的平面内，规划出一条避开障碍物、平滑的几何路径。
2. 速度规划：在已经确定好的路径上，将动态障碍物投影到时间-距离平面（ST 图）上，规划出车辆在不同时刻的速度，决定是加速超车还是减速让行。

![](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpChzaNXYFJ3NUCibWCUvFtJzhPEyhAXSuDmmSticQWibvjRySrdBeF22sg7fXXJDZIZDADrd7cS9lngqdn5W6nn3x7gyWggWBFnGE/640?wx_fmt=png&from=appmsg#imgIndex=6)

图6 | ST 图上的速度规划结果示意。横轴为时间，纵轴为沿路径的纵向距离（单位：米）。红色斜线阴影区域为前方障碍物占据的时空区域，黄色区域为可行驶的安全空间。蓝色虚线为动态规划给出的粗略速度曲线，绿色虚线为二次规划在可行域内平滑后的最终速度曲线，图中可见"follow-yield（跟车让行）"与"overtake（超车）"两种决策对应的不同可行区域。©【深蓝 AI】编译

在路径和速度的各自规划中，EM Planner 都采用了"先粗搜，后精修"的策略：先用动态规划在网格中搜索出一个粗略的凸空间（可行走区域），再使用基于样条曲线的二次规划在这个凸空间内求解出极其平滑的最优曲线。

亮点：这种架构在工程上极大地降低了计算复杂度，同时保证了轨迹的平滑性和安全性，深刻影响了国内众多自动驾驶公司的规控架构设计。

***深蓝AI*****3****—****控制篇：从经典反馈到模型预测**

当规划模块输出了一条完整的轨迹（一串带有时间戳的坐标和速度要求），接下来的任务就交给了控制模块。控制器的目标是：克服车辆惯性、路面摩擦力变化和系统延迟，精准地跟踪这条轨迹。


**1. PID 控制：最朴素且广泛存在的基线**

比例-积分-微分（PID）控制是工业界最古老、应用最广泛的控制算法。在自动驾驶中，它常被用作纵向速度控制的基础方案。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpBOiauicw94E9zG7CekfWej5bdWYH54y9RKNTTexSUvCwMwg4vk8jFibHkfS2EdV69SUNcgExhoicZSSQIDNxiboaSZntFmb525Gabw/640?wx_fmt=png&from=appmsg#imgIndex=7)

图7 | PID 控制器标准结构框图。输入为参考值与实际输出之差（误差分别经过比例（P）、积分（I）、微分（D）三个环节加权求和后，生成控制量作用于被控对象（车辆）。三个增益系数决定了控制器的响应速度、稳态精度与抗超调能力。©【深蓝 AI】编译

核心思路：PID 的原理非常直观，它计算目标速度与当前实际速度的误差，然后通过三个环节来调整油门或刹车：

- P（比例）：误差越大，踩油门的力度越大。
- I（积分）：消除稳态误差。如果遇到上坡，单纯的 P 可能无法达到目标速度，I 会累积过去的误差，逐渐增加油门直到速度达标。
- D（微分）：预测误差的变化趋势，防止油门踩得过猛导致速度超调。

局限性：PID 结构简单、调参直观，但在面对高速、强非线性的车辆横向动力学时，纯 PID 往往力不从心，容易出现蛇形走位现象。


**2. LQR（线性二次型调节器）：横向控制的主力**

为了更好地控制车辆的方向盘，工业界广泛采用了基于现代控制理论的 LQR（Linear Quadratic Regulator） 算法。

![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpAtGTNu8RmoiaL63v4O6eUr9C23vrPdu66VP9X3FuzJ5Lu3YYhJdib0fibydiaZ0emudG0whGreeMCUzbqwHVQu8ONEXoptBY0rhy8/640?wx_fmt=jpeg&from=appmsg#imgIndex=8)

图8 | LQR 横向控制的状态变量示意图。图中定义了车辆相对于目标路径的横向偏差 （车辆质心到参考路径的垂直距离）和航向偏差 （车辆朝向与参考路径切线方向的夹角），这两个量构成 LQR 控制器的核心状态输入。LQR 通过最小化包含这两项误差和方向盘转角的综合代价函数，求解出最优反馈控制律。©【深蓝 AI】编译

核心思路：LQR 是一种最优控制算法。首先，它需要建立车辆的数学模型（通常是简化的二自由度车辆动力学模型）。然后，它定义了一个代价函数（Cost Function），这个代价函数由两部分组成：

1. 状态误差代价：车辆偏离目标轨迹的横向距离和航向角误差越小越好。
2. 控制输入代价：方向盘的转角和转角变化率越小越好（为了乘客舒适度，不能猛打方向盘）。

LQR 的数学魅力在于，对于线性化的车辆模型，它能够通过求解黎卡提方程（Riccati Equation），直接计算出一组最优的反馈增益矩阵。这意味着，在每一个瞬间，LQR 都能在"跟线精准"和"打方向平稳"之间找到一个完美的数学平衡。

适用场景：LQR 计算效率高，能够系统地处理多变量反馈，是工业界量产车中长期使用的横向控制主力方案之一。


**3. MPC（模型预测控制）：约束控制的终极武器**

随着自动驾驶向更高速、更复杂的极限工况发展，车辆的非线性特征（如轮胎侧滑）变得不可忽视，同时系统需要严格遵守各种物理限制（如方向盘最大转角、最大加速度）。这时，MPC（Model Predictive Control） 登上了舞台。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpDpJ4YlPZanibeypMm0TXibxvRuc0ndEcIQnz0xzEete5qJhgPWNADDTCsnmuYib6zoiaSaLIeABFUdkdUNAhahvfcTIjd0uOXmDlw/640?wx_fmt=png&from=appmsg#imgIndex=9)

图9 | MPC 滚动优化（Receding Horizon）原理示意图。在当前时刻，控制器利用车辆模型向前预测个时间步（即"预测时域"），在此范围内求解满足约束的最优控制序列（蓝色阶梯线）。预测输出（黄色点）会逐渐逼近参考轨迹（红色曲线），但控制器只执行当前时刻的第一个指令，随后在下一时刻重新滚动求解，以应对实际状态与预测的偏差。©【深蓝 AI】编译

核心思路：MPC 的核心思想是"看长远，顾当下"。在每一个控制周期，MPC 会利用高精度的车辆模型，预测未来一段时间内（例如未来 2 秒）车辆在不同控制输入下的行驶状态。

它会在满足所有物理约束（如前轮转角不能超过 30 度，侧向加速度不能超过 0.2g）的前提下，通过在线求解一个复杂的约束优化问题，找到在整个预测时域内最优的控制指令序列。然而，虽然它计算出了一连串未来的指令，但它只执行第一个指令。到了下一个瞬间，它又会重新获取车辆当前状态，再次预测未来 2 秒并求解。这种机制被称为"滚动优化（Receding Horizon）"。

优势与局限：MPC 是目前公认性能上限较高的控制算法之一，能同时处理多变量耦合和硬约束，实现高精度的轨迹跟踪。但它的代价是较高的计算复杂度，对车载计算平台的算力提出了更高要求。

***深蓝AI*****4****—****总结：规控没有"最优解"**

纵观自动驾驶规划与控制算法的演进，我们可以看到一条清晰的脉络：

- 规划从离散的网格搜索（A\*）走向结合运动学的连续空间搜索（Hybrid A\*），再走向解耦的数学优化（EM Planner）。
- 控制从基于误差的简单反馈（PID），走向基于线性模型的最优反馈（LQR），再走向处理复杂约束的滚动预测优化（MPC）。

在这个领域，没有哪一种算法是万能的"最优解"。量产自动驾驶系统通常是这些经典算法的精妙组合：在低速泊车时使用 Hybrid A\*，在城市巡航时切换到 Lattice 或 EM 框架；纵向用 PID 兜底，横向用 LQR 保持稳定，在极限避障时激活 MPC。

当前，随着端到端大模型和强化学习的崛起，数据驱动的方法正在向规控领域渗透。但无论底层范式如何演进，安全、舒适、高效的物理约束永远存在，理解这些经典算法中蕴含的数学逻辑与工程折中，依然是每一位自动驾驶从业者和爱好者的必修课。

编辑｜阿豹

审核｜阿蓝

**参考资料：**

论文

1.Dolgov, D., Thrun, S., Montemerlo, M., & Diebel, J. (2008). Practical Search Techniques in Path Planning for Autonomous Driving. AAAI Workshop on Intelligent Driving.

2.Werling, M., Ziegler, J., Kammel, S., & Thrun, S. (2010). Optimal trajectory generation for dynamic street scenarios in a Frenet frame. IEEE ICRA 2010.

3.Fan, H., et al. (2018). Baidu Apollo EM Motion Planner. arXiv:1807.08048.

配图来源

•图1：Justin Milner, "A Visual Guide to the Software Architecture of Autonomous Vehicles", Medium.

•图2、图3：Boseong Jeon, "Gentle Introduction to Hybrid A Star", Medium.

•图4：Robotics Knowledgebase, "Trajectory Planning in the Frenet Space".

•图5、图6：Fan et al., "Baidu Apollo EM Motion Planner", arXiv:1807.08048, 2018.

•图7：Madhu Dev, "Tuning PID Controller for Self-Driving Cars", Medium.

•图8：Atsushi Sakai, "Linear Quadratic Regulator (LQR) Speed and Steering Control", PythonRobotics.

•图9：David Rose, "Vehicle MPC Controller", Medium.


**往期推荐** Recommend [![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpAhYq85CXTeKEXodjfiaIUHfDfa8hBib0502WDIBrslJKic68cZC5IiaicIGdXxcCzrBWkfDnacLHu51TWqpBIg0BezPN8zyV3CHEJc/640?wx_fmt=jpeg&from=appmsg#imgIndex=0)](https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247602525&idx=1&sn=179072d10ad35c9c441927d095c3e381&scene=21#wechat_redirect)**近五年谁在 Science Robotics 上发文最多？盘点全球顶尖机器人实验室**[![图片](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAKPXr0kicZddyXdPOg1Jm7tKusPLcWicG0ALpMqpjSHZxxsu45C13rzA4XZ2leKiaxG64fPqc9zIRIj8CR43YYibVy9ic8aRib3LUd8/640?wx_fmt=png&from=appmsg#imgIndex=0)](https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247602190&idx=1&sn=a9e9a29449a395f8c08f54f4c78fed06&scene=21#wechat_redirect)**3D目标检测经典算法全盘点：单目、双目、激光雷达****欢迎关注【深蓝AI】**持续分享人工智能领域前沿动态👇![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/943LxrS8cpCFreRWsn2fgjfEz7fB26oBpbfOsHK7zRA7xsBRS9mpSIvgQwOETOeicmb4PgKiby0nOGDo9ObI0JrvBflh4oibEdgwTEykKOSQ1w/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=16)
