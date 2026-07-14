---
title: 人形机器人运动控制 Know-How（飞书全文导出）
source_url: https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87
exported: 2026-07-14
author: RoboParty
lines: 2260
note: 用户提供的全文 Markdown 导出；内嵌图片路径为 Images_attachments/ 相对路径
---

# 人形机器人运动控制Know\-How

# 人形机器人运动控制发展趋势

![图片5\.png](Images_attachments/图片5.png)

![image\.png](Images_attachments/image%2026.png)

人形机器人运动控制在近两年得到了非常快速的发展，从传统运动控制范式（MPC）到最新的运控大模型范式（BFM）只用了短短三年，尽管如此，实现具身智能的道路前方依然充满挑战，正如[CMU石冠亚](https://zhuanlan.zhihu.com/p/1989611167883215960)老师所说的那样，**具身智能是一个硬件问题、数据问题、算法范式问题。**

**对于硬件来说**，什么样的人形机器人是符合具身智能终极需求的暂无定论，一种最可能的方案当然是造出和人类一样的人形机器人；另一种可能的方案是像波士顿动力那样突破仿人自由度的限制去做创新；也可以说在当前，宇树G1是一个面向科研场景下最好用的机器人。

**对于数据来说**，人形机器人在2025年末BFM范式的加持下，似乎也具有了大规模扩展数据集出现data scale的能力，由此便存在数据带来的问题，包括数据质量，数据采集，数据数量等等。

**对于算法来说**，必须承认强化学习在腿足机器人上的成功是促使人形机器人飞速发展的关键原因，但这只是起点，由于存在与物理世界交互的需求，大概率需要迭代出一个更符合Physical AI的算法范式而不是沿用一些LLM原本的东西（比如在当前的BFM范式中，采用监督强化学习还是无监督强化学习就是两个不同的路线）。

**另外，在evaluation和benchmark上也没有固定标准**，比如模仿学习筛选好坏数据的指标是什么，一种是人工筛选，一种是特权学习筛选。前者面临大量的工程量，后者可会面临多动作学习和单动作学习的冲突造成错误筛选。再比如符合什么样指标的机器人才是有用的机器人。

后续的篇幅中，我将从人形机器人运动控制算法的发展角度来讲述小脑的Know\-How，**从传统运动控制方法到强化学习的运动控制方法，每个方法将按照“方法原理\-基本代码\-方法局限性”的思路**去呈现，**也会介绍当前人形机器人最新的BFM范式**。本文绝大部分内容为个人的思考总结，由于个人思考可能存在不足，因此正如第一张图片描述中有参考文献，我会尽量在每个地方都注明参考，以供准确查阅，欢迎大家讨论（**PS：每一个方法背后我都尽量有作者/复现者的思考，而不仅仅是图文描述，希望能让读者常看常新**）。另外，本篇长文的重点在Know\-How，其示例代码更接近伪代码，如果需要一些完整代码和结构来复现，请参考[RoboParty官方开源地址](https://github.com/Roboparty)。

具身智能的未来需要更多开发者的加入和共建，这正是我们[RoboParty](https://roboparty.com/about)开源以及我撰写这篇文档的意义，如果这篇万字长文能给你一点点帮助，那我所做的工作就是有意义的。

# 人形机器人运动控制学习路线

这一章是在笔者写完整篇文章之后想到的，因为我觉得**在按照逻辑自下而上讲述人形机器人的运动控制之前，应该要让一些新手开发者知道怎么最快去上手人形机器人的运动控制**。正如马斯克在最新的访谈中提到AI时代面临的学习教育状况：从NASA到硅谷，巨头都在招募高中生，最根本的原因就是大学教育落后于AI的发展，当AI抹平了学历的鸿沟，更重要的是认知和学习效率。

如果我只做自下而上的介绍，那难免会跟从头上课类似——当你三个月后学会了所有理论，还是一头雾水，而那个时候可能新的问题和研究已经诞生。尽管这篇长文只是在讲Know\-How，笔者已经尽量不去深究其中的一些细节，但是一些必要的事情还是得按照自下而上的逻辑去呈现。**但就从开发者的角度，正如笔者开始入门人形也是从一个项目上手一样，更重要的事情应该是我们该如何开始，如何从实践中学习？因此这里分别给出Model base和Learning base的两条学习路线。**

## 传统运动控制学习路线

要学会传统运动控制方法，其实就是学会人形机器人的模型和求解。但传统控制算法由于开源少，理论门槛较高，确实需要花一定的时间在理论知识的学习上，笔者这里根据自己的经验，尽量让读者能够快速上手而不是深入了解。

1. 入门机器人学，推荐阅读中文教材《机器人建模和控制》，英文教材ETH的《Robot Dynamics Lecture Notes》。同时具备一定的代码阅读能力，包括C\+\+和python的代码，建立使用Linux系统。多看知乎，博客等相关内容，不要钻牛角尖，了解大概即可。

    1. 熟悉正逆运动学、正逆动力学、浮动基动力学模型

2. 入门人形机器人建模，推荐先看一遍本文的[传统运动控制方法](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-KbIWdfjwKoBff2xJrZKc8bjUn2d)，对传统控制部分有一个概念，知道Know\-How。了解**浮动基动力学模型**是怎么回事。

3. 安装动力学库[Pinocchio](https://stack-of-tasks.github.io/pinocchio/)，结合GPT，去[宇树github](https://github.com/unitreerobotics)找一个机器人的URDF模型文件，根据Pinocchio中的例子或者GPT学会以下事情：

    1. 导入人形机器人的URDF模型，看看里面包含哪些

    2. 了解人形机器人的正逆运动学，并通过Pinocchio\+GPT做一些例子

    3. 了解人形机器人浮动基动力学模型，看导入模型后的哪些变量是动力学模型里的，做几个正逆动力学的例子

4. 学会使用一个仿真环境，推荐Mujoco，学会将URDF导入mujoco，并结合Pinocchio在仿真环境中进行运动学，动力学分析

5. 入门优化控制，推荐看[CMU课程《optimal control》](https://www.youtube.com/watch?v=6rUdAOCNXAU)，课程里面包含很多，所以这阶段重点了解：

    1. 优化问题的构成，求解流程（最上层是什么，最底层是什么）

    2. 最简单的优化问题是什么，LQR，QP这两个有什么不同，了解MPC，知道MPC的特点和区别

    3. 了解现有的优化求解器都有哪些，比如OSQP、qpOASES等有什么区别

    4. 自己动手用python/C\+\+写一个QP优化问题，并至少用两个不同的求解器求解，并看看python/C\+\+的求解区别

6. 建模一个人形机器人的全身动力学，求解一个[WBC问题（参考本文）](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-LBLadYi1DoJp8axCqKecX6b4nzb)，整个问题考虑为QP问题，人为给定一个插值的期望轨迹，让人形机器人在mujoco环境中实现原地双脚站立，原地踏步，蹲起

7. 参考开源代码Qiayuan Liao的[legged\_control](https://github.com/qiayuanl/legged_control)或者ETH的[OCS2](https://leggedrobotics.github.io/ocs2/)，跑通他其中四足机器人的一些例子。

8. 熟悉MPC，熟悉WBC，看懂上面四足机器人的代码

9. 进阶系统学习优化控制理论，推荐MIT课程《[Underactuated Robotics](https://underactuated.csail.mit.edu/)》，ETH[《Optimal and Learning Control for Autonomous Robots》](https://arxiv.org/abs/1708.09342)。为了系统了解人形机器人建模和优化求解方法，推荐阅读综述*Wensing P M, Posa M, Hu Y, et al\. Optimization\-based control for dynamic legged robots\[J\]\. IEEE Transactions on Robotics, 2023, 40: 43\-63\. *重新回来看本文，也许会有不一样的收获。

## 强化学习运动控制学习路线

强化学习运动控制的上手路线就比传统容易很多，开源文档，代码框架，论文都比较容易看懂实现过程，但笔者还是建议对优化控制最好有一定的了解，可以看看本文的[最优化控制问题理论基础](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-Cff7dCOByoDUPGxqEUVcCMlznjm)和[深度强化学习运动控制方法](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-RvjodTB4noxjToxO6njcFeqKn1b)章节。

1. 了解强化学习和深度学习，这部分网上有太多好的参考文章都可以参考，教程也很多，理论细节可跳过

2. 了解腿足强化学习训练到部署的整个过程，推荐云深处科技这个小视频[入门课程](https://www.bilibili.com/video/BV1xKabz9E2d/?spm_id_from=333.337.search-card.all.click&vd_source=30216de2308cf451749cc50ccb881d29)。这个过程目的是看了能对腿足最终你能实现什么，以及大概的整个工程过程是什么有个帮助。

3. 可以选择阅读本文强化学习相关部分有个大概了解，知道现在主流的方法和框架大概有哪些。

4. 找一下腿足机器人强化学习的大量网上的文档和视频，学一学[Isaaclab](https://isaac-sim.github.io/IsaacLab/main/index.html)的安装和使用，这个东西是你的仿真训练环境，这过程的目的是要知道腿足强化学习在做一个什么事情。

5. 然后去找一个开源项目跑一跑，推荐的项目有且不包括[RoboParty](https://github.com/Roboparty/roboto_origin)\(最完整的，软硬件都有）、[Ziqi Fan](https://github.com/fan-ziqi/robot_lab)、[Unitree](https://github.com/unitreerobotics)、[Zitong Bai](https://github.com/zitongbai/legged_lab)、[Qiayuan Liao](https://github.com/HybridRobotics/whole_body_tracking)。这个过程目的是知道怎么训练，调试，哪些参数可以调，那些地方可以改。

6. 如果想做人形或者特定腿足的项目（比如人形模仿学习或者四足复杂地形行走），这个时候可以去找一些特定方向的开源的项目去跑了。同步进行包括但不限于阅读论文，与不同你感兴趣的方向的人交流，网上搜集等方法。也可以参考5中提到的开源仓库，还可以参考[sebastianstarke/AI4Animation: Bringing Characters to Life with Computer Brains in Unity](https://github.com/sebastianstarke/AI4Animation)。

# 人形机器人控制问题解决思路

这里主要讲述人形机器人的运动控制思路，我认为这个思路可以贯穿传统和learning的方法，甚至不局限于人形机器人，然后讨论人形机器人的Sim2Real问题是什么，以及人形机器人与常见的机械臂，无人机，仿真橡皮人有什么不同，因为正是这些不同给人形机器人的研究入门带来了难度，尤其这是传统运动控制方法比如MPC在人形上门槛高的主要原因之一（还有一个原因是没有好用的人形实时求解器），强化学习缩小了门槛（强化学习可以看作是一个更强大的非凸优化求解器），但理解他们的区别仍然能让我们对“为什么”有更好的直觉。

## 建模\+求解

考虑现在需要用手柄控制四轮小车以某个速度行驶，这里不讨论其他硬件细节，对于控制算法来说，思路如下：

1. **确定机器人的状态和控制输入**。在这里，四轮小车的状态是**车整体即质心的状态**和**每个轮子的状态**，详细地来说包括质心的位姿和速度，轮子的位置和速度甚至力矩。控制输入是四个轮子对应电机的位置/速度/力矩（如果是舵机那就是位置，如果是力矩电机那就可以到力矩，控制到力矩的意义在于我能让小车加减速更加平滑让你少晕车）。

2. **建立系统动态方程模型**。我们把状态记作向量X，控制输入记作向量U，此时我们需要写出下一时刻的状态和当前状态，控制输入的映射关系，即* ****Xt\+1 = F\(Xt, Ut\)***，这就是建模的过程。（不要小看小车的建模，你的模型应当包括转弯的情况）

3. **考虑约束和目标函数，构造一个正确的优化问题**。小车在实际行驶的时候可能面临很多约束条件，比如电机的速度有物理极限，电机的力矩有物理极限（不等式约束），以及动态方程模型也可以看作是我的约束条件之一（等式约束）。另外，我希望小车能按照我的指令速度前进，也就是我希望 ***Loss = \(Vcmd \- V\) ***为0，这是我的目标函数。目标函数\+模型\+约束，这就是最优控制问题（Optimal Control Problem，OCP）。

4. **选择合适的控制算法求解**。我们可以想办法构造为MPC问题，LQR问题，DDP问题等多种OCP问题的形式，并采用现成的很多OCP问题求解器求解出正确的U。

![image\.png](Images_attachments/image%201.png)

对于一个人形机器人的控制问题，我们也可以主要可以将其分为建模和求解两个过程：

![image\.png](Images_attachments/image%204.png)

1. **建立系统动态方程模型。**

    对于一个29自由度的人形机器人，他的状态 ***X ***是质心状态\+关节状态（e\.g如果考虑位置的话通常是7\+29维的向量，因为质心姿态通常用四元数来描述，单纯的三维[欧拉角](https://zh.wikipedia.org/wiki/%E6%AC%A7%E6%8B%89%E8%A7%92)存在万向锁和SO3空间插值不连续问题），他的输入 ***U*** 是29维的关节力矩/关节位置。对于传统控制来说，我们利用浮动基机器人动力学方程建立系统的动态方程模型，得到 ***Xt\+1 = F\(Xt, Ut\)。***

    而对于强化学习来说，该模型被仿真环境所取代，通常通过给定 ***Action（Ut） ***和 ***Xt***** **之后调用**step\(\)**函数就得到下一个状态 ***Xt\+1 ***，而这个step\(\)隐式地作为了我的动态方程模型。因此传统控制的gap主要来自于动力学模型的简化假设和参数，强化学习的gap主要来自于仿真环境动力学的精度。

> 可以参考ETH2016年实现四足机器人运动控制的一篇文章，从建模到求解有完整的公式的过程。Bellicoso C D, Gehring C, Hwangbo J, et al\. Perception\-less terrain adaptation through whole body control and hierarchical optimization\[C\]//2016 IEEE\-RAS 16th International Conference on Humanoid Robots \(Humanoids\)\. IEEE, 2016: 558\-564\.
> 
> 

2. **构造优化问题并求解**。

    对于传统控制来说，思路和小车一致，确定目标函数，等式约束和不等式约束，比如足底不打滑，足底接触地面时速度为0，关节速度力矩限幅等，然后采用不同的OCP方法如MPC，DDP和求解器求解，后文的Model Base部分也会详细介绍。

![image\.png](Images_attachments/image%2027.png)

    对于强化学习来说，本质也是一种OCP方法，不过是和传统方法中的DDP方法类似，将问题组织为一个基于马尔可夫决策过程的动态规划问题，区别在于，强化学习摆脱了对模型的依赖，并且靠着离线优化的特性得到动作分布，使其成为一个强大的非凸优化求解器。

![image\.png](Images_attachments/image%209.png)

## Sim2Real问题

Sim2Real的问题是具身智能Physical AI的最大特点，也是难点所在。对于传统方法来说，Sim2Real问题来自于你的模型是否精确，建模的假设是否能在现实中满足，对于强化学习方法来说，来自于你仿真环境的精度（比如Gym为了加快并行仿真效率，对碰撞体的碰撞计算就会进行简化处理，再比如我物理引擎该怎么仿真足端接触地面时候的动力学，是用spring\-damper model还是time\-stepping integrators model，再比如隐式积分和显示积分的区别）。

虽然当前Locomotion在硬件成熟，并克服如下图中所示的问题后，已经没有什么gap的存在，但正如开篇所说，这只是我们的起点，对于诸如多接触，仿真数据的可用性等复杂但真正有用的场景可以认为还没有一个很好的解决办法，毕竟，人形机器人如果达到人们期望的终极目标，他的意义就是100，如果没有，那么就是0，不如任何专用的机器人，我们并不希望人形机器人只是会跳舞。

> 仿真引擎下的多接触加速优化：Tassa Y, Erez T, Todorov E\. Synthesis and stabilization of complex behaviors through online trajectory optimization\[C\]//2012 IEEE/RSJ International Conference on Intelligent Robots and Systems\. IEEE, 2012: 4906\-4913\.
> 
> 

![image\.png](Images_attachments/image%2033.png)

## 人形机器人与其他机器人的区别

之所以单独讲讲这一个部分，是因为我发现有些人对人形机器人和其他机器人之间的本质区别并不是很了解，比如：

- 为什么我们不能把动捕演员重映射后的机器人动作直接让机器人去执行，而是必须要经过一次强化学习？

- 为什么无人车，无人机，四足机器狗，工业机械臂的运动控制发展那么早那么快，但是人形却连走路都是问题？

- 为什么仿真人形橡皮人角色几乎在很早就能够上天入地，但是人形机器人还是发展这么慢？

### 运动学可行和动力学可行

**一句话，动捕重映射之后的数据只是运动学可行，并不是动力学可行。**

想象一下你需要让一个滑块做物理运动，如果你只给合适的位置和速度，滑块的物理运动并不是确定的，因为在加速度未知的情况下，你仍然不知道滑块会如何运动。

对于动捕重映射后的轨迹来说，除非机器人实际所处的环境和动捕时的环境一样\+机器人构型也和人类一样\+转换和仿真过程中没有gap，才大概率不会出问题。运动学可行意味着机器人只知道每个关节应该怎么运动，并且确实也考虑到了实际机器人的关节范围，但并不知道接触地面时如何调整自身的力来保持平衡不打滑，并不知道重映射后的动作加速度是不是实际电机可以执行出来的。

此时你可能会觉得，可是对于机械臂，我只要做一个运动学逆解（IK），并不需要动力学逆解（ID）就可以让真实的机械臂运动，对于四足机器狗，运动学逆解之后也能走几步不会倒。这就引出了人形机器人与不同机器人的区别，也就是下面的内容。

### 人形机器人和其他机器人

**一句话，人形机器人是一个高维度非线性、动力学突变、低静态稳定裕度的浮动基机器人系统。**

- **对比无人车：高维度非线性。**对于自动驾驶的无人车来说，从实现基础功能的角度，运动控制是相对简单很多的。四轮车并没有很高的维度，比如你不用在意机器人在空间中的Z轴位置，你完全可以把它看成是2D的平面运动，普通的车并不用考虑上楼；四轮车也没有很高的非线性耦合，因为连接质心的只有四个轮子，而且是并联；在稳定程度上，汽车并不会因为你推一把就翻倒。而对于人形机器人来说，通常都有很高的维度（看看你的Obs维度），关节串联程度也非常高，随之而来的非线性耦合也很大，第一个关节的误差会影响后续所有串联关节。

- **对比无人机：动力学突变**。对于四旋翼无人机来说，其运控也是非常成熟的，尽管运动范围在整个空间内，基本的飞控在实现上也并不存在很大的瓶颈。不过最关键的一点是无人机不会产生动力学突变，就算考虑上空气动力学的复杂非线性，其系统动力学的微分平坦性也是被严格证明的，参考文章[四旋翼无人机微分平坦性](https://www.zhihu.com/question/322812159/answer/99234688037)。而对于人形机器人来说，很显然的一点是关节的运动和质心的运动之间并不像无人机一样存在直接的驱动关系，人形机器人是靠接触反作用力运动的，是靠动力学突变来产生运动的。

- **对比四足机器人狗：低稳定裕度**。对于四足机器狗来说，它已经和人形机器人很接近了，都是浮动基，都是高维非线性，都是动力学突变，那么区别就在于它是两条腿还是四条腿。而人形机器人的难点就在于此，两条腿用来走路，另外两条手用来操作，不仅让稳定裕度下降，而且如果四足机器狗不考虑操作，还有极强的越野优势，但对于人形来说，一定得做到兼具越野优势\+操作的精度优势\+全身协调优势，否则越野不如四足、操作不如机械臂、移动操作不如四足\+机械臂。

- **对比机械臂：浮动基系统**。对于机械臂和人形的区别，最大的区别就是固定基座和浮动基座之间的区别了。机械臂能发展那么快的原因就因为它是固定基座的，不存在整体质心的概念，只存在末端抓持器的概念，这样一来其关节运动和末端抓持器运动之间存在更加直接，且无动力学突变，稳定裕度拉满的关系。这也解释了上一节说到的只要做一个运动学逆解机械臂就能运动，因为gap小，因为固定基座的原因导致其稳定裕度拉满，机械臂根本就不会发生倾倒。也因此，早几年时期的工业机械臂大多都是位置控制就够用了，并不需要考虑到关节力矩输入（除非你要按摩，擦玻璃）。

因此，在运动控制这个领域，人形机器人的基础运控是发展最为缓慢的，一方面是因为有以上的高难度特质，而这些高难度特质让你构造这个最优控制问题的时候就很复杂，问题的复杂导致求解复杂，由于人形机器人又要追求实时性，另一方面就提出了对人形机器人实时求解器的需求，而现有的可用求解器并不能很好满足需求，**采用传统运动控制方法不能在不简化模型的基础上考虑多接触模型时利用现有求解器能很好求解**。传统人形MPC面对的是一个周期很长投入产出比很低的工程瓶颈，而波士顿动力可以说把传统方法求解做到了极致，从技术框架到求解器的内部都付出了巨大的工程努力才得到一个离线规划\+在线优化的demo结果，并没有带来很高的产出，我们也只能从其workshop和blog中获得一些技术框架信息[波士顿动力workshop总结](https://www.zhihu.com/people/nike-20-31/posts)。

### 人形机器人和橡皮人

**一句话，仿真里的所有的数据和状态都是可知且准确的。**

![image\.png](Images_attachments/image%207.png)

计算机图形学对橡皮人的研究其实在2010年前后就已经有非常不错的成果，利用传统方法一个优化问题就实现倒地爬起，跑跳等功能。橡皮人身处仿真环境中，主要的优势有以下三点：

1. 完全可知且准确的状态，不用估计。在现实中要通过GPS，IMU，编码器，力矩传感器等各种传感器得到的带有误差的状态数据在仿真中是真实准确的，在现实中需要靠积分或者差分或者足底里程计状态估计得到的速度数据，在仿真中可以直接拿到。

2. 橡皮人的构型结构更接近人，并且力矩无限制。橡皮人每一个关节位置通常有三个自由度，并且是一个球体关节，而现实人形机器人面对同样的情况时通常采用串联三个关节来解耦，并且存在电机力矩的限制。

3. 橡皮人不存在sim2real gap问题。在这波人形机器人浪潮之前，人形机器人很难有好用稳定的硬件产品，因为背后是整个工程团队和供应链的努力，其实那个时候能在仿真中用传统方法调出人形机器人后空翻也并非难事，但谁都不觉得能在自己实验室的那个硬件上调出什么后空翻。

因此，随着人形机器人本体硬件、强化学习算法和Sim2Real的进步，可以预料到橡皮人的成果将会不断在人形机器人上复现，[Human2Humanoid](https://arxiv.org/pdf/2403.04436)（[phc](https://openaccess.thecvf.com/content/ICCV2023/papers/Luo_Perpetual_Humanoid_Control_for_Real-time_Simulated_Avatars_ICCV_2023_paper.pdf)橡皮人）到如今的[BFM\-zero](https://arxiv.org/pdf/2511.04131)（[meta motivo](https://metamotivo.metademolab.com/)橡皮人），至少这个过程直到现在也没有结束，橡皮人的研究仍然领先于人形机器人。**但不可否认的是具身智能的研究在未来仍然需要重视认识到Physical AI的特殊性，在已有的比如橡皮人的算法基础上做更多理解Physical的范式的迭代和创新**。

# 人形机器人技术框架路线展望

在传统控制的时期，人形机器人/腿足机器人的路线经过十几年的发展已经有了一个稳定的雏形，是一个包含 " Navigation \- Planning \- Controller \- Estimator " 的完整框架，每一部分都在各司其职。

**Navigation**得到符合人形机器人基本运动学约束的路径轨迹和可通行区域；

**Planning**阶段通常是去解决一个轨迹优化问题（TO\)，在更细致地层面得到下发的期望轨迹，比如落脚点约束，速度加速度平滑，全身协调的轨迹；

**Controller**阶段通常需要考虑动力学约束，目标是的得到Dynamic Feasible的控制输入，即最终的关节力矩。人形机器人以往的运动控制，通常也是在主要研究这个阶段或者包括Planning阶段做的事情。

**Estimator**用于估计出现实中实物机器人无法直接得到或者噪声很大的状态，比如质心速度，位置，角速度，角度等等。

笔者认为除了波士顿动力以外，公开研究领域中，整个技术路线框架最为清晰的当属ETH的团队，它们给出的框架如下：

> 参考ETH文献：
> \[1\] Bellicoso C D, Gehring C, Hwangbo J, et al\. Perception\-less terrain adaptation through whole body control and hierarchical optimization\[C\]//2016 IEEE\-RAS 16th International Conference on Humanoid Robots \(Humanoids\)\. IEEE, 2016: 558\-564\.
> 
> \[2\] Grandia R, Jenelten F, Yang S, et al\. Perceptive locomotion through nonlinear model\-predictive control\[J\]\. IEEE Transactions on Robotics, 2023, 39\(5\): 3402\-3421\.
> 
> 

![图片28\.png](Images_attachments/图片28.png)

然而，在强化学习的影响下，似乎**端到端（end2end）**的方案更加受到大家的青睐，包括ETH在使用了强化学习之后，也从端到端的blind locomotion研究到perceptive locomotion甚至是端到端的navigation，并且可见的结果是端到端的效果大部分时候确实比以前的传统运控好很多。

在可见的未来，人形机器人还会在端到端上出很多漂亮的成果，那么未来的人形机器人技术路线会是什么样的，或者说真正能让人形机器人成功的技术路线会是什么样的，笔者这里也不能得出明确的结论，可以当一家之言，权当讨论。

1. **思考一：端到端方案》全身遥操作采集数据 \-\> VLA**

    让我们从第一性原理出发思考，人形机器人成功，意味着人形机器人走进家庭，并且实现通用化，按照目前已有的端到端的方案，似乎在大模型上看到了所谓的 “通用化” 路线，结合目前世界上各家公司的情况，如果LLM这条路线是可以成功的，那么一种明确的路线是如波士顿动力公司展示的方案：

![图片29\.png](Images_attachments/图片29.png)

    先利用已有的MPC技术实现人形机器人的全身遥操作，然后利用人遥操作人形机器人去采集数据，进而完成VLA的数据收集，等收集了足够多的数据后就可以实现一定程度的“泛化”。

    可是，这个遥操作的MPC技术除了波士顿动力以外其他团队是没有的，因为其背后的工程量和积累即便是很专业的人看来，也实在太大。幸运的是，随着模仿学习\+强化学习技术在人形机器人上的发展，类似于TWIST或者GAE/SONIC的方案已经可以在一台性能好用的人形机器人上实现全身的遥操作，不过目前由于强化学习梯度本身会和数据集分布有关的原因，不管是TWIST的Teacher\-student方案，还是GAE/SONIC的大数据方案，都仍然存在会有部分动作尤其是in the air的动作难学会的现象，这个也是我觉得下一步模仿学习的重点。**（之所以大数据下in the air的动作不好学，正是因为腿部的动力学运动是突变的，即想找到的解并不是连续的。读者可以看到手臂的遥操作其实没什么问题）**

    不过这条路线即使用MIMIC的方法绕开了MPC实现了很好的遥操作，也会面临数据收集效率实在太大的问题（毕竟机械臂的VLA遥操作收集的数据效率已经很慢了，人形遥操作就更慢了）。

    **因此，思路一的问题在于，能不能用learning做到很好的遥操作，更重要的是，有没有更好的数据收集方案？**

2. **思考二：端到端方案》上下肢分离 \-\> 上肢机械臂VLA**

    也有一种可行的路子是上下肢分离的方案，因为人形的纯强化学习运控基本都聚焦在下肢，那么是否能够用小模型保证下肢的鲁棒性，不用太考虑跟踪特定轨迹，然后解耦上肢出来像机械臂一样去做VLA的方案，这样甚至有可能能把机械臂VLA领域的模型和数据都能无缝迁移过来。

    这样做的意思无非是在说，腿只用管鲁棒行走，上楼梯，走梅花桩，不用管上肢怎么动，不用特意去跟踪特定轨迹。这种方案乍一看合理，但是这意味这同一个动力学模型中，我跑了两套算法，在理论上，人形所有关节的运动都是耦合的，我只是在控制层面进行了分离，而理论上来看，人形机器人应当是一个全身模型，除非你能在理论上用类似于零空间的方式解耦出关节。

    这样做必然存在的结果是，要么我们选择牺牲人形机器人特有的空间自由度，来防止摔倒，要么我们选择让双腿尽量cover掉任何情况，包括弯腰屈膝拿东西。对于前一种方案，如果牺牲，那么我认为这样的人形机器人更适合进工厂，而不适合进入家庭，人形机器人的意义也会大打折扣；对于后一种方案，最关键的问题是我们怎么保证双腿cover掉任何情况？纯靠locomotion的探索估计不太行，那就得用模仿学习或者无监督学习（本文最后提到的BFM框架中其中一个就是无监督学习），如果用模仿学习，就会存在数据分布是否足够，训练时候怎么保证动力学突变带来的分布偏移问题？如果用无监督学习，由于采样分布的原因，目前看来同样会存在动力学突变带来的有些动作难以掌握的问题。

    **因此，思路二的问题在于，我们要选择牺牲空间自由度，还是努力让双腿cover掉一切？好像怎么选都不太好。**

3. **思考三：解耦方案》Navigation \+ Planning \+ Controller**

    让我们从强化学习的诞生出发，强化学习方法本质上来源于动态规划，马尔可夫决策过程，可以在后续文章看到，无论是在ETH的分类里，还是在CMU optimal control的课程里，都将强化学习归为一种最优控制的方案，正如那张传统优化控制和RL的[对比图](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-JSz3dVNqfoavNkxP9zhcvzU8n0P)就来自ETH的最优控制讲义，其本质上是建模了分布，这里不进行详细讨论，在后续的[最优化控制问题理论基础](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-O3nnddZX7oNlruxXriocOjsSnxf)中以及[深度强化学习理论基础](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-QcwSdzbOzoF5uRxdFdicoq35nUf)中都会说明。

    既然强化学习是一种最优控制理论，MPC也是，二者的区别只在于强化学习是一个更强大的非凸优化求解器，并且强化学习只是建模了分布。MPC也可以选择端到端的把 planning\+controller 融合在一起，但最后仍然解耦的控制，除了求解频率的问题以外，planning专注于规划，controller专注于控制也是一种更好的选择。

    另一方面来说，人形机器人之所以觉得能成功，现在看到除了它是通用AGI里最合适的物理实体以外，还有一个很重要的原因是，在强化学习\+模仿学习的影响下，人形机器人结构和人类构型相似，而人类的数据集非常多，就算只是动捕，也比其他机器人多很多，因此如果有机会，一定要利用人的优势，人自身能产生大量数据的优势。而根据目前的发展也能看到，基于blind的模仿学习在大量数据集的加持下通过SONIC/GAE/TWIST这样的方案证明其大概率比perception end2end的方案更有可能接近AGI，为什么？

    因为如果同样通过数据来实现大模型，进而实现通用的controller，blind的方案可以立刻马上就利用到现成已有的大量人的本体数据集，而如果要考虑视觉，就很难找到了。换句话说，就算是比收集效率，从零开始收集blind人体运动数据集，也比收集带视觉的人体运动数据集要快得多。

    而假设我们现在能通过SONIC/GAE/TWIST这样的方案实现一个通用的controller（至少目前看上去很有希望），那我能不能在planning的层面上做一个VLM，在VLM和controller之间用一套retarget算法来打通，这样我的VLM只用收集人类数据，而不用收集机器人的数据，效率会快非常多，在controller通用的情况下，VLM只用考虑生成smpl格式的人类数据，然后经过一个retarget算法，得到机器人的参考数据。与此同时，VLM上层的Navigation也可以做一个单独的端到端导航VLN，从而实现**Navigation \+ Planning \+ Controller**的解耦架构**。其中，Controller只用通过强化学习训练，相比VLA不需要那么多数据，Planning用VLM的方式收集人类数据集，生成smpl格式的期望轨迹，实时retarget之后给Controller。**

![image\.png](Images_attachments/image%2024.png)

    尽管这样做仍然会面临一些挑战，但是至少解耦了不同环节，是一条成本更低的路线。

4. **思考四：关于轮足人形机器人**

    四轮足机器人之所以打败四足机器人，原因确实是因为轮足这个形态本身并不是仿生的，而是人类创造出来的形态，这个形态能够兼顾足和轮的优势，既能保持速度优势又能保证越野优势。

    对于机器狗来说，我认为越野优势就是其最大的优势，但是对于人形来说，无论是足式还是轮足，其最大优势仍然需要在通用移动和通用操作上，因此如果只想打造轮足人形，只管他的越野优势其实不如四轮足。

    但是从通用的角度来说，如果通用需要的一定是Whole body的数据，那么轮足人形收集数据又只能回到效率很低的思路一：遥操作收集。从现有的大量数据以及收集效率来看，最好的方案就是在仿人和工程之间找到一个平衡。当然，如果通用不一定需要数据，比如无监督学习如果能够成功，那也许轮足人形就能走上前台，只不过现在看来，一个性价比高的仿人机器人是最有可能实现通用化的。

5. **思考五：如果通用基于LLM，那具身智能真的需要更多的Physics。**具身智能必须和物理世界产生交互，要么仿真更物理，要么算法框架更物理，只有懂这个世界，才能迈向真正的具身智能。

# 传统运动控制方法（Model Base\)

本章将介绍人形机器人传统运动控制的几个主要方法，由于时间和精力关系，只能挑出几个代表不同时代的主流方法进行介绍，一方面是让读者更能了解人形机器人技术的发展过程，另一方面是作者并不认为传统控制就是完全需要被抛弃的，其中有很多重要的model框架思想和直觉是值得建立和参考的，比如后续能想到DreamWaq和Attention落足点估计方法就带有这种直觉。正如ETH在2017年介绍RL以及CMU的Optimal Control课程中就已经将其作为最优化控制方案的进阶一样，**Physical AI在未来需要更多model的直觉。**

![图片1\.png](Images_attachments/图片1.png)

![图片2\.png](Images_attachments/图片2.png)

## 最优化控制问题理论基础（OCP）

推荐看过的两个最优化讲的最好的学习课程：
[MIT Russ Tedrake：Underactuated Robotics](https://underactuated.csail.mit.edu/)
[CMU Zac Manchester: Optimal Control](https://www.youtube.com/watch?v=6rUdAOCNXAU)

![规控脉络图\.png](Images_attachments/规控脉络图.png)

在开头的《建模\+求解》中，我们把人形机器人的控制问题解决思路大致分为了建模\+求解两个部分，在经过建模之后我们都可以得到一个最优控制问题（Optimal Control Problem, OCP），该问题通常由目标/损失函数\+等式约束\+不等式约束构成，其中建模得到的动态方程 ***Xt\+1 = F\(Xt, Ut\) ***是最重要的等式约束。

由于目前的主流方法中一般都会采用离散化进行求解，因此这里对连续求解的变分法、HJB方程或者连续优化之类的理论不涉及，否则就是很大的一个部分内容，该文章的目的只在于Know\-How整个机器人的控制架构，因此这里贴出一些比较好的最优控制及求解器系列文章链接。

> 最优控制及求解器系列文章：
> https://www\.zhihu\.com/people/yang\-tai\-wen\-24/posts
> 
> https://zhuanlan\.zhihu\.com/p/594769376
> 
> https://zhuanlan\.zhihu\.com/p/629131647
> 
> https://zhuanlan\.zhihu\.com/p/450696809
> 
> Acados求解器例子参考：包括DDP、OCP、SQP、QP
> https://deepwiki\.com/acados/acados/7\.2\-python\-examples
> 
> 

接下来我们可以将OCP问题变成离散化的形式以方便代码编写求解，即把时间切分成N步，相邻步之间采用某种积分器（Euler/RK4/implicit）来近似连续动力学。

![image\.png](Images_attachments/image%2019.png)

**从离散的OCP问题出发，主流的解决方法通常有两条路线**，一种是利用建模得到的动态方程递推，考虑为一个马尔科夫决策过程，进而可以把整个问题转化为动态规划（Dynamic Programming, DP）下的**DDP问题**。另外一种方法是把所有时刻的状态和控制拼接成一个大变量，不存在step by step的概念，转化为**NLP问题**。

![image\.png](Images_attachments/image%2012.png)

1. **离散OCP问题 \-\> DDP问题 \-\> iLQR问题 \-\> LQR子问题：**

![图片4\.png](Images_attachments/图片4.png)

2. **离散OCP问题 \-\> NLP问题 \-\> SQP问题 \-\> QP子问题：**

![图片3\.png](Images_attachments/图片3.png)

> **关于MPC**：MPC（Model Predictive Control, 模型预测控制）是构造OCP问题的一种形式，特点是**有限时域、在线滚动执行**，即每次得到的不仅是第K时刻的控制律Uk，而是**得到未来一段时间的控制律Uk\-Uk\+n，然后每次只取Uk作为真正当前时刻的输入**，这样做的好处是整个模型能起到一定的预测效果，可以提前防止发散，鲁棒性更强。至于求解MPC，因为只是OCP的一种形式，所以可以用基于动态规划的DDP，也可以用基于规模化问题的NLP求解。
> 
> 

![image\.png](Images_attachments/image%2025.png)

> **关于OCP问题到NLP问题的离散化**：在之前离散化时提到把时间切分成N步，相邻步之间采用某种积分器（Euler/RK4/implicit）来近似连续动力学。实际在转为NLP问题的过程中，根据问题的需要和为了问题构造的稀疏性，通常有一些专门的方法，包括collocation（配点法）和shooting（打靶法），在这里也不过多叙述，感兴趣的朋友可以参考上面贴出的[最优控制及求解器系列文章链接](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-JiWxd0GZVo2RaXx3xMzcdPajn5e)。
> 
> 

**重要！！！**：在传统控制时代，不同时期诞生了一些主流好用的控制方法，在后续的传统方法讲述中，我会把**大量篇幅放在建模和构造优化问题**的角度，即**得到 “模型\+约束\+损失函数的OCP形式”** ，在OCP优化问题的求解侧不再做过多的叙述，比如在模型上，后续将不止停留在 ***Xt\+1 = F\(Xt, Ut\)***** 这种抽象形式层面，而是更加具体地写出人形机器人不同模型下的动力学方程**。

更加详细的优化控制理论和求解内容可以去参考我在本节中贴出的课程和链接，也可以去不同类型求解器的官网去看具体的求解例子。

![图片12\.png](Images_attachments/图片12.png)

本章后续将对人形机器人传统优化控制各个主流的建模和控制方案进行介绍，即将要登场的各个模型关系如下：

![图片11\.png](Images_attachments/图片11.png)

## 线性倒立摆模型\+零力矩点算法（LIP\+ZMP）

### 方法原理（建模\+约束\+损失函数）

记得当初最早入门人形机器人的时候，除了机器人学和控制方法论的书籍以外，第一次知道人形机器人该怎么建模控制的就来自于日本教授梶田秀司（Shuuji Kajita）的著作[《仿人机器人》](https://www.tup.tsinghua.edu.cn/booksCenter/book_02098601.html)：

![图片6\.png](Images_attachments/图片6.png)

![image\.png](Images_attachments/image%2014.png)

回到人形机器人的开端，从静态的角度来看，判断一个人形机器人是否平衡，我们只需检查其[重心投影](https://zhida.zhihu.com/search?content_id=265600708&content_type=Article&match_order=1&q=%E9%87%8D%E5%BF%83%E6%8A%95%E5%BD%B1&zhida_source=entity)（Projection of Center of Mass）是否落在[支撑多边形](https://zhida.zhihu.com/search?content_id=265600708&content_type=Article&match_order=1&q=%E6%94%AF%E6%92%91%E5%A4%9A%E8%BE%B9%E5%BD%A2&zhida_source=entity)（Support Polygon）内部，就像人类站立时，身体的重心必须在双脚之间，才能不倒。而在动态情况下（例如机器人行走），重心在不断移动，这时仅靠重心投影就无法准确判断稳定性。**于是需要建立运动模型和找到能刻画动态稳定性的指标。**

1. **基于线性倒立摆（LIP）\+ 零力矩点（ZMP）模型建模。**

    LIP模型把人形机器人看成一个如下图的倒立摆，并基于以下假设：机器人质心高度恒定；机器人角动量变化忽略（假设上半身姿态固定）；假设地面为刚性，足端不滑动；只考虑水平平面的运动（x,y两个方向解耦）。

    ZMP 是[地面反作用力](https://zhida.zhihu.com/search?content_id=265600708&content_type=Article&match_order=1&q=%E5%9C%B0%E9%9D%A2%E5%8F%8D%E4%BD%9C%E7%94%A8%E5%8A%9B&zhida_source=entity)（Ground Reaction Force, GRF）的合力作用点，也就是地面对机器人脚底施加的所有支撑力的“合力位置”。如果机器人身体产生的惯性力和重力导致的合力矩在这个点上恰好为零，那么该点就是零力矩点。ZMP可以在支撑多边形外部，也可以在内部，只要落在内部，机器人就能利用足底反作用力处于稳定状态，如果ZMP在外部，地面无法再提供足够的反力矩，机器人就会开始倾倒。

![图片7\.png](Images_attachments/图片7.png)

![image\.png](Images_attachments/image%2020.png)

![图片8\.png](Images_attachments/图片8.png)

2. **约束和目标函数。**

    LIP的假设约束已经在动力学方程内部所体现，剩下的约束就是ZMP的边界约束，即ZMP要在支撑多边形的内部，由于我们将x，y已经解耦，因此分别给Px，Py一个限幅即可。

3. **整个OCP问题。**这里就是一个简单的QP（线性二次型）问题，因为损失函数是二次型，约束都是线性，非常快就能求出解（就像对于二次函数你可以直接写出最低点的解析式一样）。

![图片9\.png](Images_attachments/图片9.png)

### 基本代码

```Python
# --- 模型 ---
def define_LIP_model(params):
    omega = sqrt(params.g / params.z_c)
    dt = params.dt
    A_d = [[cosh(omega*dt), sinh(omega*dt)/omega],
           [omega*sinh(omega*dt), cosh(omega*dt)]]
    B_d = [[1 - cosh(omega*dt)],
           [-omega*sinh(omega*dt)]]
    return A_d, B_d


# --- 约束 ---
def define_constraints(params):
    return {
        'zmp_x_min': params.support_x_min,
        'zmp_x_max': params.support_x_max,
    }


# --- OCP构造和求解 --- ocp求解器接口通常如下
def solve_ocp(A_d, B_d, constraints, x0, x_ref, u_ref, N, Q, R, Qf):
    # 构造优化变量X, U，初始状态x0固定
    # 添加动力学约束：x_{k+1} = A_d x_k + B_d u_k
    # 添加输入约束：zmp_x_min <= u_k <= zmp_x_max
    # 构造代价函数：状态误差+控制误差+终端误差
    # 调用求解器求解并返回最优解
    pass  # 伪代码，细节根据具体求解器实现
```

### 方法局限性

受制于早期浮动基动力学理论研究不成熟，电机也以位置控制为主，因此读者可以发现LIP模型之所以简单，是因为有很多严格的假设，而正因为简单导致了很多局限性（比如固定质心高度这一点，导致不能跳跃，不能奔跑，不抗干扰，只能实现准静态的运动），机器人的状态也不包含加速度，控制输入也是足端位置而不是力矩，其实就是把人形机器人当成一个能运动的机械臂，而不是一个真正的浮动基系统。尽管如此，优美的解析式仍然能得到很多不错的demo效果，在2000年代ASIMO的demo水平就算是放在现在也能让人称赞。

## 弹簧负载倒立摆模型\+虚拟模型控制算法（SLIP\+VMC）

### 方法原理（建模\+约束\+损失函数）

日本的人形机器人研究在80\-90s已经领先了世界，但随着力控技术的完善，美国的人形机器人研究在90s后期也逐渐走上道路，这将成为后续主流传统运控方法的开端，而这个新的理论来自于那时的[MIT Leg Lab](http://www.ai.mit.edu/projects/leglab/people/people.html), 该lab的创始人就是后来波士顿动力的创始人[Marc Raibert](http://www.ai.mit.edu/projects/leglab/people/mxr.html)教授，这个实验室也走出了很多如今很有名的教授，比如[Jerry Pratt](http://www.ai.mit.edu/people/jpratt/jpratt.html)，[Russ Tedrake](http://hebb.mit.edu/people/russt/)。

1. **弹簧负载倒立摆（SLIP）\+ 虚拟模型控制（VMC）模型建模。**

    SLIP在Marc Raibert教授的[《Legged Robots that Balance》](https://www.google.com.hk/books/edition/Legged_Robots_that_Balance/EXRiBnQ37RwC?kptab=editions&sa=X&ved=2ahUKEwi478euhPmRAxXGn68BHXxTH-AQmBZ6BAgIEAs&cshid=1767775376984376)书中把机器人建模为带有负载且足端与质心通过虚拟弹簧连接的倒立摆模型，如下图所示。

    另一方面，借助于力控技术的发展，Jerry Pratt提出VMC算法通过控制虚拟弹簧产生的力和落足点来实现机器人的动态稳定。具体来说，将腿接触地面开始到离开地面的过程称为**支撑相**，将腿摆动过程称为**摆动相**，在支撑相时控制支撑足端产生合适的弹簧虚拟力，在摆动相时控制摆动足端到达合适的落脚点。**SLIP\+VMC首次实现了人形机器人的动态控制，这是点足机器人系统能够稳定的算法来源。**

![图片10\.png](Images_attachments/图片10.png)

2. **约束和OCP问题**。由于这里已经把机器人质心状态x和质心合外力/力矩建立了直接的关系，具体来说，可以看到状态中的质心高度和质心姿态（roll,pitch）与质心合外力/力矩建立了直接的PD关系，质心水平方向的线速度（Vx,Vy\)与落脚点Xf之间建立了直接的启发式函数关系。因此我们可以不在这里去解一个优化问题，直接通过这两个启发式子求解控制输入。

    **重要提醒！**：虽然图中可以看到控制输入为质心合外力/力矩，尽管这个合外力矩并不能直接作用于质心，但是这个合外力矩可以通过**机器人静力学中的虚功原理**构造[合外力矩与关节力矩之间的雅可比映射关系](https://www.cnblogs.com/caster99/p/4733988.html)，进而得到关节力矩。

![image\.png](Images_attachments/image%2021.png)

### 基本代码

这里以一个**点足式单腿机器人动态稳定控制**为例，包括hip\_roll, hip\_pitch, knee三个自由度，支撑相控制质心姿态和质心高度，摆动相控制落足点。

```Python
# =========================
# 状态
# =========================
# p_com, v_com        # 质心位置、速度
# p_foot              # 足端位置
# R_body, omega_body  # 机体姿态与角速度

# =========================
# 支撑相：SLIP + VMC
# =========================
def stance_control(p_com, v_com, p_foot, R_body, omega_body, params):
    # --- SLIP 几何 ---
    leg_vec = p_com - p_foot
    l = norm(leg_vec)
    e = leg_vec / l
    l_dot = dot(e, v_com)

    # --- 质心高度控制（虚拟弹簧-阻尼） ---
    F_leg = params.k * (params.l0 - l) * e \
            - params.d * l_dot * e

    # --- 姿态控制（虚拟力矩） ---
    tau_body = -params.kp * log(R_body) \
               - params.kd * omega_body

    # --- 合力 / 合力矩 ---
    F_com = F_leg + [0, 0, -params.m * params.g]

    return F_com, tau_body


# =========================
# 摆动相：落足点控制
# =========================
def swing_control(p_com, v_com, params):
    # Raibert 启发式落脚点
    x_f = p_com.x + v_com.x * params.Ts / 2
    y_f = 0
    z_f = 0
    return [x_f, y_f, z_f]


# =========================
# 全控制循环
# =========================
if contact_state == "STANCE":
    F_com, tau_body = stance_control(...)
    joint_torque = map_task_to_joint(F_com, tau_body)

elif contact_state == "SWING":
    p_foot_des = swing_control(...)
    joint_torque = foot_position_control(p_foot_des)
    
# 对于如何检测contact_state一般有两种方案：
# 1.足底力传感器反馈（检测法）
# 2.通过电机关节力矩反馈利用虚功原理估计出接触力（估计法）
```

### 方法局限性

尽管SLIP\+VMC的方案没有严格按照优化控制思路去进行，但是其启发式的结论打开了腿足机器人动态运动的大门，它让后来的人形机器人或者四足机器狗能实现奔跑，跳跃等更多动态的动作，而不只是局限在LIP\+ZMP这样准静态且没有鲁棒性的运动中。

然而，这样启发式的方案存在很多缺点，比如没有考虑太多约束，参数调整很玄学，落足点无法考虑多约束等等。当然，在此之后，随着人形机器人浮动基动力学的进一步完善，真正的属于人形机器人的动力学方程和优化控制方案将马上出现。

## 全身动力学模型\+全身运动控制/任务空间逆动力学优化控制算法（WBD\+WBC/TSID）

### 方法原理（建模\+约束\+损失函数）

本节介绍人形机器人浮动基动力学，并以此建立人形机器人的全身动力学模型（Whole Body Dynamic, WBD）。

在得到全身动力学方程的基础上，结合约束和损失函数，只考虑当前时刻进行泰勒展开线性化化简，可以得到一个线性二次型（QP）形式的最优控制问题（OCP\)，该OCP问题在人形机器人的运动控制中通常被称为全身运动控制（WBC），**这里的WBC在之后通常被用来作为人形运控低层的高频控制器**（是的，由于人形机器人过于高维和复杂且没有太好的实时求解器，后续的研究一般采用高\-低层解耦控制，高层只负责优化得到接触外力/力矩，低层再优化得到关节力矩）。线性二次型（QP）问题我们在之前有提到，他对于求解非常友好，正如你可以直接写出二次函数的最小值解析式一样，不用去数值迭代，这就是WBC能够充当低层高频控制器的原因。

> 对于WBC，更准确的说法应该是任务空间逆动力学优化（TSID），这里只是为了不纠结其叫法。TSID可以参考文章：Del Prete A, Nori F, Metta G, et al\. Prioritized motion–force control of constrained fully\-actuated robots:“Task Space Inverse Dynamics”\[J\]\. Robotics and Autonomous Systems, 2015, 63: 150\-157\.
> 
> 

**说明**：针对WBC/TSID中的多任务和约束有不同的处理方式，这里不做详细描述和区分，以最一般（Weight\+Linear system）的形式进行建模求解，目的是让读者了解整个WBC的建模求解过程，知道Know\-How。（但我会附上相关链接）

![图片13\.png](Images_attachments/图片13.png)

1. **全身动力学模型（WBD）**

    一个完整的人形机器人动力学模型的状态通常是6\+N（实际工程通常是7\+N，角度用四元数表示）\+ 6\+N，包括质心的位置和姿态，线速度和角速度，以及关节的角度和角速度。控制输入通常是关节加速度、接触外力、关节力矩，虽然最终给到机器人的只是关节力矩，但是在这里关节加速度和接触外力仍然作为辅助的决策变量求出，因为他们都是加速度层面的值。

![图片14\.png](Images_attachments/图片14.png)

2. **对于多个任务（目标函数）的处理：加权（Weight）**

    我们为什么要求解优化问题，除了有很多约束以外，另外一个很关键的事情是人形机器人的任务空间可能有很多task需要去完成，换句话说，我可能有质心需要跟踪的期望轨迹，也可能同时有手末端或者足末端的期望轨迹，另外机器人自身还要保持平衡，该怎么平衡这个任务。研究上通常有两种办法，一种是显而易见的加权（Weight），另外一种是零空间（Null Space）。在这里，我们采用Weight的方式：

![图片15\.png](Images_attachments/图片15.png)

    > 如果想要了解另外一种方式，即**零空间（Null Space）**的读者，可以参考下面几篇文章：
    > \[1\] Dietrich A, Ott C, Albu\-Schäffer A\. An overview of null space projections for redundant, torque\-controlled robots\[J\]\. The International Journal of Robotics Research, 2015, 34\(11\): 1385\-1400\.
    > 
    > \[2\] Herzog A, Rotella N, Mason S, et al\. Momentum control with hierarchical inverse dynamics on a torque\-controlled humanoid\[J\]\. Autonomous Robots, 2016, 40: 473\-491\.
    > 
    > \[3\] Saab L, Ramos O E, Keith F, et al\. Dynamic whole\-body motion generation under rigid contacts and other unilateral constraints\[J\]\. IEEE Transactions on Robotics, 2013, 29\(2\): 346\-362\.
    > 
    > \[4\] Escande A, Mansard N, Wieber P B\. Hierarchical quadratic programming: Fast online humanoid\-robot motion generation\[J\]\. The International Journal of Robotics Research, 2014, 33\(7\): 1006\-1028\.
    > 
    > 

3. **对于约束的处理：线性化（Linear）**

    对于等式约束和不等式约束，由于**WBC只考虑当前瞬间这个时刻的问题，除了决策变量以外的任何系数都可以看作常数**，因此自然可以很方便地将动力学方程和其他等式/不等式约束都写成线性的。这个部分可以在之后的[QP形式的OCP问题](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-EV7GdUb1aonntexzBTRcok0Gngh)看到。

    当然尽管如此，就算是考虑当前时刻，只是线性的约束也存在近似的问题，因为本身系统在时域上是非线性的，那么有没有可能既保留非线性，同时又可以用QP这种线性二次型的形式求解？可以的，这就是将动力学方程转为为CLF（控制李雅普诺夫函数），或者将约束转化为CBF（控制障碍函数）工作的原因。

    > CLF和CBF是非线性约束的表达式，可是为什么能被QP这种需要线性约束的求解形式求解？因为所谓的线性本质上只要求对控制输入U线性，而不要求对状态也线性，即可以是 ***dX = f\(X\) \+ u*** 。感兴趣的读者可参考下面的文章：
    > Galloway K, Sreenath K, Ames A D, et al\. Torque saturation in bipedal robotic walking through control Lyapunov function\-based quadratic programs\[J\]\. IEEE Access, 2015, 3: 323\-332\.
    > 
    > 

4. **QP形式的OCP问题：二次型加权损失函数\+线性等式/不等式约束**

![图片16\.png](Images_attachments/图片16.png)

### 基本代码

1. **加载URDF，通过动力学库Pinocchio和casadi变量符号化得到动力学信息**

    ```Python
    import pinocchio as pin
    import pinocchio.casadi as cpin
    import casadi as ca
    
    
    def build_symbolic_dynamics_from_urdf(urdf_path, contact_frame_names):
        """
        从 URDF 构建基于 Pinocchio + CasADi 的符号动力学模型
        """
    
        # ------------------------------------------------------------
        # 1. 从 URDF 构建 Pinocchio 模型
        # ------------------------------------------------------------
        model = pin.buildModelFromUrdf(
            urdf_path,
            pin.JointModelFreeFlyer()   # 浮动基座人形机器人
        )
        data = model.createData()
    
        nq = model.nq   # 广义坐标维度
        nv = model.nv   # 广义速度 / 加速度维度
    
        # ------------------------------------------------------------
        # 2. 创建 CasADi 符号变量
        # ------------------------------------------------------------
        q   = ca.SX.sym("q", nq)
        dq  = ca.SX.sym("dq", nv)
        ddq = ca.SX.sym("ddq", nv)
    
        # ------------------------------------------------------------
        # 3. 将 Pinocchio 模型转换为 CasADi 版本
        # ------------------------------------------------------------
        cmodel = cpin.Model(model)
        cdata  = cmodel.createData()
    
        # ------------------------------------------------------------
        # 4. 质量矩阵 M(q)
        # ------------------------------------------------------------
        M = cpin.crba(cmodel, cdata, q)          # Composite Rigid Body Algorithm
        M = ca.symmetrize(M)                     # 保证数值对称性
    
        # ------------------------------------------------------------
        # 5. 偏置项 h(q, dq) = C(q,dq)dq + g(q)
        # ------------------------------------------------------------
        h = cpin.nonLinearEffects(cmodel, cdata, q, dq)
    
        # ------------------------------------------------------------
        # 6. 接触 Jacobian 及其时间导数项
        # ------------------------------------------------------------
        Jc_list = []
        Jc_dot_dq_list = []
    
        for frame_name in contact_frame_names:
            frame_id = model.getFrameId(frame_name)
    
            # 接触点 Jacobian
            Jc = cpin.getFrameJacobian(
                cmodel, cdata, q, frame_id, pin.LOCAL_WORLD_ALIGNED
            )
            Jc_list.append(Jc)
    
            # J_dot * dq 项
            Jc_dot_dq = cpin.getFrameJacobianTimeVariation(
                cmodel, cdata, q, dq, frame_id, pin.LOCAL_WORLD_ALIGNED
            )
            Jc_dot_dq_list.append(Jc_dot_dq)
    
        J_c = ca.vertcat(*Jc_list)
        J_c_dot_dq = ca.vertcat(*Jc_dot_dq_list)
    
        # ------------------------------------------------------------
        # 7. 构造 CasADi Function（供 WBC / QP 调用）
        # ------------------------------------------------------------
        dyn_fun = ca.Function(
            "dynamics_terms",
            [q, dq],
            [M, h, J_c, J_c_dot_dq],
            ["q", "dq"],
            ["M", "h", "Jc", "Jc_dot_dq"]
        )
    
        return dyn_fun
    ```

2. **WBC的OCP问题（QP问题）求解**

    ```Python
    def wbc_qp_solver(q, q_dot, ddx_des, contact_info, robot_model):
        # 1. 计算动力学相关项
        M = robot_model.mass_matrix(q)
        h = robot_model.bias_forces(q, q_dot)
        J_c = robot_model.contact_jacobian(q, contact_info)
        J_c_dot_q_dot = robot_model.contact_jacobian_dot_times_q_dot(q, q_dot, contact_info)
        J_i = robot_model.task_jacobian(q)
        J_i_dot_q_dot = robot_model.task_jacobian_dot_times_q_dot(q, q_dot)
    
        # 2. 构造决策变量 u = [ddq, lambda, tau]
        # 维度信息由robot_model提供
        n_ddq = M.shape[0]
        n_lambda = J_c.shape[0]
        n_tau = robot_model.n_joints
    
        # 3. 构造等式约束
        A_dyn = np.hstack([M, -J_c.T, -robot_model.selection_matrix().T])
        b_dyn = -h
        A_contact = np.hstack([J_c, np.zeros((n_lambda, n_lambda + n_tau))])
        b_contact = -J_c_dot_q_dot
        A_eq = np.vstack([A_dyn, A_contact])
        b_eq = np.concatenate([b_dyn, b_contact])
    
        # 4. 构造不等式约束 (摩擦锥 + 力矩限制)
        A_ineq, b_ineq = robot_model.build_inequality_constraints(contact_info)
    
        # 5. 构造目标函数
        W_i = np.eye(J_i.shape[0])  # 任务权重矩阵，示例用单位矩阵
        # 目标：min ||J_i * ddq + J_i_dot_q_dot - ddx_des||_W^2
        H = np.zeros((n_ddq + n_lambda + n_tau, n_ddq + n_lambda + n_tau))
        H[:n_ddq, :n_ddq] = J_i.T @ W_i @ J_i
        g = np.zeros(n_ddq + n_lambda + n_tau)
        g[:n_ddq] = J_i.T @ W_i @ (J_i_dot_q_dot - ddx_des)
    
        # 6. 调用 QP 求解器（例如 OSQP, qpOASES, cvxpy等）
        u_star = solve_qp(H, g, A_eq, b_eq, A_ineq, b_ineq)
    
        # 7. 返回关节力矩作为控制输入
        ddq_star = u_star[:n_ddq]
        lambda_star = u_star[n_ddq:n_ddq + n_lambda]
        tau_star = u_star[n_ddq + n_lambda:]
    
        return tau_star
    ```

### 方法局限性

人形机器人浮动基全身动力学模型的完善，以及单一时刻下WBC优化问题的求解，已经将整个人形机器人传统运动控制的脉络搭建清楚。但如果只用WBC的这种单帧QP特殊形式进行优化，相当于没有考虑模型的非线性变化，这就是很多控制算法中使用模型预测控制算法（MPC）的原因。MPC很强大，因为其考虑的是一段时间内的控制问题，如果说瞬时控制的方法一定会经历超调再稳定，那么MPC就可以做到提前预判到超调的出现，从而提前抑制。

![图片17\.png](Images_attachments/图片17.png)

**MPC的关键特点在于**：在t时刻对系统状态进行采样，采样后计算未来一段很短的时域\[t,t\+T\]内使最优化问题成本最小的控制策略轨迹，并且最终只会执行控制策略中的第一步，然后在下一个采样点进行重复。

![图片18\.png](Images_attachments/图片18.png)

一个很显然的想法当然是“人形机器人建立全身动力学模型\+MPC构造OCP多帧优化问题”，而如果真的这么做，尤其在10\-15年前人们想到的时候，发现这样做要么你的控制频率会被压的很低（实时求解不过来），要么你的MPC预测窗口很窄（这样又会破坏MPC效果）。

**因此在人形机器人运动控制上，一种有效的方式是“人形机器人建立简化动力学模型\+MPC构造OCP多帧优化问题”作为高层低频控制器（30\-100Hz\)，“人形机器人全身动力学模型\+WBC构造单帧QP优化问题”作为低层高频控制器（500\-1000Hz\)，从而有了下面两种在10\-20年代最常见的控制方式。**

## 单刚体动力学模型\+凸模型预测控制算法\+WBC（SRBD\+Convex MPC\+WBC）

### 方法原理（建模\+约束\+损失函数）

在2016年的时候，波士顿动力放出了其人形机器人Atlas跳舞和跑酷的视频，在21年其技术负责人[Scott Kuindersma](https://scholar.google.com.hk/citations?user=tKSxEgQAAAAJ&hl=zh-CN&oi=ao)的讲座上，透露了当时所使用的模型是单刚体动力学（Single Rigid Body Dynamic, SRBD），具体可以参考我之前的[blog文章](https://www.zhihu.com/people/nike-20-31/posts:)。

![图片19\.png](Images_attachments/图片19.png)

在2017\-18年，伴随着MIT [Sangbae Kim](https://scholar.google.com.hk/citations?user=y4E4buEAAAAJ&hl=zh-CN&oi=sra) 组的几篇mini cheetach的四足机器人狗的文章，不仅开源了低成本腿足力矩电机的方案，为后来市场上的腿足力矩电机设计提供了很多基础，而且开源了四足机器狗的MPC控制算法。在其机器狗的MPC控制算法中，采用了单刚体动力学模型对四足机器狗进行建模，并在几个假设的基础上，**将原本多帧的非线性MPC问题转化为一个多帧的线性MPC问题（又称为Convex MPC），原本复杂的非线性OCP问题一下子又变成了可以快速求解的QP（线性二次型）问题，从而能够做到显著提高MPC的控制频率，实现高动态且鲁棒的四足机器狗行走**（上一代运控方案通常基于VMC的控制方案，当时的四足机器人狗走的还没有那么灵活稳当）。

![图片20\.png](Images_attachments/图片20.png)

> MIT cheetah开源地址：https://github\.com/mit\-biomimetics/Cheetah\-Software
> 
> 参考文章：
> \[1\] Wensing P M, Wang A, Seok S, et al\. Proprioceptive actuator design in the mit cheetah: Impact mitigation and high\-bandwidth physical interaction for dynamic legged robots\[J\]\. Ieee transactions on robotics, 2017, 33\(3\): 509\-522\.
> 
> \[2\] Di Carlo J, Wensing P M, Katz B, et al\. Dynamic locomotion in the mit cheetah 3 through convex model\-predictive control\[C\]//2018 IEEE/RSJ international conference on intelligent robots and systems \(IROS\)\. IEEE, 2018: 1\-9\.
> 
> \[3\] Bledt G, Powell M J, Katz B, et al\. Mit cheetah 3: Design and control of a robust, dynamic quadruped robot\[C\]//2018 IEEE/RSJ International Conference on Intelligent Robots and Systems \(IROS\)\. IEEE, 2018: 2245\-2252\.
> 
> 

接下来仍然对人形机器人单刚体动力学模型进行分析建模，之前说过，如果对人形机器人进行全身动力学建模并且考虑MPC算法会对工程实时求解提出比较大的挑战，因此通常会在MPC的环节进行简化建模分析：

1. **单刚体动力学模型（SRBD）**

    单刚体动力学模型，就是把系统整体当做一个刚体系统来分析，因此系统的状态只有刚体质心的状态q\_com，系统的输入只有接触外力F。因此我们这里分析建模也不用建立什么浮动基的全身动力学方程，实际上只要浮动基动力学方程的前6行即可。

![图片21\.png](Images_attachments/图片21.png)

2. **多帧优化问题变为线性的原因：单刚体动力学模型到线性单刚体动力学模型的假设**

    虽然除了动力学以外的其他约束我可以变为线性，但是在动力学方程约束上，单纯的单刚体动力学模型还不足以当作一个线性模型，在此基础上还需要加入一些假设：

![图片22\.png](Images_attachments/图片22.png)

3. **Convex MPC形式的OCP问题**

    至此，我们可以考虑一个线性的动力学模型\+线性的约束\+MPC优化问题。从而可以得到一个Convex MPC问题，得到的Convex MPC问题符合QP的特殊形式，可以采用QP求解器进行快速求解。

![图片23\.png](Images_attachments/图片23.png)

4. **MPC\+WBC框架：高层Convex MPC \+ 低层 QP\-WBC**

    可以注意到在刚刚的**Convex MPC中得到的控制输入是足端接触外力，还不能直接驱动机器人的关节电机**。因此如果此时要驱动关节电机，有两种方式，一种是利用VMC中提到的虚功原理，即利用雅可比矩阵已知外力求得关节静力矩，但是该方法存在明显缺点，即会使模型进一步线性化，削弱MPC的作用，并且没有考虑其他任何约束，导致在实际执行过程中会产生更大的偏差。

    因此通常选择在低层加一个WBC，**将Convex MPC得到的足端外力F作为WBC全身动力学模型中外力的期望轨迹带入**，通过求解一个高频的QP问题得到合适的关节力矩。

    > 尽管在实际上，采用第一种方法（虚功原理）和第二种方法（QP\-WBC）在实际机器人表现上也许并没有太大区别，因为线性单刚体的假设已经很强，但是QP\-WBC仍然相比虚功原理在理论上有更好的解。该方法来源于MIT cheetah的WBIC论文：
    > 
    > Kim D, Di Carlo J, Katz B, et al\. Highly dynamic quadruped locomotion via whole\-body impulse control and model predictive control\[J\]\. arXiv preprint arXiv:1909\.06586, 2019\.
    > 
    > 

![图片24\.png](Images_attachments/图片24.png)

### 基本代码

这里展示Convex MPC \+ QP WBC的伪代码，用来清晰展示整个框架的工作过程：

```Python
'''
Convex MPC (50–200 Hz) def convex_mpc_step()
    ↓  足端接触力参考 F_ref
QP-WBC (1kHz)          def wbc_qp_step()
    ↓  关节力矩 τ
Robot
'''
def convex_mpc_step(x0, x_ref_traj, contact_schedule):
    """
    Single Rigid Body Convex MPC
    Decision variable: contact forces over horizon
    """

    # ----------------------------------------
    # Parameters
    # ----------------------------------------
    N  = MPC_HORIZON
    dt = MPC_DT

    # ----------------------------------------
    # State: x = [c, dc]
    # Input: u = [f_1, ..., f_nc]
    # ----------------------------------------
    A, B = build_com_dynamics(dt)

    # Stack prediction model
    A_bar, B_bar = build_prediction_matrices(A, B, N)

    # ----------------------------------------
    # Cost
    # ----------------------------------------
    Q = diag([w_c, w_c, w_c, w_dc, w_dc, w_dc])
    R = w_f * eye(3 * num_contacts)

    H = B_bar.T @ Q @ B_bar + block_diag(R, ..., R)
    g = B_bar.T @ Q @ (A_bar @ x0 - x_ref_traj)

    # ----------------------------------------
    # Constraints (friction, normal force)
    # ----------------------------------------
    G, h = build_contact_constraints(contact_schedule)

    # ----------------------------------------
    # Solve QP
    # ----------------------------------------
    U_star = solve_qp(H, g, G, h)

    # Only apply first-step contact force
    f_ref = U_star[:3 * num_active_contacts]

    return f_ref

def wbc_qp_step(q, dq, f_ref, contact_state):
    """
    Whole-Body Control QP
    Decision variable:
        u = [ddq, tau, f_c]
    """

    # ----------------------------------------
    # Get dynamics from Pinocchio
    # ----------------------------------------
    M, h = compute_dynamics(q, dq)
    Jc, Jc_dot = compute_contact_jacobian(q, dq, contact_state)

    # ----------------------------------------
    # Decision variable
    # ----------------------------------------
    ddq  = Variable(nv)
    tau  = Variable(nj)
    f_c  = Variable(3 * num_contacts)

    # ----------------------------------------
    # Equality constraints
    # ----------------------------------------
    constraints = []

    # Full-body dynamics
    constraints += [
        M @ ddq + h == S.T @ tau + Jc.T @ f_c
    ]

    # Contact acceleration constraint
    constraints += [
        Jc @ ddq + Jc_dot @ dq == 0
    ]

    # ----------------------------------------
    # Inequality constraints
    # ----------------------------------------
    constraints += friction_cone_constraints(f_c)
    constraints += torque_limits(tau)

    # ----------------------------------------
    # Cost function
    # ----------------------------------------
    cost = 0.0

    # Track MPC contact force
    cost += norm(f_c - f_ref, W_f)

    # Regularization
    cost += norm(ddq, W_ddq)
    cost += norm(tau, W_tau)

    # ----------------------------------------
    # Solve QP
    # ----------------------------------------
    solve_minimization(cost, constraints)

    return tau.value

while robot_is_running:

    # ----------------------------------------
    # State estimation
    # ----------------------------------------
    q, dq = get_robot_state()

    # ----------------------------------------
    # Low-frequency Convex MPC
    # ----------------------------------------
    if mpc_clock.tick():
        f_ref = convex_mpc_step(
            x0=get_com_state(q, dq),
            x_ref_traj=com_reference,
            contact_schedule=gait_plan
        )

    # ----------------------------------------
    # High-frequency WBC
    # ----------------------------------------
    tau = wbc_qp_step(
        q, dq,
        f_ref=f_ref,
        contact_state=current_contacts
    )

    # ----------------------------------------
    # Send torque
    # ----------------------------------------
    send_torque_command(tau)
```

### 方法局限性

Convex MPC \+ QP\-WBC这套方案已经能让机器人有很强的动态性能，尤其对于四足机器狗来说，由于本身腿部的质量相比于质心就不大，这套算法的出现已经能让四足的Locomotion能够作为一个稳定的产品进行工作，也因此，大部分的四足机器狗产品直到强化学习的出现其Locomotion的算法才发生变化。

但对于人形来说，就算部署这套算法，也只是解决了行走的问题，而且由于**单刚体仍然存在很多假设**的问题，当我们想要在上面扩展，比如考虑规划落足点，考虑加入视觉，加入手臂考虑操作和全身协调运动的时候，一切就都变得不可行了，正是因为极致的假设优化，导致这套算法如果不做任何其他工作的话，就只适用于行走。

因此，**对于人形机器人来说，需要一个更加完整的模型，该模型应该要能规划全身的运动，包括手臂和腿，而且机器人不仅应该要能保持全身平衡，在面对操作任务时也要能协调全身尽力去完成任务**。由此，随着算力的发展和人形硬件的提升，质心动力学模型成为了下一个更好的选择。

## 质心动力学模型\+非线性模型预测控制算法\+WBC（CD\+NMPC\+WBC）

### 方法原理（建模\+约束\+损失函数）

同样在2021年波士顿动力Atlas技术负责人[Scott Kuindersma](https://scholar.google.com.hk/citations?user=tKSxEgQAAAAJ&hl=zh-CN&oi=ao)的讲座上，讲到2016年使用的是单刚体模型之后，对后来波士顿动力新发布的视频进行介绍的时候，讲到他们的MPC是如何从MPCV1\.0过渡到MPCV2\.0的，其中这个MPCV1\.0指的就是单刚体动力学模型，MPCV2\.0指的就是质心动力学模型（Centroid Dynamic, CD），具体可以参考我之前的[blog文章](https://www.zhihu.com/people/nike-20-31/posts:)。

![image\.png](Images_attachments/image%2032.png)

人形机器人的质心动力学最早在2013年的时候被提出，文章中指出**系统的动能不能完全靠外力对质心动量的影响来刻画，还应包括了各个连杆对质心的影响，因此，质心动力学考虑的是外力\+连杆对质心的共同作用，与此同时，质心动力学忽略了连杆之间的相对运动，这是它和完整动力学的主要区别。**

换句话说，我们现在分析人形机器人的模型，仍然只对其质心进行分析，但是现在不仅要考虑整体合外力对质心的影响，还要考虑各个连杆对质心的影响，不过由于我们忽略关节力的作用，我们考虑各个连杆对质心影响的方式是将其作用效果投影到质心上，这个投影的映射关系是质心动力学中一个很重要的映射矩阵，被称为**质心动量矩阵**（Centroid Momentum Matrix，CMM），该矩阵保证了为什么质心动力学是可行的。

**【理解质心动力学的核心！！！：读者可以理解为我们考虑关节，但是只考虑连杆时变的惯量对质心的影响，而不考虑关节动力学，因此不考虑关节力矩。因而我的整个状态中同时存在质心位置速度\+关节角度，但是我的控制输入中不存在关节力矩，仅仅包含接触外力和关节速度，而CMM保证了为什么质心动力学是可行的。这和单刚体模型是不一样的，SRBD会直接认为没有关节连杆。】**

尽管CMM不直接参与之后的公式推导，但是它是我们证明能够这样做的原因，因为CMM揭示了关节状态和质心状态之间严格的映射关系。

![image\.png](Images_attachments/image.png)

更复杂的模型意味着我们要从之前的线性MPC算法转为采用一般的非线性模型预测控制算法（Nonlinear Model Predictive Control, NMPC），针对NMPC问题的求解也要回到一开始在介绍[优化控制原理](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-RVGkdFGh9obE0PxbfiOcTodbnyd)那节中的一般方法，即同样的，你可以选择DDP或者NLP的方式去求解整个NMPC问题。

> 参考文献：Orin D E, Goswami A, Lee S H\. Centroidal dynamics of a humanoid robot\[J\]\. Autonomous robots, 2013, 35\(2\): 161\-176\.
> 
> 

1. **质心动力学建模（CD Model）**

    在质心动量矩阵CMM的保证下，我们就可以进一步考虑关节但是又不会过于复杂。**这次我们分析的对象仍然只是质心，但是状态包含质心位置速度\+关节角度，控制输入是接触外力和关节速度。**

![图片25\.png](Images_attachments/图片25.png)

2. **NMPC问题构建（状态方程\+约束\+目标函数）**

    NMPC的动力学方程推导简单（如上图），但是可以看到其动力学方程是非线性的形式 ***dX=f\(X,U\) ***，这代表着后续MPC不仅求解速度下降，处理难度增大，并且我们要求解的是一个非线性MPC问题。但可见的是，该NMPC问题不仅能带有非线性的预测的能力，并且**该算法能够让人形机器人同时做到完整操作任务、落足点任务、质心跟踪任务的同时还能协调全身平衡。**

![图片26\.png](Images_attachments/图片26.png)

3. **NMPC问题的求解过程**

    如果我们现在已经构建了NMPC问题，那么接下来的求解就可以回到[优化控制原理](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-RVGkdFGh9obE0PxbfiOcTodbnyd)小节介绍的一般方法去求解，你可以直接开始在这一步就利用求解器，也可以自己线性化化简得到SQP/iLQR问题之后求解。

4. **NMPC\+WBC求解框架**

    **NMPC得到的是期望接触力，期望关节速度和优化之后的期望状态轨迹**，WBC拿到这些之后当作自己的目标轨迹建立全身动力学去求解QP\-WBC即可，在WBC这里，我们又回到了瞬时时刻的优化问题，因此所有约束都是线性的，最终得到的QP形式的问题是可以很快求解的，读者可参考之前的[全身动力学模型章节](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-FIWhdIyD4oMP6oxS7KhcXztNnXf)。

### 基本代码

这里给出NMPC\+WBC框架的伪代码，针对质心动力学模型或者单刚体动力学模型的腿足MPC问题求解，推荐参考ETH的开源框架OCS2，尽管OCS2中没有WBC的内容，但是其对理解整个代码级别的建模到优化的求解过程是非常有利的。

```Python
# 初始化机器人状态
x_current = get_current_state()  # 包含质心位置、速度、姿态角、角速度、关节角
q, dq = extract_joint_states(x_current)

while True:
    # 1. 高层 NMPC 求解（周期较长，约几十到百毫秒）
    # 输入：当前状态x_current，期望轨迹x_ref, u_ref
    x_pred, u_pred = NMPC_solve(x_current, x_ref, u_ref)

    # 取NMPC求解的第一步控制输入：足端接触力和关节速度
    f_desired = extract_contact_forces(u_pred[0])
    qdot_desired = extract_joint_velocities(u_pred[0])

    # 2. 低层 WBC 求解（高频，1kHz以上）
    # 输入：当前关节状态(q, dq)，期望接触力f_desired，期望关节速度qdot_desired
    M, C, g = compute_dynamics_matrices(q, dq)
    Jc, Jc_dot = compute_contact_jacobians(q, dq)

    # 构建QP问题
    # 决策变量: ddq, f
    # 目标：最小化 ||f - f_desired||^2 + ||ddq - ddq_desired||^2 + ||tau||^2
    # 约束：动力学方程 M ddq + C dq + g = tau + Jc^T f
    #      摩擦锥约束，关节力矩限制，接触约束等

    H, g_vec = build_objective(f_desired, ddq_desired=None)
    A_eq, b_eq = build_dynamics_constraints(M, C, g, Jc, dq)
    A_ineq, b_ineq = build_physical_constraints()

    ddq_star, f_star = solve_qp(H, g_vec, A_eq, b_eq, A_ineq, b_ineq)

    # 根据求解结果计算关节力矩
    tau_star = M @ ddq_star + C @ dq + g - Jc.T @ f_star

    # 3. 下发低层控制命令
    send_joint_torque_command(tau_star)

    # 4. 状态更新
    x_current = get_current_state()
```

### 方法局限性

老实说，基于质心动力学的人形机器人如果做好的话，已经能完成不少问题了，在不考虑复杂交互接触的情况下，可以比目前的RL更能兼顾操作精准度\+平衡性\+落脚点选择，但是读者可以看到模型复杂之后，如果要实现质心动力学的人形机器人控制需要什么东西。首先你需要一个好的人形机器人硬件，然后你要看懂论文（记得当初刚开始看人形论文看几遍讨论几遍都是看不懂的，远远不如现在learning的方法框架清晰），接着你要学习理论会推导公式（各个模型的假设，优化控制理论，甚至是高维空间的理论，否则你都不知道某个的偏导是怎么来的），然后你要会写代码（这里面要先构建出代码级别的建模，然后需要结合自动微分工具和求解器写你的算法），最后你要调试仿真和实物（就算是MPC也会存在sim2real gap），你会经历很多工程的难点，但最后可能就是实现一个走路或者慢动作开门，到最后看着波士顿动力的成果很多时候就算是一个实验室也只能望而却步。

当然另一方面来说，传统在实时求解上来看是有瓶颈的，正如Atlas花了很多成本和时间做了MPC的工作，在目前也转为了基于learning的方法，最新消息是他们用自己的MPC作为遥操作的运控算法，采集真机数据之后去做VLA的训练，这里面的工程量和得到的收益是不是匹配的是最大的问题。

## 人形机器人的状态估计

### 方法原理

做状态估计这个事情的需求和背景是，**机器人的一些状态是我们不能直接观测到的**，在之前的伪代码中，可以看到的是，我经常需要得到质心的速度作为反馈，有时候还需要判断我的足端是否触地，以便及时切换到支撑相算法。

1. **判断足端是否触地**

    如何判断足端触地一般有两种方法，一种是直接在足端用力传感器检测力的大小，但这种办法成本比较高，并且力传感器的噪声是很大的，另外一种办法是直接通过关节力矩估计足端反作用力，这种办法最好建立在你的腿足机器人是直驱关节的基础上（QDD），好在现在力矩电机的能力越来越强，通常是可以做的。

    对于关节力矩估计足端反作用力的方法，最简单，最直接的方法是利用虚功原理得到关节力矩和足端反作用力的静力学关系，即之前提到的雅可比矩阵，这种方法不一定会非常及时，但是在你打印出估计的力变化图像的时候，还是非常明显的，对于不太需要精准控制足端力的场景，比如行走奔跑之类的场景是够用的。

![image\.png](Images_attachments/image%2023.png)

    除此以外，有很多接触检测的论文可以参考，比如因为动力学方程中是有接触外力这一项的，很多论文以此为接触检测的模型，结合算法对接触力进行估计。

    > 接触估计算法参考文献：
    > \[1\] Camurri M, Fallon M, Bazeille S, et al\. Probabilistic contact estimation and impact detection for state estimation of quadruped robots\[J\]\. IEEE Robotics and Automation Letters, 2017, 2\(2\): 1023\-1030\.
    > 
    > \[2\] Brubaker M A, Sigal L, Fleet D J\. Estimating contact dynamics\[C\]//2009 IEEE 12th International Conference on Computer Vision\. IEEE, 2009: 2389\-2396\.
    > 
    > 

2. **质心速度估计**

    对于质心速度的估计方法，最经典的组合是足底里程计\+IMU结合进行线性卡尔曼滤波：

    **足底里程计**就是利用质心速度和关节速度之间的运动学关系，即雅可比矩阵，已知关节速度可以得到质心速度，但是**噪声很大**，因为腿部的动力学会频繁突变。IMU能返回的值包括角速度和线加速度，IMU通常安装在质心位置，因此**线速度可以通过加速度积分得到，但是IMU的误差会累积**，一个最直接的原因是芯片周围的温度会升高。

    前者方差大，后者偏差累计，为了减小噪声，消除漂移，通常将二者的数据融合进行卡尔曼滤波。

![图片27\.png](Images_attachments/图片27.png)

    > 关于足端速度估计和线性卡尔曼滤波，以及更进一步的非线性卡尔曼滤波这部分网络上有非常多写的很好的可参考的资料。其中非线性卡尔曼滤波EKF和线性卡尔曼滤波KF的关系就像是iLQR和LQR的关系一样，非线性的就是在当前时刻泰勒展开线性化处理求解。更本质的是，KF/EKF和iLQR/LQR虽然二者一个是观测器，一个是控制器，但他们的性质是极其相似的，甚至解的形式都是一样的，背后的原因是因为二者本身存在[特殊的对偶关系](https://www.zhihu.com/question/641137647/answer/3376125099)。
    > 
    > 

    

### 基本代码

这里给出足底里程计\+IMU的线性卡尔曼滤波质心速度估计的伪代码：

```Python
x = zeros(6)          # [p, v]
P = eye(6) * 1e-2

Q = diag([1e-4]*3 + [1e-3]*3)   # process noise
R = diag([1e-3]*3 + [1e-2]*3)   # measurement noise

def kalman_step(x, P, a_imu, Rwb, p_foot_w, p_foot_b, dt):

    # ---------- Prediction ----------
    A = [[I, dt*I],
         [0,    I]]

    B = [[0.5*dt*dt*Rwb],
         [dt*Rwb]]

    c = [0.5*dt*dt*g,
         dt*g]

    x_pred = A @ x + B @ a_imu + c
    P_pred = A @ P @ A.T + Q

    # Base position from foot contact
    p_base_meas = p_foot_w - Rwb @ p_foot_b

    # Finite difference velocity
    v_base_meas = (p_base_meas - x[:3]) / dt

    z = concat(p_base_meas, v_base_meas)
    
    H = eye(6)

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ inv(S)

    x_new = x_pred + K @ (z - H @ x_pred)
    P_new = (eye(6) - K @ H) @ P_pred

    return x_new, P_new
```

### 方法局限性

对于状态估计来说，其实可以作为一个完整的课题或者完整的一个内容来做，因为状态估计在传统机器人控制领域太重要了，我们建模的很多状态是我们不能通过什么传感器直接得到的，或者单靠传感器得到的可能有大的误差。由于笔者并不是专门搞状态估计的，所以只能在这里放一章把做过的说一说。

对于传统的状态估计方法，本质上的缺点是因为，为了求解器求解方便，为了实时性，需要考虑到很多假设，比如线性，比如凸优化，因此导致得到的结果可能没有learning的方法得到的健壮。不过尽管现在到了强化学习的时代，状态估计仍然是一个很重要的思想，诸如DreamWaq和BeyondMimic中都用到了状态估计的思想，后续文章会提到，这也是我为什么会花时间写传统控制的原因。

# 深度强化学习运动控制方法（Learning Base\)

本章开始对人形机器人的深度强化学习（Deep Reinforcement Learning，RL）运动控制方案进行介绍，RL在腿足上的应用在近两三年得到了非常快速的发展，结合模仿学习和大量可利用的人类数据，目前看来已经出现了一些新的范式，让人形机器人的底层运动控制即controller很有机会能够实现通用化，配合LLM的影响，这是人形机器人能够突然走上台前的原因。本文将依次介绍主流的一些框架和算法，同样，本文的目的在于让读者Know\-How，讲一些必要的知识，更多细节会通过参考文章的方式注明。

## 深度强化学习理论基础（RL）

![图片31\.png](Images_attachments/图片31.png)

深度强化学习（Deep Reinforcement Learning，RL）和强化学习的区别就在于所谓深度是指Policy采用了神经网络的方式来实现。因此，最重要的是理解强化学习，**后续的介绍中，笔者将从最优控制思想的角度介绍RL算法，然后引出Actor\-Critic控制框架和当前最流行的PPO算法**，更多更详细的强化学习推导和介绍可以参考书籍，或者[强化学习的几个主要方法（策略梯度、PPO、REINFORCE实现等）](https://www.cnblogs.com/GreenOrange/articles/18582769)。

在[前文中我们多次提到](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87#share-CClUdHI8doF2KpxvKbZcpLpDnbe)，强化学习也是一种最优控制方法，其本质基于动态规划的原理，是一个马尔科夫决策过程，类似于传统优化控制中的DDP。

![image\.png](Images_attachments/image%2028.png)

在用DDP求解传统运动控制问题的时候，我们会定义两个函数，一个是状态值函数（Value function），一个是动作值函数（Q function），因此在强化学习这里，也会定义相应的Value function和Q function，**他们的来源都是相同的**。即得到Value function和Q function的核心假设均是动态规划\+马尔科夫决策过程，核心原理都是贝尔曼最优（Bellman Optimization\)。

> **假设**：马尔可夫决策过程（下一时刻状态只依赖于上一时刻状态和当前输入） ，即满足动态规划，而动态规划基于代价最优原理。
> 
> **贝尔曼最优性**：从任意时刻 t 开始，剩余问题仍然是一个最优控制问题。
> e\.g 从北京（起点）到广州（终点）找一条最短路线（最优策略）假设最优路线是：北京→石家庄→郑州→武汉→广州根据 BOP，这条路线中任意一段子路径（比如郑州→武汉→广州），必然是 “从郑州到广州” 的最短路线。如果郑州到广州还有更短的路线（比如郑州→长沙→广州），那原路线就不是最优的，矛盾）
> 
> **Value function**：从状态x出发，后面都是最优
> 
> **Q function**: 从状态x出发，任意一个u可能面对的所有情况。因此Value function等价于找到最优决策u对应的Q function
> 
> 

![图片32\.png](Images_attachments/图片32.png)

**最优化问题迭代找到最优解需要梯度提供方向，因此最重要的事情是找到梯度：**

DDP必须有模型方程，是因为这是DDP求解控制律u时候梯度的来源，有了模型方程才能找到这些function梯度，才知道 minJ 迭代收敛的方向。

RL可以做到无模型方程，是因为用 “概率论\+采样” 得到了梯度，我们知道RL每次决策之后会拿到一个奖励，而且这个奖励是折现奖励（当前时刻的奖励可能当前很有用，未来可能没那么有用，就像通货膨胀一样），**那么我们最关键的就是要知道RL的梯度是怎么来的，只要有这个梯度就能迭代**。

![图片33\.png](Images_attachments/图片33.png)

经过推导，我们可以得到如图红框中的式子，这是我们的梯度，我们已经找到它了，那么步骤就是：

1. 通过神经网络得到动作输出（由于我们Actor网络建模的是动作分布，因此Actor得到的应该是均值loc和方差scale，然后我们再利用这两个参数采样得到一个Action作为动作输出），然后得到 log 的概率密度值：

    ```Python
    def log_prob(self, value):
        var = self.scale ** 2
        log_scale = torch.log(self.scale)
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    ```

2. 收集当前轮次下的累计折现奖励Gt

3. 累加求和，取平均

**然而在实际过程中，我们会发现Gt的随机性很大，方差很大**，因为其包含未来所有随机动作 \+ 环境噪声，这会让我的梯度更新非常不稳定，那有没有什么办法既能保证梯度是无偏的，同时又能保证我的方差能小呢？

**Actor\-Critic框架**

我们已经有了Actor，理论上已经可以进行梯度更新，但是这样做方差很大，因此我们现在**引入一个新的Critic网络，用来估计Gt**，这样让策略梯度不单单依赖于Gt，而是**依赖于Gt减去Gt\_critic**，这样做就能**去除状态相关的平均回报，估计只剩下动作选择带来的增量变化**，就不会被不同状态带来的不同水平的回报影响到。

![图片34\.png](Images_attachments/图片34.png)

这里给出Actor\-Critic框架的伪代码如下所示：

```C++
初始化 Actor 网络参数 θ
初始化 Critic 网络参数 φ
初始化环境 Env

for episode in 1 to M do
    初始化状态 s
    for t in 1 to T do
        # Actor 根据当前状态选择动作
        πθ(a|s) ← Actor 网络输出动作分布
        a_t ~ πθ(a|s)  # 从策略采样动作

        # 执行动作，得到奖励和下一状态
        s_{t+1}, r_t ← Env.step(a_t)

        # Critic 计算当前状态价值估计
        V_φ(s_t) ← Critic 网络预测当前状态价值
        V_φ(s_{t+1}) ← Critic 网络预测下一状态价值

        # 计算 TD误差（优势估计，δ_t 就是Advantage function）
        δ_t = r_t + γ * V_φ(s_{t+1}) - V_φ(s_t)

        # Critic 更新：最小化TD误差平方
        φ ← φ - η_c * ∇_φ [δ_t²]

        # Actor 更新：策略梯度
        θ ← θ + η_a * ∇_θ log πθ(a_t | s_t) * δ_t

        s ← s_{t+1}

        if s is terminal:
            break
```

**PPO算法（Proximal Policy Optimization, 近端策略优化）**

尽管在Actor\-Critic的框架下我们可以得到一些稳定的梯度，但是在实际工程中，仍然发现我们没法保证梯度更新幅度，而人们发现**如果一次更新策略参数幅度过大，可能导致策略分布剧烈变化**，从而引起训练不稳定甚至性能崩溃，也会存在安全的问题，因此为了让实际的梯度更新给更加鲁棒，就有很多不同的算法，其中PPO是目前在强化学习领域使用最广泛的。

> 一种启发性的类比是：如果说Actor\-Critic是传统优化控制中内点法，活动集法的位置，那么PPO就是线搜索的位置。前者决定优化梯度的方向，后者决定梯度更新的步长。
> 
> 

**PPO算法限制梯度步长的更新一般有两种方式**，一种比较暴力的直接限幅裁剪新旧策略动作的幅度（clip 方法），另外一种通过比较新旧策略分布的KL散度差异来保证步长更新不要太大。这里采用使用最为方便最多的clip方法来实现，其他方法可以参考很多文章。

![image\.png](Images_attachments/image%203.png)

这里给出Actor\-Critic框架配合PPO的伪代码：

```C++
初始化策略网络参数 θ_old（Actor）
初始化价值网络参数 φ（Critic）
设置剪切参数 ε，学习率 η_a, η_c

for iteration = 1 to M do
    收集一批轨迹数据 {s_t, a_t, r_t, s_{t+1}}，用旧策略 π_{θ_old} 采样

    计算优势估计 Â_t（比如用 GAE 或 TD误差）：
        Â_t = r_t + γ * V_φ(s_{t+1}) - V_φ(s_t)

    for epoch = 1 to K do
        # Actor前向，得到当前策略动作概率
        π_θ(a_t|s_t)

        # 计算概率比率 r_t(θ)，这里是 PPO 关键：
        r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)   # <-- PPO核心：重要性采样比

        # PPO剪切目标：
        L_clip = mean( min( r_t(θ) * Â_t,
                            clip(r_t(θ), 1 - ε, 1 + ε) * Â_t ) )  # <-- PPO核心：限制更新幅度

        # Critic价值函数损失（均方误差）
        L_vf = mean( (V_φ(s_t) - target_t)^2 )
        # target_t = r_t + γ * V_φ(s_{t+1}) 或 GAE返回值

        # 熵奖励（鼓励探索，可选）
        L_entropy = mean(Entropy(π_θ(·|s_t)))

        # 总损失（要最大化策略目标，所以 Actor 损失取负）
        Loss = -L_clip + c1 * L_vf - c2 * L_entropy

        # 更新策略参数和价值网络参数
        θ ← θ - η_a * ∇_θ Loss
        φ ← φ - η_c * ∇_φ Loss
    end for

    # 用当前策略参数更新旧策略参数
    θ_old ← θ
end for

```

在接下来人形机器人运动控制不同的深度强化学习方法中，大多都是在描述深度部分，即神经网络环节怎么设计和处理，除了下节的DAgger算法、Retarget重映射和BFM中的FB算法以外，均是通过Actor\-Critic框架\+PPO算法实现的。因此在接下来的讲述中，笔者给出伪代码的时候不会太详细去描述Actor\-Critic框架\+PPO算法的内容，而是重点写深度神经网络框架中的函数。

换句话说，**以上的内容就和传统运控的最优控制原理一样，是一个共同的基础理论，对于之后的内容，笔者会重点讲述深度神经网络的不同框架**，有的能实现鲁棒行走，有的能实现视觉端到端，有的能实现跳舞，有的实现全身运动跟踪。同样尽管我不在这里写出实战代码，但是我会尽量给出可以参考的文章和开源代码。

## Teacher\-Student模型和DAgger训练算法

### 方法原理

基于Teacher\-Student框架在腿足上应用的初衷是为了让其更好的Sim2real，在大家对Sim2real还没有现在这么容易的时候，**为了Sim2real到复杂鲁棒地形，必须要让机器人在一个充满噪声和延迟，部分状态不可知，尽量模拟部署实物情况的环境中训练**，然后大家发现这样**直接训练得到的效果似乎不太行**，就算学会，适应性也很差。由此，ETH提出了一套Teacher\-Student的训练框架，旨在帮助四足机器人的强化学习去落地，克服实物部署的问题，在如今的[rsl\_rl库](https://github.com/leggedrobotics/rsl_rl/tree/main)中，可以看到ETH留下一个Teacher\-student训练框架。

笔者在长文开头回答过一个问题，人形机器人和仿真橡皮人的区别，在于**仿真里所有的数据和状态都是可知且准确的**。Teacher\-Student框架正是利用了这个特点，即先在一个所有状态都可知（**比如质心速度，全局复杂地形，甚至噪声和干扰力大小都当作实时变化的已知状态项**）且准确的环境中训练一个策略，显然这个策略无法直接上实物部署（我们不可能知道每个时刻的噪声是多大），但是它应该代表了在机器人结构下能达到的最好的效果，把这个策略就称为**Teacher策略**。然后再让一个带有不可知噪声且状态部分可知的可以实物部署的策略去监督学习Teacher策略的输出，这样不就可以既能获得Teacher的表现又能实物部署吗，后者把监督学习的**这个过程称为蒸馏（distillation）**，把蒸馏出来的可部署的策略称为**Student策略**。

**Teacher\-Student策略通常有以下几个作用**：

1. Teacher策略被称为特权学习，通常训练出来的是你想要完成该任务的最佳表现

2. Student不用考虑奖励偏好的问题，因为是针对teacher的Action做纯监督的学习

3. Student可以考虑蒸馏成你想要的任何指令形式，比如teacher阶段需要全身的轨迹参考作为输入，student阶段可以考虑只需要质心的轨迹参考作为输入，因为student只是在做Action loss

![image\.png](Images_attachments/image%2038.png)

**DAgger算法**

怎么让student去学习teacher策略呢，一种直觉的办法是训练好的teacher策略在仿真环境中收集大量数据，制作一个包含很多**（S\_teacher, Action\_teacher）**状态动作对的数据集，我们知道 **S\_student **是S\_teacher的子集，是S\_teacher去除实物部署时不可知的部分，比如具体的噪声值，干扰力大小，地形信息（对于盲走来说），质心速度等，只保留实物部署时容易得到的比如关节角度和关节速度，身体姿态等。然后让student网络输出的 **Action\_student **监督学习 **Action\_teacher，即二者之间采用MSE误差：**

![image\.png](Images_attachments/image%2031.png)

**在这种直觉的方法中，我们只在训练teacher的时候建立了仿真环境去让机器人交互收集状态动作对作为数据集，然后在student的时候就不用了**，但是，这样做是有问题的，因为纯模仿学习往往假设训练数据分布和测试时环境状态分布一致，但学生在实际执行时会进入训练时没见过的状态，导致错误累积，性能下降。**因此，为了解决纯模仿学习中的分布偏移（covariate shift）问题，通常采用DAgger算法，即在线蒸馏**。

在线蒸馏的步骤就是，在我student学习的时候，也要为我的student单独创造一个交互的环境，已有的teacher策略需要输入的是student在这个新的环境交互更新得到的状态，而不是只拿自己原本teacher环境交互过的数据。换句话说，teacher策略需要实时根据student的状态推出新的**Action\_teacher**，然后**Action\_student 实时**监督学习 **Action\_teacher。**

总结，为了解决腿足机器人复杂地形不带视觉运动的问题，我们先训练teacher策略，在teacher眼中，神经网络能拿到所有仿真的状态，包括当前的噪声大小，干扰大小，地形情况，由于所有状态都已知，teacher在仿真中很轻易就实现了复杂地形的运动。由此我们蒸馏到student网络中之后，就可以在真正的机器人上也部署一个能跨越复杂地形的机器人算法。

### 基本代码

这里给出DAgger算法的伪代码，假设teacher策略已经训好：

```Python
# Student 策略基于 Action Loss 的更新流程（伪代码）
teacher_policy = mlp(obs_teacher_dim, action_dim) # teacher网络，由简单三层mlp构成[512，256，128]
student_policy = mlp(obs_student_dim, action_dim) # student网络，由简单三层mlp构成[512，256，128]
loss = 0
for epoch in range(num_epochs):
    # 让student与环境交互
    obs_teacher, obs_student = env.step(action_student) 
    
    # 教师策略前向推理
    action_teacher = teacher_policy(obs_teacher).detach() # 已经训练好的teacher策略，detach()表示不用梯度更新
    
    # 学生策略前向推理
    action_student = student_policy(obs_student)

    # Action Loss（行为克隆损失）
    loss += l2_norm(action_student - action_teacher)

    step += 1

    # 每 K 步执行一次梯度更新
    if step % K == 0:
        loss.backward()          # 反向传播
        optimizer.step()         # 更新参数
        optimizer.zero_grad()    # 清空梯度
        loss = 0
```

### 方法局限性

teacher\-student方法解决了单一网络下，由于存在不可知状态和噪声的影响，只一阶段无法学出太好的复杂地形通过能力的问题。其缺点在于：一是训练成本比较高，因为需要训练两次；二是由于第二阶段是蒸馏，导致student一定超过不了teacher，由于直接在Action上做MSE的loss，student很有可能会被压缩表现退化。

随着硬件水平的上升，为了缩短训练成本，也为了充分利用强化学习探索的优势，在此之后，越来越多的研究在探索一阶段复杂地形的训练算法。

当然，teacher\-student DAgger的框架并没有就此终结，在人形机器人的模仿学习到来之后，为了同时学会更多的动作，该框架又会重新回到台前。

## DreamWaq盲走一阶段鲁棒行走训练算法

### 方法原理

之前采用一阶段学习复杂地形行走行不通的核心原因，**一方面是实物硬件水平有限，gap较大**，比如ETH一直以来用的ANYmal四足机器人关节结构都是**SEA（Series Elastic Actuator 串联弹性驱动器）**的，具有比较高的弹性，这是时代的产物，他们甚至特地为了关节建了模型分析，不管是在传统控制领域，还是在强化学习阶段，相信该关节导致的gap都给他们带来了不少的问题，要是再过几年，等到**QDD的关节力矩电机直驱方案**成熟，也许曾经的一阶段也能训练出不错的结果。**随着QDD方案的成熟，这方面带来的问题逐渐被解决**。

![image\.png](Images_attachments/image%2010.png)

**另一方面的原因，是因为如果只是一阶段，就要直接面临部分状态不可知\+存在很多噪声下复杂地形训练的问题**，复杂地形下的质心速度和质心高度，都是不太好被准确估计的。直到2023年1月的一篇提出算法为DreamWaq的论文解决了该问题，其核心思想是，既然传统方法可以做到卡尔曼滤波的状态估计，说明质心速度一定有内在的模型，只是无法被直接准确观测得到，那么能不能利用神经网络去做状态估计替代传统方案，这样就可知了。该篇文章在四足机器人实现了质心速度、质心高度等变量的状态估计，很大地提升了机器人一阶段的性能表现。而同样在2023年底，我们课题组也将[DreamWaq在人形上实现了鲁棒行走](https://www.bilibili.com/video/BV1EK421Y7UG/?spm_id_from=333.1387.homepage.video_card.click)。

> DreamWaq方法原论文：
> Nahrendra I, Yu B, Myung H\. DreamWaq: Learning robust quadrupedal locomotion with implicit terrain imagination via deep reinforcement learning\[J\]\. arXiv preprint arXiv:2301\.10602, 2023\.
> DreamWaq人形方法论文：
> Wang Z, Wei W, Yu R, et al\. Toward understanding key estimation in learning robust humanoid locomotion\[C\]//2024 IEEE/RSJ International Conference on Intelligent Robots and Systems \(IROS\)\. IEEE, 2024: 11232\-11239\.
> 
> 

![image\.png](Images_attachments/image%2030.png)

以上图这个框架为例，首先整体一阶段的框架仍然是Actor\-Critic\+PPO。DreamWaq算法体现在Estimation loss, 其中的真值由仿真环境提供，估计值由Encoder网络得到，即et。估计的变量包括质心速度、质心高度、足底周围的地形高度图。根据论文结果以及笔者多次实验得出的效果来看，Encoder得到的相关变量对机器人表现提升的重要程度如下：

![image\.png](Images_attachments/image%2036.png)

**在DreamWaq上得出的结论是，对于鲁棒行走任务，估计质心速度对于提升鲁棒行走任务是最重要的，甚至可以没有其他变量**。笔者认为这个结论的直觉来自于，对于鲁棒行走任务，跟踪好质心速度通常是最重要的一个奖励，因此如果能观测到准确的质心速度，自然提升就会很明显。（在后来笔者做模仿学习时也做过类似的测试，把DreamWaq用在模仿学习跟踪动作上，提升就没有那么明显了，或者说，并不会让你的动作有像现在这样质的提升）。**因此，如果你想要在人形或者四足机器狗上实现一个带视觉/不带视觉的鲁棒行走，强烈推荐试试使用这个方案，做的好就可以得到**[**这样的效果**](https://www.bilibili.com/video/BV13Jm8YbEpg/?spm_id_from=333.337.search-card.all.click&vd_source=30216de2308cf451749cc50ccb881d29)**。**

### 基本代码

这里写出DreamWaq算法的伪代码，核心包含三个loss，网络框架都使用MLP，还是比较简单的。

```Python
# ===============================
# DreamWaQ: Training Loop
# ===============================
est_dim = com_V_dim + com_h_dim + height_map_dim
encoder = mlp(obs_history_dim, z_dim + est_dim) # mlp均为[512,256,128]
decoder = mlp(z_dim, obs_dim)
policy = mlp(obs_dim + z_dim + est_dim, action_dim)
critic = mlp(priviledge_obs_dim, 1)

for iteration in range(num_iterations):

    # ---------- 数据采集 ----------
    obs_t = env.get_proprioception()          # o_t
    obs_hist = env.get_history()               # o_{t-1:t-h}
    privileged_info = env.get_privileged()     # p_t（仅训练可见）
    estimation_gt = env.get_estimation_gt()    # e_t（训练监督信号）

    # ---------- 状态估计（Encoder） ----------
    z_t, e_hat_t = encoder(obs_hist)            # μ: 编码历史观测
                                                 # z_t: latent
                                                 # e_hat_t: 估计量

    # ---------- 策略执行 ----------
    action_t = policy(obs_t, z_t)               # ψ: Backbone policy
    torque = PD_controller(action_t)            # PD 控制
    env.step(torque)

    # ---------- 动力学重构（Decoder） ----------
    obs_hat_t1 = decoder(z_t)                   # η: 预测下一时刻观测

    # ---------- Critic ----------
    state_t = concat(obs_t, privileged_info)    # s_t（含特权信息）
    V_t = critic(state_t)                       # φ: 价值函数

    # ---------- 损失函数 ----------
    # 1) 状态估计损失
    L_est = mse(e_hat_t, estimation_gt)

    # 2) 重构损失
    L_rec = mse(obs_hat_t1, obs_t1)

    # 3) 强化学习损失（Actor-Critic）
    L_actor = policy_loss(action_t, advantage)
    L_critic = value_loss(V_t, return_t)

    # ---------- 总损失 ----------
    L_total = L_actor + L_critic + λ1 * L_est + λ2 * L_rec

    # ---------- 参数更新 ----------
    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()

```

### 方法局限性

该方法的好处很明显，就是在一阶段内，实现机器人的鲁棒行走。但正如实验表明的那样，鲁棒行走任务由于对质心速度跟踪的需求最为强烈，因此估计出质心速度是尤为重要的，但是如果不是鲁棒行走的任务就没有那么好用了。

因此，这类方法对于四足机器狗/四轮足机器狗/人形机器人的鲁棒行走任务是非常友好的，但是你的任务是多模态的任务，或者模仿学习，涉及到多个重要程度的切换，就不一定好用了。

## PIE感知一阶段鲁棒行走算法

### 方法原理

有了DreamWaq算法框架给四足/人形机器人不带视觉的鲁棒行走任务带来了质的提升，研究人员就会尝试加入视觉，因为加入视觉之后，理论上就能实现稳定上下楼梯（不用靠着盲走去试探），爬高台，跳跃沟壑等盲走做不到的效果，更进一步提升腿足Locomotion的能力。

在原有的DreamWaq框架上加上视觉在总体上来说没有很困难，最简单的就是在我的状态中多加入视觉信息，这里以笔者课题组在2023年底四足机器狗上的研究成果为例，讲述框架，再探讨其中的细节。以下内容细节来自论文作者亲自答[@lsx](https://www.zhihu.com/people/luo-shi-xin-51)。

**PIE框架 = 视觉 \+ DreamWaq框架**

![image\.png](Images_attachments/image%206.png)

1. **对于视觉图像的处理**

    1. **渲染加速**：Isaac Gym自带渲染接口占用显存空间太多，渲染太慢，采用Nvidia Warp库raytracing机制使用warp kernel多个机器人间多个像素点间都是并行渲染，性能提升n倍。

    2. **图片处理**：为了极致的速度，没有进行任何滤波抗锯齿之类的操作。图片质量其实相对于实物realsense来说都不是特别高，但也许这反更有利于克服sim2real gap。图片也没有加任何噪声，只是加了时延和位姿的随机化。对于大块的、距离视线在0\.1m之外两米以内的物体来说，realsense打开各种spatial、temporal和hole\-filling filter之后与仿真图像质量差别不大。

2. **对于Encoder的改进**

    1. **网络架构**：参考[LocoTransformer](https://rchalyang.github.io/LocoTransformer/)的结构设计Encoder，有没有Transformer不是关键因素，但可以加速收敛。由于使用的是Transformer Encoder，所以没有时序记忆功能。一开始是堆叠多帧图像和历史本体信息去保存一段时间记忆，但学出来的步态一直是trot，跳远跳高也是trot比较奇怪成功率也不高，改在后面加一个GRU之后步态变得非常自然。

    2. **估计值选择**：除了DreamWaQ的估计，由于加了深度图，所以要有对地形的明确感知。在我们没有采用Warp做渲染时，显存占用过大，并发环境数只能到256、512这个量级。当我们只使用VAE做地形编码时完全学不出来；使用AE做地形编码时地形重构（decoder）会学的不错，但是输入给policy的隐向量太抽象，机器狗对沟壑等地形探索多次失败之后就摆烂不走了。只有引入显示的地形估计（Ztm\)，比如显示估计高程图全部地形点才能让机器狗探索出来跳沟等动作。考虑到真实环境与仿真环境仍然有很大差距，显式估计现实的复杂地形会有大量out of distribution的场景，因此折中采用显式估计四个抬脚高度并隐式编码高程图的方法。（**如果采用raytracing camera并发大量环境，设计更合理更细粒度的课程，网络结构可以更简单更直接更暴力也能学出来**）

![image\.png](Images_attachments/image%205.png)

### 基本代码

```Python
# --------------------------------
# Network Definitions
# --------------------------------

# Estimator (Temporal Encoder)
proprio_encoder = MLP(obs_hist_dim, enc_dim)        # [512, 256, 128]
depth_encoder  = CNN(depth_img_shape, enc_dim)

temporal_encoder = TransformerEncoder(enc_dim)
temporal_fusion  = GRU(enc_dim, hidden_dim)

# Estimator outputs
latent_head      = Linear(hidden_dim, z_dim)
velocity_head    = Linear(hidden_dim, v_dim)        # \hat{v}_t
height_head      = Linear(hidden_dim, h_dim)        # \hat{h}_t

# Decoders (auxiliary supervision)
proprio_decoder  = MLP(z_dim + v_dim + h_dim, obs_dim)      # \hat{o}_{t+1}
height_decoder   = MLP(z_dim, height_map_dim)               # \hat{m}_t

# Policy & Critic
actor  = MLP(obs_dim + z_dim + v_dim + h_dim, action_dim)
critic = MLP(privileged_obs_dim, 1)

# Optimizer
optimizer = Adam(
    params = (
        proprio_encoder +
        depth_encoder +
        temporal_encoder +
        temporal_fusion +
        latent_head +
        velocity_head +
        height_head +
        proprio_decoder +
        height_decoder +
        actor +
        critic
    ),
    lr = lr
)

# --------------------------------
# Training Loop
# --------------------------------

for iteration in range(num_iterations):

    # ===============================
    # 1. Data Collection
    # ===============================

    o_t      = env.get_proprioception()        # current proprioceptive obs
    o_hist   = env.get_proprio_history()       # temporal proprioception
    d_hist   = env.get_depth_history()         # temporal depth images

    p_t      = env.get_privileged_obs()        # privileged state (critic only)

    o_tp1    = env.get_next_proprio_gt()       # GT next observation
    h_gt     = env.get_height_map_gt()          # GT height map
    v_gt     = env.get_velocity_gt()            # GT base velocity

    # ===============================
    # 2. State Estimation (Estimator)
    # ===============================

    # Encode temporal observations
    e_prop = proprio_encoder(o_hist)
    e_dep  = depth_encoder(d_hist)

    e_t = e_prop + e_dep                        # feature fusion

    # Temporal modeling
    h_enc = temporal_encoder(e_t)
    h_t   = temporal_fusion(h_enc)

    # Estimator outputs
    z_t     = latent_head(h_t)                  # latent dynamics code
    v_hat_t = velocity_head(h_t)                # estimated velocity
    h_hat_t = height_head(h_t)                  # estimated height

    # ===============================
    # 3. Policy Execution
    # ===============================

    actor_input = concat(o_t, z_t, v_hat_t, h_hat_t)
    a_t = actor(actor_input)

    torque = PD_controller(a_t)
    env.step(torque)

    # ===============================
    # 4. Auxiliary Decoding
    # ===============================

    o_hat_tp1 = proprio_decoder(concat(z_t, v_hat_t, h_hat_t))
    h_hat_map = height_decoder(z_t)

    # ===============================
    # 5. Critic Evaluation
    # ===============================

    V_t = critic(p_t)

    # Advantage & Return (standard GAE / TD)
    A_t, R_t = compute_advantage_and_return(V_t)

    # ===============================
    # 6. Loss Functions
    # ===============================

    # (1) Estimation loss
    L_vel = mse(v_hat_t, v_gt)
    L_hgt = mse(h_hat_t, h_gt)
    L_est = L_vel + L_hgt

    # (2) Reconstruction loss
    L_rec_obs = mse(o_hat_tp1, o_tp1)
    L_rec_map = mse(h_hat_map, h_gt)
    L_rec = L_rec_obs + L_rec_map

    # (3) Actor-Critic loss
    L_actor  = -mean(log_prob(a_t) * A_t)
    L_critic = mse(V_t, R_t)

    # ===============================
    # 7. Total Loss
    # ===============================

    L_total = (
        L_actor +
        L_critic +
        λ_est * L_est +
        λ_rec * L_rec
    )

    # ===============================
    # 8. Optimization Step
    # ===============================

    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()
```

### 方法局限性

1. **视线遮挡**：在障碍物距离realsense太近或realsense被遮挡时会有非常大的噪声，由于没有太多处理这类问题，因此在高草丛等视线遮挡环境表现不佳。（在后续move改善）

2. **仅能前向运动**：训练地形完全是前向加转弯，因为仅有前向视角，不好利用视觉信息全向移动，操作不便（在后续move改善）。

3. **遇墙大跳**：训练的高台地形都是机器狗能跳上去的，对于机器狗上realsense的fov来说，0\.75m的高台已经看不到顶了，对它来说就像墙一样。因此看到墙它也认为是高台，手柄推的近了也想跳。

4. **训练地形单一**：野外环境基本全靠鲁棒性，并未专门训练。对于一些特殊场景，比如镂空工业楼梯等完全out of distribution，需要加入更加丰富的地形环境。

## Attention落足点优化算法

### 方法原理

上文实现了盲走和感知的鲁棒行走，但是上文的纯强化学习方法相较传统的基于模型的方法**在稀疏地形上缺乏精度**。**Attention\-Based Map Encoding for Learning Generalized Legged Locomotion**提出了一种结合了强化学习和注意力机制的新方法，旨在解决足式机器人在复杂和稀疏地形上的运动问题。

其主要基于**多头注意力模块**，该模块不仅接收地形高度图，还以机器人的本体感知状态（如速度指令、当前姿态）为条件。这使得网络能够根据机器人的运动意图，**动态地关注地形图中对当前决策最重要的区域**（例如即将踩踏的落脚点），而不是盲目处理整个地图。

![image\.png](Images_attachments/image%208.png)

同时与许多黑盒神经网络不同，该方法的注意力权重可以可视化。论文中表明，高注意力区域通常对应于最佳的未来落脚点。**这意味着网络实际上学会了一种隐式的“落脚点选择”机制**，为理解神经网络如何感知地形提供了直观的窗口。

![image\.png](Images_attachments/image%2034.png)

下面是算法实现的核心细节：

1. **逐点特征提取的MHA**：系统首先利用轻量级卷积神经网络对以机器人为中心的局部高程图进行扫描，提取每个网格点周围的坡度、边缘等局部几何特征；随后，**显式地将每个点相对于机器人的三维空间坐标 **$(x, y, z)$** 拼接到该特征向量中**，从而生成兼具“几何语义”与“空间位置”信息的独立特征序列；这些特征最终作为注意力机制中的键（Key）和值（Value），使网络能够根据机器人的运动意图（Query）精确索引并关注特定的地形落脚点，从而在保留高精度的同时实现对复杂稀疏地形的有效编码。

2. **两阶段训练流程**：为了学习一种既鲁棒又具有泛化能力的地图编码，设计了一个两阶段训练流程，旨在逐步优化控制器的能力。**在第一阶段，控制器在拥有完美感知的基础地形上进行训练**。这一阶段对地图编码的学习进行了预热，使控制器能够利用真值传感数据习得基本的运动技能。**在第二阶段，引入了包含扰动和不确定性的更复杂地形。**这些地形模拟了感知可能存在缺陷的更真实的现实环境，从而将学到的运动策略塑造得更具适应性和韧性。通过将机器人置于更广泛的地形环境中并施加额外的扰动，这一阶段不仅提高了控制器在多样化地形上的泛化能力，还增强了其应对现实世界不确定性的鲁棒性，从而确保最终习得的地图编码能够在未见过的环境中有效运行。

### 基本代码

```Python
# --------------------------------
# Network Definitions
# --------------------------------

# Embeeding
proprio_encoder = MLP(obs_dim, embedding_dim)
feature_extractor = CNN(1, embedding_dim - 3)

# MHA
mha = nn.MultiheadAttention(embedding_dim, num_heads)

# Policy & Critic
actor = MLP(obs_dim + embedding_dim, action_dim)
critic = MLP(privileged_obs_dim + embedding_dim, 1)

# Layer norm
ln_q = LayerNorm(embedding_dim)
ln_kv = LayerNorm(embedding_dim)
ln_out = LayerNorm(embedding_dim)

# Optimizer
optimizer = Adam(
    params = (
        proprio_encoder +
        feature_extractor +
        mha +
        actor +
        critic
    ),
    lr = lr
)

# --------------------------------
# Training Loop
# --------------------------------

for iteration in range(num_iterations):

    # ===============================
    # 1. Data Collection
    # ===============================

    o_t      = env.get_proprioception()        # proprioceptive obs
    m_t      = env.get_map()                   # map

    s_t    = env.get_states()                  # state (critic only)
    
    pos_encoding = create_position_encoding()  # pos_encoding

    # ===============================
    # 2. State Estimation (Estimator)
    # ===============================

    # Encode and feature extract
    embedding = proprio_encoder(o_t)
    feature = feature_extractor(m_t)

    pointwise_feature = concat(feature, pos_encoding)    # Pointwise feature

    # Q&KV
    q = ln_q(proprio_embedding)
    kv = ln_kv(pointwise_features)

    # Attention
    attn_output = mha(
            query=q,
            key=kv,
            value=kv
    )
    
    map_encoding = ln_out(attn_output)

    # ===============================
    # 3. Policy Execution
    # ===============================

    actor_input = concat(o_t, map_encoding)
    a_t = actor(actor_input)

    torque = PD_controller(a_t)
    env.step(torque)

    # ===============================
    # 4. Critic Evaluation
    # ===============================

    V_t = critic(s_t)

    # Advantage & Return (standard GAE / TD)
    A_t, R_t = compute_advantage_and_return(V_t)

    # ===============================
    # 5. Loss Functions
    # ===============================

    L_actor  = -mean(log_prob(a_t) * A_t)
    L_critic = mse(V_t, R_t)

    # ===============================
    # 6. Total Loss
    # ===============================

    L_total = (
        L_actor +
        L_critic +
    )

    # ===============================
    # 7. Optimization Step
    # ===============================

    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()
```

### 方法局限性

End\-to\-End训练过程以及注意力模块的加入使得训练非常耗时，通常需要数天的训练时间。注意力模块也略微增加了推理时的计算负载，需要在机载计算机的算力与实时性之间做平衡。

该方法本质上仍依赖于视觉/激光雷达生成的局部地图。如果环境感知极度退化（如浓烟、强光干扰导致地图完全错误），严重OOD，其性能将甚至不如盲走。

该方法依赖于从高程图采样生成的机器人中心高度图。这是一种2\.5D表示，虽然对大多数地面有效，但无法有效表示复杂的3D几何结构（如悬垂物、洞穴顶部或极其复杂的立体空间），这限制了其在某些极端环境下的应用。

## 模仿学习基础：人\-机器人重映射算法（Retarget）

### 方法原理

在后续的方法中，我们将介绍基于数据集学习的模仿学习\+强化学习算法，即存在用于监督学习的机器人仿人动作数据集以供学习。得益于虚拟角色设计和计算机图形学的几十年来的发展对人体动作数据的需求，产出了很多比如AMASS、[LAFAN1](https://huggingface.co/datasets/johnny095212/lafan1/tree/main)、[OMOMO](https://github.com/lijiaman/omomo_release)等开源人体数据集，正如电影花絮中看到的那样，他们都是**由专业的动捕演员身穿动捕服采集得到的数据**。

而随着AI的发展，对数据集需求的增大，动捕采集效率低的弊端也显现出来，这就提出了**从视频中采集数据**的需求，因为互联网的视频有很多，因此，在之后的很多开源数据集都因此扩展到了更大的规模，比如Motion\-X，Motion\-X\+\+、Humanoid\-X等。这些数据集不仅有人体数据，而且通常有对应动作的文本标签。

> 上述提到但未链接的数据集地址：https://github\.com/myismyname/awesome\-bfm\-papers
> 
> 

通常从开源数据集中得到的人体数据集是有标准格式的，统称为**SMPL，一种参数化人体模型，包括了pose（θ）和shape（beta）参数。其中pose描述人体关节点位姿，shape描述人的体型差异。**

![图片36\.png](Images_attachments/图片36.png)

> SMPL格式通常包含SMPL、SMPL\-H、SMPL\-X
> 
> SMPL: 能够用 **形状参数 β（≈10 维）** 与 **姿态参数 θ（24\*3维）** 生成完整的三维网格
> 
> SMPL\-H: 在 SMPL 的基础上加入了**双手的关节和细节, 每只手 14 个手指关节, 即姿态参数 θ（52\*3维）**
> 
> SMPL\-X: 在 SMPL 基础上加入 **全身手部（MANO）** 与 **表情面部（FLAME）**，形成统一的 **55 个关节**模型，能够一次性捕获 **身体‑手‑脸** 的细节，**即姿态参数 θ（55\*3维）**
> 
> 更多SMPL内容请参考：https://zhuanlan\.zhihu\.com/p/696878001
> 
> 

**重映射的过程**，就是将SMPL格式的人体数据（可以是动捕也可以是视频得到）拟合到机器人合适的各个关节位置上（机器人未必有相同的pose数量，选取合适的即可）。

![图片37\.png](Images_attachments/图片37.png)

- **从人体SMPL格式数据到机器人的具体重映射步骤**（通常无论是什么SMPL格式，前24\*3维的关节序号就如上图所示，一般情况下吗，模仿学习采用该24个关节即可）

    1. 下载数据集，拿到数据集当中的shape、pose、root（全局位姿）三个参数

    2. **坐标轴统一**，由于SMPL全局坐标系world frame可能是Y\-up/Z\-up，因此建议先全部转化到Z\-up。**选取并root对齐**，选取以人体pelvis为身体坐标系原点对齐（pelvis原本可能不为0）。

    3. 选择合适的SMPL joint 与 robot joint的对应表，不追求1:1，追求功能即可。如果你的robot joint数量要大于SMPL joint数量，就需要给定约束或者正则化。

    4. **必须先拟合shape\(beta\)参数**，即先将数据集人体SMPL数据缩放到对应机器人的shape上，这里目前有两种方式：

        **优化法（PHC论文方法）**：选定一个/多个合适的位姿，比如站立平举，即将人体的24\*3维设置为站立平举位姿，同时将机器人的关节也设置为站立平举位姿（URDF可视化）。将shape参数设置为优化变量，理论上收敛时人体和机器人选定的对应关节位姿应该相同（至少位置相同），此时优化得到的shape参数即为所求。

        **放缩法（GMR论文方法）**：全局放缩\+局部放缩。全局放缩即按照身高/腿长比例缩放全局XYZ，然后局部缩放就是按照对应各个连杆的长度人为给定缩放因子进行缩放（这个人为设计其实是不符合运动学约束的，但是由于shape参数对最终的拟合结果没有那么敏感）。

    5. **拟合关键点位姿，这里结合目前已有的三种方法进行比较分析**。

        第一种最早的是PHC论文的方法，单纯地在关键点位之间做梯度下降拟合，优点在于其由于没有任何硬约束，可以采用ADAM这种GPU并行的优化器，对于这样就可以进行多动作同时优化，缺点是由于没有任何约束，得到的结果有些时候无法保证合理。

        第二种是GMR论文中的方法，由于采用Mink库进行求解，具有严格的运动学约束（Mink是Pinocchio逆解库pink的衍生\)，这也是为什么GMR得到的重映射效果和后续的训练效果都很好的原因，并且逆解本身耗时低，这意味着GMR的重映射可以实时进行。缺点是对于多接触效果不好，对于多约束没有严格的硬约束。

        第三种是Omiretarget论文中的方法，Omiretarget的作者师从MIT优化大师Russ Tedrake，因此所做出的效果也非常好，核心是随机采样更多的关键点且以拉普拉斯变换能量最小为原则去做拟合，天然能够适应多接触和无接触的重映射场景，并且采用cvxpy求解器可以处理多约束。缺点就是比较耗时。

![image\.png](Images_attachments/image%202.png)

> 人形机器人现有开源的三篇重映射文章即开源地址如下：
> 
> PHC方法：He T, Luo Z, He X, et al\. Omnih2o: Universal and dexterous human\-to\-humanoid whole\-body teleoperation and learning\[J\]\. arXiv preprint arXiv:2406\.08858, 2024\.
> 
> 开源地址：https://github\.com/LeCAR\-Lab/human2humanoid
> 
> GMR方法：Araujo J P, Ze Y, Xu P, et al\. Retargeting matters: General motion retargeting for humanoid motion tracking\[J\]\. arXiv preprint arXiv:2510\.02252, 2025\.
> 
> 开源地址：https://github\.com/YanjieZe/GMR
> 
> Omiretarget：Yang L, Huang X, Wu Z, et al\. Omniretarget: Interaction\-preserving data generation for humanoid whole\-body loco\-manipulation and scene interaction\[J\]\. arXiv preprint arXiv:2509\.26633, 2025\.
> 
> 开源地址：https://github\.com/amazon\-far/holosoma/tree/main
> 
> 

- **笔者的思考和尝试结论**

从这三种重映射方法的迭代改进可以看到，方法的改进不仅带来了更好的retarget效果，并且在之后的RL训练过程中效果也会更好，效率也会更高。因此得到改进过程中有一个**共同的结论：越符合运动学约束和规律的重映射方案，在之后的强化学习环节越容易学习，越不用特别调整奖励**。

PHC没有什么约束，GMR有严格的运动学约束（运动学逆解IK保证），Omiretarget在利用扩大采样点数符合运动学约束的前提下严格去解一个SQP优化问题。因此，**作者在此基础上去尝试解一个完整的轨迹规划（TO）问题**（如果是实时的重映射，那就是解一个MPC问题，面对多约束场景换用更高效的优化求解器，用C\+\+编写是可以做到的），**发现确实比现有的方法展现了更好的效果**。不过前提是有一台性能良好的人形机器人，从笔者自己的经验和了解的情况上来看，最终效果影响程度 **人形机器人本体\>重映射轨迹质量\>RL算法**。

![图片38\.png](Images_attachments/图片38.png)

**为什么轨迹质量越高，得到的强化学习越不用调整奖励呢？**除了本体的原因，笔者认为，因为强化学习仿真最后也是隐式的动力学，一个机器人的运动只有在指定位置速度加速度之后才会有确定的轨迹，或者说，**确定的解**。因此，当笔者尝试将关节位置\+速度\+加速度都加入到优化问题中，并且考虑多帧一起优化的时候，得到的轨迹放到强化学习中训练的效果就是，**在RL探索找解的过程中，那些连续的，不突变的解往往是很容易找到并收敛的，而我在retarget的时候保证了运动学连续，这一点导致RL很顺利，相反，如果你的参考轨迹总是突变的，就算位置连续速度突变，由于底层动力学仿真连续，也会让RL难以找到解**。

### 方法局限性

然而，事实证明，再好的重映射算法都会受限于两个前提，一是有好的人形机器人本体，二是SMPL数据来自高质量的数据源，比如AMASS，由于年代久远的原因，AMASS数据集的质量比较一言难尽，而视频数据的整体质量都没有那么高，你虽然可以训练出来一个好的策略，但你不能评价它为“非常好”，如果你对比训练过LAFAN和AMASS，你也一定感同身受。随着BFM算法到来的未来，数据集缺口可能越来越大，是花钱买数据，还是花时间处理数据可能是一个很大的问题，哪个看着成本都很高。

## Deepmimic模仿学习\+强化学习算法（跳舞）

### 方法原理

在我们完成了人体动作重映射到机器人上的这个工作后，我们将得到一堆可用于人形机器人训练的数据集。

同样得益于计算机图形学的发展，Deepmimic方法最早在橡皮人控制中在2018年被[Xuebin Peng](https://scholar.google.com/citations?user=FwxfQosAAAAJ&hl=zh-CN&oi=sra)提出，其目的正是为了利用已有大量人体数据集的优势去做模仿学习\+强化学习。

相比于机械臂VLA，在人形机器人领域使用强化学习先于模仿学习是很自然的，机械臂作为一个固定基座的系统，不用担心太多的动力学不可行（工业机械臂大部分都是位置控制，运动学可行足以），而反观人形机器人，如果没有好好分析其动力学，他就能马上摔倒给你看，因此研究人员都会先用强化学习让机器人走起来。

而模仿学习\+强化学习在人形的行走慢慢成熟之后也是很自然的想法，由于强化学习是以奖励驱动的，鲁棒行走完成之后，自然可以将目光从单纯的质心速度奖励移到全身关节轨迹跟踪奖励上，既然有可用的跟踪的数据集，奖励又是全关节的轨迹跟踪，那自然就叫做模仿学习了，这样人形就天然将模仿学习和强化学习结合在了一起。在此之后的跳舞都是用该方法训练出来的。（仿人行走更多是下面的AMP方法）

**因此，人形机器人模仿学习\+强化学习的原理很简单，就是将强化学习的奖励改为跟踪期望轨迹奖励（包括质心和关节）。**

![image\.png](Images_attachments/image%2039.png)

- **Human2Humanoid**

CMU的Tairan He@WhynotTV是第一个完整给出人形机器人模仿学习架构的人（在之前有exbody跟踪了上半身\)，利用其计算机图形学学长Zhengyi Luo@正合时宜的PHC retarget算法，给出了如下的架构：

![image\.png](Images_attachments/image%2035.png)

- **BeyondMimic**

UC Berkeley的Qiayuan liao@廖洽源在2025年的[beyondmimic](https://github.com/HybridRobotics/whole_body_tracking)可以说是目前人形机器人最主流的代码框架。作者本身是很厉害的优化控制大佬，在笔者的NMPC那一部分中，廖佬开源了他在四足上的成果[NMPC\+WBC](https://github.com/qiayuanl/legged_control)。

- **模仿学习\+强化学习训练的关键点**

1. 有一个好的本体\+合适的PD参数\+清晰的代码框架（一个好的本体很重要！！！）

2. **高质量的重映射轨迹**

3. **自适应采样算法**：为了很好学会一段完整的舞蹈，将一段动作分为N段长为1s的切片，按照失败次数得到概率，按照概率进行重置，这样能让越难完成的动作被抽中重置的概率越高

4. **Reference State Initialization\(RSI\)**：随机重置机器人在期望轨迹附近，不需要按照时间顺序学习轨迹

5. **Early Termination**：一些条件下满足下，提前终止某个机器人的训练，不让机器人倒在地上还在试图模仿走路的动作，做浪费的学习污染数据

### 方法局限性

在Deepmimic中，机器人学习的核心目的是跟踪期望轨迹，机器人基本只能学会数据集中存在的期望轨迹，这一般会导致以下几个问题：

1. 少量数据集下，机器人动作泛化性差，机器人最多只能学会数据集中已有的动作

2. 鲁棒性和新地形适应能力差，尤其是机器人硬件不好的情况下。

3. 需要完整的期望轨迹作为输入，因此Deepmimic通常用于跳舞，不能用于手柄等只有质心的任务指令控制，除非用多阶段进行蒸馏训练。

## AMP模仿学习\+强化学习算法（仿人行走）

### 方法原理

AMP（Adversarial Motion Priors，对抗运动先验）最早也是在计算机图形学的橡皮人控制中在2021年被[Xuebin Peng](https://scholar.google.com/citations?user=FwxfQosAAAAJ&hl=zh-CN&oi=sra)提出，其目的是为了解决Deepmimic需要完整轨迹硬输入的问题。在人形机器人的AMP开源算法中，可参考[zitongbai](https://github.com/zitongbai?tab=repositories)这里的legged\_lab和rsl\_rl库中的AMP部分。

AMP的核心原理基于对抗生成网络，通过对抗训练让机器人在完成任务的同时生成符合参考运动风格（如人类动捕数据）的自然动作。**核心特点是在少量数据集下可学出一个类似数据集风格的动作，尽管在训练时期用到参考轨迹，但由于学习的是风格，在部署时无需任何参考轨迹，这样即可做到在一阶段内实现手柄等质心任务指令控制。**

- **如何做到既能学习到风格，又在部署推理时无需任何期望轨迹？**

    核心原因在于AMP在原有的Actor\-Critic网络基础上多加入了一个**判别器网络（Discriminator）**，**该判别器网络的作用是区分策略生成的动作和数据集种的动作。而策略网络也就是Actor的目的就是能让机器人做出和数据集尽可能相似的动作，进而骗过判别器**（一开始Actor输出不了正确的动作，能被判别器轻松识破，训练的目的就是让Actor有一天能骗过判别器，而判别器同样也在不停训练，目的是尽可能挑出非数据集类似的动作，因此Actor和Discriminator是在对抗中训练的，这是对抗网络的含义）。

![image\.png](Images_attachments/image%2017.png)

    - **核心：判别器网络Discriminator的独立更新和风格奖励Style reward的融合**

![图片35\.png](Images_attachments/图片35.png)

    - **！实际情况下的Expert score和Policy score**：

        在实际训练过程中，**即使人形机器人已经成功学习到稳定且自然的 AMP 行走行为，仍然常常观察到判别器对专家轨迹与策略轨迹给出明显不同的评分**：专家轨迹的 Expert score 通常接近 0\.9，而策略轨迹的 Policy score 则接近 −0\.9。这一现象并不意味着策略尚未成功模仿专家行为，而是对抗模仿学习框架下的一个正常结果。

        其原因在于，AMP 判别器的训练目标并非判断“运动是否可行或自然”，而是区分**来自专家数据分布与策略生成分布的差异**。即使策略在宏观运动形态上已经高度接近专家，其生成的轨迹仍不可避免地在状态分布、时序相关性以及高频细节等方面与离线专家数据存在系统性偏差。因此，判别器依然能够利用这些细微但一致的统计差异对两者进行区分，从而导致 Expert score 与 Policy score 在数值上保持较大的间隔。

        进一步地，当使用一个已训练完成的策略所生成的期望轨迹作为判别器的“专家参考”，而非原始的重映射动作捕捉轨迹时，可以观察到 Expert score 与 Policy score 分别收敛至约 0\.2 与 −0\.2。这一结果符合直觉，因为此时判别器面对的两类轨迹实际上来源于高度相似的策略分布，其可区分性显著下降。

![image\.png](Images_attachments/image%2013.png)

- **对抗网络下的模式崩溃问题**：当数据集中存在多种风格或多种行为的数据集时，actor只学会其中一种就能够拿到很高的风格奖励，自然没必要去探索其他的动作。**解决模式崩溃的几个关键点：**

    1. 尽量保证数据集中的风格统一

    2. 需要有一些其他的奖励（如仿人行走中的步态奖励），来避免actor陷入到局部最优值

    3. 判别器网络添加权重损失，防止网络过拟合到某个局部细节

    4. 平衡判别器网络的输入帧数，输入帧数太少，所能描述机器人运动的信息太少，时间帧过短

    5. 当风格很不一样时，可以在不同环境上采用不同的判别器网络

### 基本代码

```Python
# Networks
actor  = Policy()
critic = Value()
disc   = Discriminator()

for iter in range(num_iterations):

    # -------- Rollout --------
    s = env.reset()
    traj = []

    for t in range(T):
        a = actor(s)
        s_next, r_task = env.step(a)

        feat = motion_feature(s, a, s_next)
        traj.append((s, a, r_task, feat))

        s = s_next

    # -------- Discriminator --------
    feat_exp = sample_expert_motion()
    feat_pol = [x.feat for x in traj]

    L_disc = -log(disc(feat_exp)) - log(1 - disc(feat_pol))
    disc.update(L_disc)

    # -------- Style Reward --------
    for x in traj:
        r_style = -log(1 - disc(x.feat))
        x.r = x.r_task + λ * r_style

    # -------- Actor-Critic --------
    A, R = compute_advantage(traj, critic)

    actor.update( -log_prob(actor, traj) * A )
    critic.update( mse(critic(traj.s), R) )
```

### 方法局限性

正如AMP的原理是为了学习风格，这既是优势也是劣势。如果你想要做准确跟踪的模仿学习，只有AMP就不能给你一个很好的结果，因为AMP无论怎么样都存在一定程度的模式崩溃，他只会学习风格。与此同时，AMP相比于Deepmimic的方法对期望轨迹的质量更加敏感，如果你的期望轨迹本身存在打滑或者突变的一些情况，AMP很容易学不会。

另外，**不管是Deepmimic方法还是AMP的方法，都会存在对复杂地形和复杂任务的适应能力不足**，对于上下台阶，斜坡等的任务，仍然需要有对应的数据集作为参考标准，如果你的数据集中没有上台阶，只是平地行走的数据，无论是Deepmimic还是AMP在当前都没有很好的表现。

## （Data based）BFM行为基础模型算法

在本章中，笔者将开始介绍目前最前沿的算法框架，这些算法框架都是为了完成多动作学习（单策略实现多个动作）而诞生的，其中的TWIST/SONIC框架基于现有大量数据集的优势进行多阶段/单阶段的模仿学习，BFM\-Zero是基于无监督的FB算法框架（代替PPO）实现的模仿学习。

BFM（Behavior Foundation Model, 行为基础模型）这个名字，最早来源于Meta 2024年的文章*Fast Imitation via Behavior Foundation Models*, 只不过当时包括以前的BFM指的都是基于无监督学习算法框架FB（Forward\-Backward）的研究，直到**2025年的综述***A Survey of Behavior Foundation Model: Next\-Generation Whole\-Body Control System of Humanoid Robots*中，**将BFM概念扩展到三类范式的集合：FB、Deepmimic、Intrinsic reward。**其中FB和Deepmimic的相关方法后续会主要介绍，Intrinsic reward（内在奖励模型）指的是一类好奇心探索奖励，这个好奇心探索奖励也已经被集成在Isaac\_lab的rsl环境中，被称为rnd，感兴趣的读者可以自行探索。

![image\.png](Images_attachments/image%2015.png)

**BFM最明显的地方在于，其所使用的模型和算力与数据量呈现明显上升的趋势，背后揭示的是模仿学习\+强化学习的scale law，而这个scale law的成功有可能帮助人形走上通用道路**：

![image\.png](Images_attachments/image%2016.png)

> 上述提到的所有BFM相关文章均在：https://github\.com/myismyname/awesome\-bfm\-papers/tree/main
> 
> 

### 基于Teacher\-Student多动作学习

#### 方法原理

笔者也在很早的时候就开始入局人形的模仿学习，那个时候exbody和Human2Humanoid刚出新论文，当时还是单策略的单动作学习，因此笔者研究的方向从那个时候开始就是**单策略的多动作模仿学习**（不过说起来也很尴尬，由于没有性能很好的人形，笔者一年来没有什么实际的产出，其实还是笔者太菜了）。

对于单策略多动作的模仿学习，由于所在课题组一直以来一阶段学习的经验，笔者一开始希望的是能做出一个单策略，一阶段训练，并且学会多动作且具有鲁棒性的策略。在初期笔者成功在训练上同时实现了行走、奔跑、蹲起等动作，但在**后来加入踢腿、打拳这种高动态动作的时候，发现会导致机器人所有动作学习效果都在下降**，并且也发现奔跑数据集是一定程度有利于行走的。

尽管在后来笔者尝试过：

1. 在一阶段加入了类似于[ASAP](https://arxiv.org/abs/2502.01143)的学习课程，引导其收敛

2. 尝试将最终输出的Action由原来的单纯PD控制，改变到带有期望位置和期望速度的增量PD控制（即所谓的残差控制）

这两个方法确实对机器人的表现有更好的提升，但方法1牺牲了鲁棒性，方法2让机器人的表现好了很多（毕竟靠位置控制拉上去了），但最后总是不能很好学会所有动作。

**因此得出的结论是，单策略多动作学习通常需要同时训练整个数据集，而数据集中既有低动态数据，又有高动态数据，同时还有类似于单腿平衡这样准静态的数据，如果同时训练，会导致他们的梯度方向互相冲突而学不会**。

受限于算力资源，以及大家对Locomotion的算力都停留在单卡4090的时期，**基于Teacher\-Student方式就成为了一种很好的选择。其核心思想是**，既然同时训练所有数据集是有梯度冲突的，那么就分开训练，先在第一阶段用Teacher训练几个不同的策略，比如训练一个策略负责低动态动作，训练一个策略负责高动态动作，训练一个策略负责准静态动作，然后在第二阶段用Student将其蒸馏在一个策略里。

**一般的，基于Teacher\-Student方法的单策略多动作训练分为三个阶段**：

1. **Teacher训练多个策略（多个Teacher）**

    这个阶段核心就是怎么对数据集分类的问题，通常有两种方法。一种是人工预先分类数据集，你可以直接人工挑选出行走、奔跑、跳跃、踢腿、拳击、单腿平衡的数据集，如果没有标签的话也可以用K\-means这种无监督分类的方式进行分类；另一种是采用PHC论文的方法，需要训练一个evaluation的函数，初始化第一个teacher网络，然后每训练一段时间后评估一下，筛选出那些还不能学会的动作，并初始化第二个teacher网络，同时冻结第一个teacher网络，**即不断用新网络去学习越来越难的动作，该方法称为Hard mining机制**。

    **这里要注意，Teacher阶段的学习是模仿学习\+强化学习。**

![image\.png](Images_attachments/image%2029.png)

2. **Student蒸馏到一个策略**

    \(如果你在teacher阶段最后用一个加权网络/门控网络先把各个teacher策略合在一起了，那这里就直接蒸馏student，不用区分环境）

    你在Teacher阶段得到了多个teacher策略，由于不同动作是属于不同的teacher策略，**在蒸馏的时候就需要区分环境进行训练**。比如训练好了三个teacher行走、奔跑、跳跃，其中包含的数据集动作数量为行走10，奔跑20，跳跃30，那么你在蒸馏student的时候最好设置 N \* \(10 \+ 20 \+ 30\)个环境，比如6000个环境，其中1000个用于行走，2000个用于奔跑，3000个用于跳跃，和你动作数量一致。（如果不一致也可以训练，比如6000个环境平均分配各2000个环境用于不同teacher策略蒸馏，但笔者实测觉得这样效果更好，而且更符合“平均”\)

    **这里要注意，在蒸馏的过程中，Student的训练是一个纯监督学习的过程，即Loss单纯是Action loss。**

    > 也有一些Student蒸馏的Loss采用的是KL散度去拟合Teacher的分布，这也是一种路子，这样就可以不用第三阶段再去微调，但前提最好是你最终想要的策略更倾向于风格，而不是跟踪轨迹。
    > 
    > 参考TWIST论文：[TWIST: Teleoperated Whole\-Body Imitation System](https://arxiv.org/abs/2505.02833)
    > 
    > 

3. **强化学习微调（RL\-finetune\)**

    做这个步骤的核心原因，是**因为Student阶段的训练只是单纯的模仿学习，在没有强化学习奖励偏好的作用下，机器人在模仿过程中摔倒并不会让其觉得是“严重不能去做的行为”**，因此如果直接拿Student策略去部署实物，机器人摔倒的概率是非常大的，尤其是在你训练的人形机器人性能本身并不好的情况下。

    因此我们有必要进行强化学习的微调，微调的方式通常是在冻结Student的网络结构基础上，新初始化一个小的Actor\_res网络，我们同样采用和Student阶段相同的训练环境分配，但**在该阶段，我们只训练这个Actor\_res网络**。另外，在该阶段，我们的Loss除了PPO的Loss以外，**还可以加入一个KL散度的Loss用来正则化最终的Action输出分布**，让Actor\_res不会过多影响到整个动作的分布，而只是起到一个微调的作用。

![图片39\.png](Images_attachments/图片39.png)

    

    > 另外一种方法可以参考[Track Any Motions under Any Disturbances](https://arxiv.org/abs/2509.13833)中的做法，用World model去引导微调
    > 
    > ![image\.png](Images_attachments/image%2037.png)
    > 
    > 

#### 基本代码

```Python
# Teacher阶段训练多个策略
teacher_policy_i = mlp_i(obs_teacher) # mlp结构为[512,256,128]
loss = PPO_loss
# 可选多专家融合

# Student阶段蒸馏到一个策略
env_nums = sum(N * [motion_1_nums, motion_2_nums, motion_3 nums])
# motion_1_nums使用teacher_policy_1，motion_2_nums使用teacher_policy_2，motion_3_nums使用teacher_policy_3，
student_policy = DAgger(obs_student, teacher_policy)
loss = action_loss

# RL-finetune阶段
init actor_res = mlp
freeze(student_policy)
loss = PPO_loss + kl_loss
```

#### 方法局限性

采用Teacher\-Student方法去做多动作模仿学习的时候，受限于计算资源、硬件和视野，尽管最终做到了单策略下的多动作，但是局限性也比较明显：

1. 训练阶段长，工程量较大，优点是算力资源少，细节处理好的话，单卡4090就可以出不错的效果

2. 如果想要人形机器人通用化，就需要更大量的数据，用这种方法去实现通用化的工程量只会越来越大

3. 机器人的跟踪能力和鲁棒性在Teacher\-Student框架下容易被压缩和丢失

### 基于Deepmimic（SONIC\)

#### 方法原理

2025年末的一篇论文*SONIC: Supersizing Motion Tracking for Natural Humanoid Whole\-Body Control*探索了模仿学习\+强化学习的scale law，即**如果给足够多的数据，足够大的环境数，足够深的网络，强化学习就能学会几乎任何动作**。该篇文章的作者是CMU刚毕业的phd[ Zhengyi Luo](https://scholar.google.com/citations?user=lHPTxGsAAAAJ&hl=zh-CN&oi=sra)，其在phd期间的研究内容就是探索基于仿人橡皮人的多动作模仿学习，其很多工作（PHC, PULSE）都在指导着人形机器人的模范学习和强化学习算法。

![image\.png](Images_attachments/image%2011.png)

SONIC论文框架中表达的东西很多，比如扩大了数据集到100M帧（一亿帧数据），使用不同的encoder编码不同的模态，采用FSQ区分不同特征的动作，并结合上层的planner网络做一些相关的部署等等。**在笔者看来，SONIC最核心的东西是去证明模仿学习\+强化学习的Scale law**，这和Zhengyi Luo的phd答辩所讲述的他未来希望做的内容一致。

![image\.png](Images_attachments/image%2018.png)

- **笔者复现SONIC的尝试、结论**

    - **实验设置：**

        1. 只用一个mlp作为Actor网络在Unitree G1机器人上进行测试（没有encoder，actor输入完整轨迹），控制频率和数据集轨迹频率均为50Hz

        2. 采用beyondmimic作为底座，并魔改为多动作导入训练框架，多动作导入方式参考SONIC将其拼接为一个动作

        3. 分别尝试了0\.4M帧的LAFAN，10M帧的LAFAN\+AMASS进行训练

    - **实验结论：**

        1. 措施重要程度：自适应采样 \> 环境数 \> 网络参数容量 \> 未来帧

        2. 自适应采样方式和SONIC一致

        3. 环境数不够奖励无法收敛，或者说，收敛到不好的结果，验证了SONIC论文中的观点。大数据集下想要都学会需要一定数量的环境数\+一定程度深的网络

        4. 需要很大的显存容量，0\.4M帧需要8\*4096，10M帧需要48\*4096

        5. 未来帧对倒地爬起，机器人稳定性有较大帮助

        6. 训练数据量上升，机器人的鲁棒性增加，但如果数据质量不好，会导致跟踪效果变差。因此数据质量仍然是一个关键指标

        7. 与你人形机器人本体的性能也强相关，如果机器人本身PD不够，力矩不够或者运动范围不够，训练效果就不可能好

![image\.png](Images_attachments/image%2022.png)

#### 方法局限性

SONIC最有价值的事情就是**验证了模仿学习\+强化学习的scale law**，尤其是环境数不够奖励函数就无法收敛到你想要的结果这个结论可以被得到验证。但在笔者看来，这样力大砖飞的方法也有一些局限性：

1. 这意味着人形机器人运控也进入大数据和大算力的时代，算力缺口很大，很少实验室能有这么多的算力，笔者LAFAN\+AMASS采用了6块H20进行训练了5天

2. 前期的数据集处理需要花费很长时间，不管是从SONIC论文还是西湖大学的[GAE访谈](https://zhuanlan.zhihu.com/p/1982101283356841191)都能感受到处理和挑选数据集所消耗的成本

3. **交互不友好**，本质上Deepmimic方法学到的是一个动作字典，一旦你的动作out of distribution，动作容易变得很剧烈，比如跟踪走路但是突然倒地，要么在地上抽搐，要么突然弹起，动作幅度很大。这背后都是因为在训练的时候为了学出好的效果和数据利用率，做了Early termination，这一切导致了机器人害怕某些误差，因此在超过容忍的误差时，只想着最快恢复。（接下来的BFM\-Zero方法能从理论上解决该问题）

### 基于FB的无监督学习（BFM\-Zero\)

#### 方法原理

基于FB（Forward Backward）原理的无监督强化学习也是实现人形机器人的多动作学习并且可能通用化的一种方式。**其核心思想是，**

**在训练时，可以在没有任何数据集的帮助下，假设我随机采样的隐空间里已经包含了所有动作的特征，那就可以从隐空间Z中进行充分采样，采取某种方式将特征 Z 与机器人状态 S 绑定，并训练 Actor 。**

**在推理时，只要能将期望达到的状态 S\_goal 编码到对应的隐空间特征 Z 上，再将其输入 Actor 即可让机器人达到状态 S\_goal **。

25年底的论文[BFM\-Zero:A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning](https://arxiv.org/abs/2511.04131)是参考 [Meta motivo](https://github.com/facebookresearch/metamotivo) 橡皮人BFM项目的人形机器人实现。

- **FB训练原理**：特征隐空间采样，特征Z和机器人状态S绑定，利用FB（Forward Backward）算法更新参数

![图片40\.png](Images_attachments/图片40.png)

    **重要！！**：利用FB框架，Actor找到的action，是在当前状态s和特征z的约束下，能走到访问概率最大的下一个状态。尽管z在训练过程中是随机采样得到的，但是FB框架采用了B\(s\)后向网络在Actor收敛的过程中，也建立了不同状态s和特征z的关系。

    正是因为特征z在一开始并没有任何标签含义，我们只会把step后得到的下一时刻状态St\+1作为所谓的目标状态，因此理论上保证了动力学的连续，导致整个策略在训练出来后的动作都非常柔顺。

    实际过程中对于Z的采样通常是混合采样的形式，即均匀采样\+从B\(s\)网络采样，这样做可以让动力学更稳定。

- **FB推理阶段**：分为无微调推理和有微调推理

    - **无微调推理**：由于在训练的时候已经建立了不同状态s和特征z的关系，因此此时的Backward网络B\(s\)已经完全是一个可以由目标状态S\_goal编码出运动特征z的网络，因此可以实现以下三个场景：

![图片41\.png](Images_attachments/图片41.png)

        1. Goal reaching：设定身体任意自由度到达某个状态S\_goal，通过B\(S\_goal\)得到Z

        2. Motion tracking：设定跳舞序列\{S\_1, \.\.\., S\_k\}，每时刻推理更新B\(S\_k\)得到Zt

        3. Reward guide：设定奖励函数r\(S\)，通过E\(B\(s\)r\(s\)\)得到Z

    - **有微调推理**：在实际过程中可能面临Out of distribution的情况，比如没有见过的负重，会导致机器人无法正常完成动作，此时需要微调（Adaption），**重点是，这里的微调不用去微调网络参数，而是去微调S\_goal编码出来的特征z**。

        1. 假设要执行单腿站立，但是现在有负重导致机器人out of distribution，先正常通过单腿站立S\_goal推出Z\_init

        2. 构建损失函数J\(z\)，以最大化单腿站立奖励为目标，优化得到新的Z\_adap，把这个Z给到Actor

- **FBCPR训练原理**：虽然是无监督训练，但是如果在人形上使用该算法框架，训练出来的动作能实现Goal reaching、Motion tracking、Reward guide，却不一定具备拟人的动作（动力学连续，但可以不是拟人的动力学。**这也是BFM\-Zero这篇文章中，用到了LAFAN数据集配合GAN loss的意义，目的是为了正则化训练动作，使其更加拟人**。

    **FBCPR = FB \+ GAN loss \+ penalty reward\(碰撞，关节超限惩罚）**

![图片42\.png](Images_attachments/图片42.png)

#### 基本代码

笔者这里给出的是只有FB框架下的伪代码

```Python
# ===================== FB (Forward-Backward) 算法 核心更新框架 极简伪代码 =====================
def update(replay_buffer):
    # 1. 从经验池采样batch数据
    obs, action, next_obs, terminated = replay_buffer.sample_batch()
    discount = γ * (1 - terminated)  # γ:折扣因子，终止状态无后续收益
    
    # 2. 观测归一化 + 采样隐变量z
    obs, next_obs = normalize(obs), normalize(next_obs)
    z = sample_z(train_goal=next_obs)
    
    # 3. 核心步骤1：更新 前向映射F + 反向映射B 网络 (FB核心)
    fb_metrics = update_FB(obs, action, next_obs, discount, z)
    
    # 4. 核心步骤2：更新 Actor策略网络 (基于FB的隐空间Q值优化)
    actor_metrics = update_Actor(obs, z)
    
    # 5. 目标网络软更新 (τ:软更新系数)
    soft_update(F_target, F, τ)
    soft_update(B_target, B, τ)
    
    return {**fb_metrics, **actor_metrics}

# -------------------- 核心1：FB网络更新 (Forward映射 + Backward映射) --------------------
def update_FB(obs, action, next_obs, discount, z):
    # 无梯度计算目标值（用目标网络）
    with no_grad():
        next_action = Actor(next_obs, z).sample()
        target_Fs = F_target(next_obs, z, next_action)
        target_B = B_target(next_obs)
        target_M = matmul(target_Fs, target_B.T)
        target_M = aggregate_target(target_M, 悲观惩罚系数)
    
    # 计算当前网络输出 & FB核心损失
    Fs = F(obs, z, action)
    B = B(next_obs)
    M = matmul(Fs, B.T)
    diff = M - discount * target_M
    
    # FB核心损失：对角最大化匹配度 + 非对角最小化干扰
    fb_loss = 非对角误差平方损失(diff) - 对角误差均值(diff)
    # 反向映射B正交正则损失：约束B的正交性，提升泛化
    fb_loss += α * orth_loss(B)  # α:正则系数
    
    # 可选Q值辅助损失：隐空间时序差分损失
    fb_loss += β * q_loss(Fs, B, z, discount, target_Fs)  # β:Q损失系数
    
    # 梯度回传+更新F/B网络参数
    optimize(fb_loss, F, B)
    
    return {"fb_loss": fb_loss}

# -------------------- 核心2：Actor策略网络更新 --------------------
def update_Actor(obs, z):
    # Actor采样动作 + 计算隐空间Q值 Q = (F*z).sum
    action = Actor(obs, z).sample()
    Fs = F(obs, z, action)
    Q = (Fs * z).sum(dim=-1)
    Q = aggregate_target(Q, 策略悲观系数)
    
    # Actor损失：最大化隐空间Q值 → 损失取负均值
    actor_loss = -mean(Q)
    
    # 梯度回传+更新Actor参数
    optimize(actor_loss, Actor)
    
    return {"actor_loss": actor_loss}

# -------------------- 工具函数（极简抽象） --------------------
def soft_update(target_net, net, τ):
    # 软更新公式：θ_target = τ*θ + (1-τ)*θ_target
    target_net.params = τ * net.params + (1-τ) * target_net.params

def orth_loss(B):
    # B的正交正则：协方差矩阵 对角趋近1，非对角趋近0
    cov = matmul(B, B.T)
    return 非对角协方差平方损失(cov) - 对角协方差均值(cov)
```

#### 方法局限性

人形机器人采用FBCPR算法的好处在于它是天然动力学连续的，正如BFM\-Zero视频展示的那样，动力学连续导致机器人动作非常“友好”。

基于FB算法的无监督强化学习有可能是人形机器人的通用化方向，但也要**注意到目前该方法可能存在的局限性**：

1. 无监督是否能覆盖所有运动情况，取决于FB网络的大小，因为FB是一个理想概率空间下的近似投影

2. **由于FB原理导致动力学连续，因此很难泛化到不同动力学**（FB 会对所有观测动力学的未来占据状态分布取平均，不可避免导致策略表示干扰）。尽管实际过程中会对特征Z进行混合采样，能一定程度缓解这个问题，但在动力学突变大的情况下仍然存在，这也许是 in the air 动作难以实现的原因。

