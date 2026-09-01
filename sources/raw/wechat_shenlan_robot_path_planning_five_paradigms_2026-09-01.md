---
title: 机器人路径规划5大主流算法详解：MPC、强化学习、图搜索等
author: 深蓝具身智能
date: "2026-09-01 10:56:00"
source: "https://mp.weixin.qq.com/s/buw_88K8DlR9Tw70NDTzkw"
---

# 机器人路径规划5大主流算法详解：MPC、强化学习、图搜索等

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/kaugqJpv9nuCktylvYoMKHYNAVojoRUpfyf1py08JvUnkfPXArzj4t5bMiaS6RBCXHHGhf8xlyw8icHrJcjEyYoA/640?wx_fmt=gif&from=appmsg#imgIndex=0)

![Image](https://mmbiz.qpic.cn/mmbiz_gif/uwFbeBKoFGeOljbnsJJ3lRjw7icOtrk0kxWpdSoc2iaQMQZQASLHZjkGWKHBxYU5RNFbpSMoH56Lyic5Gt1zRZyaicjMK34nTEOB70OVzZiaWXsA/640?wx_fmt=gif&from=appmsg#imgIndex=1)

任务：机器人必须穿过一扇仅比自身宽5厘米的门

判断：五类路径规划算法同台实测，谁能在狭窄门缝前‘活’下来？

> 大家好，这里是【深蓝具身智能】。
>
> 本文出自《具身智能基础》专栏，是本栏目下的第11篇文章，聚焦机器人路径规划算法五大流派

---

[💙](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)[订阅《具身智能基础》专栏](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)

你的订阅和收藏，将支持我们把这件事持续做下去✨

机器人在真实世界中导航，面临的核心问题并不是单纯抵达目标：它还必须在拥挤环境、复杂障碍物和难以预先穷尽的突发变化中，持续做出安全、可执行的移动决策。

今天这篇文章，我们将沿着“如何解决不同环境难题”这条主线，梳理具身智能领域常见的五类路径规划范式：

基于图的搜索、基于采样的方法、人工势场法、最优控制方法，以及强化学习与数据驱动方法。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGdPHVweNiaJenmAQ3uibwR6ykhhKLhb3tqiaFSLqMj9T8UnQjxcHcMfoy6jqwLl2S4WfAXBgN3Jks32Lpoo6xvuAWJbfDAxWnVQuY/640?wx_fmt=jpeg&from=appmsg#imgIndex=2)

▲图1 | 路径规划方法的工程性分类。全局规划、局部规划与学习式方法关注的问题不同；本文据此将常见做法重组为五条技术路线。©【深蓝具身智能】编译

五条路线的起点不同，但都在回答同一个问题：

面对充满未知与约束的现实环境，机器人究竟依靠什么信息来决定下一步怎么走？

我们先从最直观的一类方法——在地图中搜索——开始。

文末阅读原文，系统学习高飞老师团队主讲的搜索类、采样类、基于动力学的路径规划算法。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4YWHX9iazKkdkgh363zh9GFAfZia4RWWoYhutUeS8g43MnicLMfe9kUAZg/640?wx_fmt=jpeg&from=appmsg#imgIndex=3)

## 图搜索：当有一张完整地图时，如何找一条可行的路？

## 图搜索的思路很清楚：先把连续的物理环境“切碎”，变成离散的网格或节点图，然后通过搜索算法在这些节点中寻找一条连通路径。

## 在地图较准确、代价定义明确的前提下，它能为机器人提供一条经碰撞约束筛选的全局路径。

Dijkstra 算法和 A\* 算法是这一流派的代表。

Dijkstra 按照从起点到各节点的累计代价依次扩展；在边权为非负时，它能找到最短路径，但会访问较多节点；

A\* 则加入启发式函数，在考虑已走代价的同时预估到终点的距离，因此会更优先地检查可能靠近目标的区域，从而减少不必要的搜索。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGeNQOyCfiaWIsYKXIpbbSabkLyUgJuaY5iaEGoVvJXIG53DuEzMx4STRMBiaud2Ge4Xkk8FHLEP5csooc2PdYSqficoSfiau9YZALd4/640?wx_fmt=jpeg&from=appmsg#imgIndex=4)

▲图2 | Dijkstra 与 A\* 的搜索范围对比。A\* 使用启发函数优先探索更接近目标的节点，因此通常比均匀扩张的搜索访问更少区域；前提是启发函数与实际代价相匹配。©【深蓝具身智能】编译

在自主泊车场景中，由于车位狭窄且车辆受限于阿克曼转向几何，传统 A\* 搜索出的折线往往无法直接行驶。

面对这一难题，一种典型的改进思路是：系统首先调用 Hybrid A\* 算法，在扩张状态时就将车辆朝向和最大转向角等限制考虑进去，得到一条满足运动学条件的粗路径；

随后，再将它交给底层控制模块进行拟合与平滑修整。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/uwFbeBKoFGcsa26aDJzXqjNiaQ2L3UUDIZTiaSOsEj14kibOiaep6F546wBRU8r8rIbOmKJ4icQia2aVlPf9xh6eJEQ1hFRRNw2WXbOpIOibDveIOE/640?wx_fmt=jpeg&from=appmsg#imgIndex=5)

▲图3 | 不同 Hybrid A\* 变体在狭窄并行车位中的规划轨迹对比；图中同时给出 Bi-RRT\*（下文就讲） 作为参照。将车辆运动学约束纳入搜索状态，有助于得到可行的入位轨迹。©【深蓝具身智能】编译

这种设计的重点，不是让机器人机械地走出一条折线，而是快速确认空间连通性。

但这一结果高度依赖预先建好的地图，如果在执行过程中遇到未建模的突发障碍物，由于缺乏对未来状态的预测，往往会导致机器人频繁停顿或规划失败。

此外，一旦把这种方法用于高自由度机械臂的三维避障，网格数量会快速增长，计算成本可能难以满足实时要求。

既然网格搜索存在维度灾难的限制，那么，有没有一种方法可以在高维空间中快速找到出路呢？

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibqgVaodXH45G6Pdbk9xSEsUtlicqgxKkAiaK0P8QzGwuLiatibYiaIagQoOg/640?wx_fmt=jpeg&from=appmsg#imgIndex=6)

## 采样方法：当空间维度太高，如何快速“钻”过去？

为了克服图搜索在高维空间中的算力瓶颈，基于采样的方法应运而生。

这类方法不再死磕精确的网格划分，而是在状态空间中随机或伪随机地“撒点”（采样），然后通过碰撞检测把这些安全的点连起来，拼凑出一条路径。

RRT（快速探索随机树）是该流派的标志性算法。

它以起点为根节点，在空间中随机采样并不断向外长出“枝干”，直到触及终点。

这种采样策略使它成为处理高维运动规划的常用工具。

比如在移动操作任务中，让带有 7 自由度机械臂的底盘机器人从杂乱的桌面上抓取水杯，构型空间维度极高，传统的网格搜索会直接失效。

此时，工程上常会使用 RRT 或 PRM（概率路线图）一类方法，在关节空间中采样并进行碰撞检测，为机械臂寻找避开其他物品的可行轨迹。

近期的机械臂避障研究常通过目标偏置和受限空间采样来减少无方向的扩张，并对候选路径做平滑处理。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGe9eHvbic1NicYJQkXRGic9sK55z2iaYa9BCQc6BDJuTSicC7M1bQIo62Khmx8BCyBma3t7oewyx198UpwrIE8MFx2BtEt8ZgaPsjl8/640?wx_fmt=png&from=appmsg#imgIndex=7)

▲图4 | 四类障碍环境中的 RRT 系列路径对比。不同颜色对应 RRT、RRT\*、RRT\*-Connect 与改进算法；在狭窄通道和迷宫中，随机采样容易形成绕行与冗余节点。©【深蓝具身智能】编译

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGfEMfiaDDzAabhAK3Xz93MM8tjBnDV1tVYt5TTNxicvibg0t2MvrOwLN3qxQVWXQNZyT1FXIwjq2Euv1pn4k5n0icJZCTVich2D8mjc/640?wx_fmt=png&from=appmsg#imgIndex=8)

▲图5 | 真实桌面环境中的机械臂避障执行序列。机械臂需要在彩色障碍物间抬升、绕行并移动至目标位置，展示了采样规划在高维构型空间中的避障能力。©【深蓝具身智能】编译

但采样方法也有让人头疼的“失效条件”——“狭窄通道”问题。

当机器人必须穿过门缝或密集障碍物间隙时，随机采样命中该区域的概率极低，容易导致规划超时。此外，由于采样的随机性，每次生成的路径轨迹都不一样，且常常是锯齿状的折线，实际部署时必须额外串联一个路径平滑模块。

面对快速移动的动态障碍物，采样方法的重规划实时性也往往难以保证。那么，当危险突然逼近时，机器人该如何做出瞬间的本能反应？

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibTvia4qLjYyoM2Do58jX9J71HickLLA3NxCQp6fPljkgY26WIeaoeeYVQ/640?wx_fmt=jpeg&from=appmsg#imgIndex=9)

## 人工势场：当遇到突发障碍，如何凭本能做出瞬间反应？

人工势场法（APF）借鉴了物理学中电磁场的概念。它将目标点变成产生“引力”的源头，障碍物变成产生“斥力”的源头。机器人在两种力的叠加下，顺着合力方向（势能下降的方向）“滑”向目标点。

在工程应用中，服务机器人常需要应对突然闯入视线的行人或障碍物。

此时，全局规划未必能在每个瞬间完成重算，而人工势场法可作为底层反应式避障的一个选择。

当雷达或相机检测到靠近的障碍物时，系统可根据相对位置与速度调整斥力项，使机器人改变航向或减速让行。

这种方法的数学模型直观、计算量较小，适合需要快速更新的近场避障。势场产生的方向信息也便于转换为底盘的速度或转向参考。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGff0CVagy8s08BuB9mdVDQsgFicnJsmjy1PlTSgs88Cq3FK3gN4ntJwicQFFfDM48jNQ87gmicar1JN943pKOlw2hHyfxabehe3Xs/640?wx_fmt=jpeg&from=appmsg#imgIndex=10)

▲图6 | 多障碍环境中的势场路径比较，每个圆形障碍物周围的紫色环可以理解为这个障碍物产生的“碰撞磁场”，机器人一旦靠近这些“碰撞磁场”就会受到“人工势场的作用力”，从而被排斥远离障碍物。不同颜色轨迹对应不同变体，能够看到势场设计会直接影响绕障半径、转弯幅度。©【深蓝具身智能】编译

但它有一个典型限制——“局部极小值”。

当机器人在“U”型障碍物内部，或引力与斥力恰好抵消时，会陷入停滞甚至原地打转。此外，如果目标点紧贴着墙壁，强大的斥力会将机器人推开，导致永远无法抵达终点。

因此，实际系统常将 APF 放在全局路径之后，作为近场修正或局部避障的一环。

无论是全局搜索还是局部反应，几何上避开障碍往往还不够。对高动态机器人来说，脚能否落稳、电机能否提供足够力矩，必须一并考虑。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/uwFbeBKoFGeMoJGMSnz3eiakJ5TnkPaUKees0o3Jttg9LCwJpXQOfSSBstuJicYLmIR9Q0VLBiamQ4xXaNQTE7JYvVTndgBl4OvMEIP5ngpbY4/640?wx_fmt=gif&from=appmsg#imgIndex=11)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4yZibwaOWpB3DrcxuiafpXicx2ibHiaHAZFr7ptU6ud2hsxgCXvV0JGHtTDw/640?wx_fmt=jpeg&from=appmsg#imgIndex=12)

## 最优控制：当动作逼近物理极限，如何保证每一步都踩稳？

随着四足、人形或高速移动平台的发展，路径规划必须严格遵守物理极限。最优控制方法将规划变成了一个带约束的最优化数学问题。

模型预测控制（MPC）是这一类方法中的典型代表。

它会在有限时间窗口内预测系统状态，解一个带约束的优化问题，只执行当前最优动作的第一步，下一周期再根据新状态滚动更新。

在四足机器人的高动态控制中，MPC 是常见路线之一。腿足机器人需要持续决定落点、关节力矩与机身姿态。研究者们常在多种地面与扰动条件下验证其四足机器人的控制框架，以考察模型预测与反馈机制对外部环境的适应能力。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGdEm83mhzeIdHgq3qwndSEkmL6g08MFUmJEKGBwaANKVslwUjibic2qYYlKWWjmyPTS0yFzwfmpAzQxoHJdG4My5sIJhYeObQkJ8/640?wx_fmt=jpeg&from=appmsg#imgIndex=13)

▲图7 | 四足机器人在跑道、草地、人工草坪与实验室环境中的测试场景。MPC 可将模型与物理约束纳入滚动优化，并据状态反馈持续调整控制量。©【深蓝具身智能】编译

MPC 的价值在于能把电机扭矩、摩擦条件等约束显式写入优化问题，使生成轨迹更贴近动力学可行范围。但代价同样明确：它依赖模型质量与快速求解器，面对未建模接触、传感误差或算力不足时，性能可能明显下降。

如果物理模型难以精确建立，环境规则又无法穷举，机器人还能怎么学导航？

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4hKdH0P2rRHX1TlxUqlAx7X6m2hcl7XttttyRW05mhbEa1msX7zEzvw/640?wx_fmt=jpeg&from=appmsg#imgIndex=14)

## 强化学习与AI：当环境规则无法穷举，如何靠直觉导航？

数据驱动的 AI 技术为路径规划提供了另一种思路。深度强化学习（DRL）让智能体在仿真或真实环境中通过奖励信号学习策略，可直接把激光雷达、相机等观测映射为速度或转向动作。

在具身智能研究中，这类方法常用于传统建模难以完全覆盖的非结构化场景，例如让移动机器人在拥挤室内空间中学习局部脱困策略。

相比手工设计每一条规则，策略可从大量交互中学习近似的避障反应。

但落地仍有明显门槛：

- “Sim2Real”差距可能使仿真中有效的策略在真实环境中性能下降；
- 神经网络的决策也较难解释，增加了验证与安全评估的难度。

因此，更稳妥的工程路线往往不是完全用神经网络替代规划器，而是让大语言模型（LLM）等 AI 技术负责高层语义理解。

例如，在家庭辅助机器人中，当用户下达“去书房拿杯子”的指令时，系统结合感知算法在地图上锚定家具和物品的位置，为 LLM 提供空间语义锚点。LLM 将任务拆解为子目标后，底层的导航模块依然会调用传统搜索与控制算法去规避物理障碍。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcJhSsS0RPRnAzzPDyyXwQBvtxWiaDSup1YPFDh1u1Mz1OLoU7wVJa60iaIgzhVbNwP0q4hjQZt4fvHg15qz5ic4UHGUR3bQsMsq8/640?wx_fmt=png&from=appmsg#imgIndex=15)

▲图8 | 室内物体三维检测与语义标注。将家具和物品位置锚定到地图后，可以通过LLM在地图中识别用户指定的目标物体的位置，从而获得地图的起点和终点，随后系统再调用传统的路径规划算法，即可高效的解决复杂的视觉-语言-导航问题。©【深蓝具身智能】编译

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX7OGyOG8gzibCyLX91kbhEcWl0mnLk5Zb5uVRabIn51LEKNicYT8OlZZQ/640?wx_fmt=png&from=appmsg#imgIndex=16)

## 从选择算法，走向分层协同

回看这五条路线，可以发现它们解决的是机器人移动过程中的不同问题：

图搜索提供全局连通性，采样方法解决高维可行性，人工势场处理近场快速反应，最优控制保证物理约束下的动态稳定，而 AI 方法补充了感知与语义决策。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGdZXsFjtDDR29VhiceNicDuvRiaUkfTlhM6zu1Jom4nOdElqDzGoS9gNGbLpw0kjrl69DEeeQGg9zswibr4CQjh5SfHFSt0nFWQYick/640?wx_fmt=png&from=appmsg#imgIndex=17)

### ▲图9 | 用“狭窄通道”场景同时检验五类算法。©【深蓝具身智能】编译

在实际的具身智能系统中，这些方法往往需要配合使用：

先用A\*（低分辨率）确认门的位置，再用RRT在门附近精细化采样，同时底层始终开着APF做防撞，MPC负责执行微调。

因此，与其寻找一套能够解决所有路径规划问题的完美算法，不如回到更实际的问题：在给定任务、算力和安全边界下，什么算法组合最可靠？

本文梳理的路径规划范式，在高飞老师（微分智飞创始人&CEO，浙江大学控制学院长聘副教授、研究员、博士生导师，入选国自然青A）主讲的《移动机器人运动规划》课程中有系统讲解——从搜索、采样到动力学规划与MPC。

阅读原文，从“知道有什么算法”走向“能搭建一套可部署的规划系统”

编辑｜阿豹

审编｜具身君

 ****推荐阅读**
[![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcibJS8986MfCcVATGOkcK6lNQfiaTORbuhSFoATTmZ5kA6nV8l8REia7nm4A4OxC1yOePBqrzWHQQd0ALicYANgOoNmRbibChjcAuQ/640?wx_fmt=png&from=appmsg#imgIndex=18)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=3824573915845640194&scene=126#wechat_redirect)[![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGcRfEtsGjVkl7cXB7QYAAib4wOMhdRcvsQicHnmiaxqoibw9LUCGGcPGSYnUPeUlZEoiaBlQezclFhp5yZQ6yLcLAjYeI67pJmvhOMw/640?wx_fmt=jpeg&from=appmsg#imgIndex=19)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=944555238&lang=zh_CN#wechat_redirect)

**![Image](https://mmbiz.qpic.cn/mmbiz_png/qKE443uRvLo6ic3ZPUttmFZ2AefQ4wjHSlQluSDkaxL9icWicpPYYmpo1Wa37Scjhh4AS5VwYJtmlTf5cKMiaIXg5g/640?&random=0.17349735674179656&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&wx_fmt=other#imgIndex=20)**

**【深蓝具身智能】****的原创内容均由作者团队倾注个人心血制作而成，希望各位遵守原创规则珍惜作者们的劳动成果；未经授权禁止任何机构或个人抓取本账号内容，进行洗稿/训练，否则侵权必究⚠️⚠️**


![Image](https://mmbiz.qpic.cn/mmbiz_png/Nabxc8rdYriaKqxCUjcZ8sSCnSNlWpqdI1kyXXQjXbtv95xvACqQoqL2ibbKXt9PB0FLPibKiawGsTcQrnKDGWVw2Q/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1#imgIndex=21)

点击❤收藏并推荐本文**
