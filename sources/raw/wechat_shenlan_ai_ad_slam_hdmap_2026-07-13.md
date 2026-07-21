---
title: 自动驾驶核心算法盘点｜SLAM与高精地图篇
author: 深蓝AI
date: "2026-07-13 17:32:00"
source: "https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247602862&idx=1&sn=9918db21b11a1d5fcec96482798bbff7"
---

# 自动驾驶核心算法盘点｜SLAM与高精地图篇

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAKPXr0kicZddyXdPOg1Jm7tKusPLcWicG0ALpMqpjSHZxxsu45C13rzA4XZ2leKiaxG64fPqc9zIRIj8CR43YYibVy9ic8aRib3LUd8/640?wx_fmt=png&from=appmsg#imgIndex=0)

在之前的[《自动驾驶核心算法盘点》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzY4NjA5NTgyMQ==&action=getalbum&album_id=4596755873481310212#wechat_redirect)系列中，我们已经深入探讨了规划与控制算法。

如果在陌生的地下车库，GPS 信号完全丢失，车辆该如何定位？在暴雨或大雪中，车道线被完全遮挡，车辆又该如何保持在正确的车道内？这就必须依赖自动驾驶的“记忆与定位”系统——SLAM与高精地图。

本文是该系列的第4篇，聚焦SLAM与高精地图。

本篇推文，我们将继续顺着自动驾驶系统的架构，为您盘点工业界与学术界最具代表性、引用量极高的经典 SLAM 算法与高精地图技术。

我们将尽量减少复杂的数学公式，用客观通俗的语言，带您看懂从传感器实时建图到高精地图在线生成的演进脉络。

**欢迎关注【深蓝AI】**将持续分享人工智能领域前沿动态👇***深蓝AI*****1****—******为什么我们需要 SLAM 与高精地图？****

###

## 人类驾驶员在熟悉的路上闭着眼睛都能开一小段，是因为大脑里有一张“地图”。对于自动驾驶而言，这张“地图”和“我在地图上的位置”同样不可或缺。

SLAM，顾名思义，就是让车辆在未知环境中，一边移动一边估计自己的位置（定位），同时把周围的环境绘制成数字地图（建图）。而高精地图（HD Map），则是 SLAM 建图结果的终极升华，它不仅包含三维坐标，还被人工或算法赋予了丰富的语义信息（如车道线、红绿灯、限速标志等），成为车辆的“超视距传感器”。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/943LxrS8cpDDTWbSibbG1JHEfL81u0UiaXAMdVyuGfNUjkQEthzuKicibZ8ibRibCxibRdSKJAA95UGGl0LW3wEonfbv76Js4G3hficVHsUJl7MKkOQ/640?wx_fmt=jpeg&from=appmsg#imgIndex=1)

图1 | 经典 SLAM 系统基本流程总览。系统接收传感器数据（如激光雷达、相机），前端（Front end）负责特征提取与局部里程计跟踪，后端（Back end）通过图优化（Graph optimization）和回环检测消除累计误差，最终输出精确的全局轨迹与环境地图。©【深蓝 AI】编译

***深蓝AI*****2****—****SLAM 篇：从视觉标杆到激光融合**

在自动驾驶领域，SLAM 主要分为两大流派：视觉 SLAM（基于相机）和激光 SLAM（基于激光雷达）。近年来，多传感器融合（尤其是激光与惯导融合）成为了绝对的主流。

### **1. ORB-SLAM2：视觉 SLAM 的经典标杆**

提到视觉 SLAM，就绕不开西班牙萨拉戈萨大学的 Raul Mur-Artal 等人于 2017 年发表的 ORB-SLAM2。作为视觉 SLAM 领域最著名、传播最广的开源系统之一，它在学术界积累了极高的引用量。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpAr1rSelz6bibX1SEOuZ3lpwOdAhHeWQV2ibZJ46M6ibGh3rudZ0T99Wh8ibVTOU6hgibIckGq7o9jLxp87hE4QIHvA2au8ftZBau4k/640?wx_fmt=png&from=appmsg#imgIndex=2)

图2 | ORB-SLAM2 系统架构与多线程工作流。系统巧妙地设计了三大并行线程：跟踪（Tracking）负责实时计算相机位姿，局部建图（Local Mapping）负责管理局部地图点，回环闭合（Loop Closing）负责在检测到曾经走过的地方时进行全局优化。©【深蓝 AI】编译

核心思路：ORB-SLAM2 的核心优势在于它极其完整的系统工程设计。它使用 ORB 特征点来提取图像中的关键信息，这种特征点提取速度极快，且对旋转和光照变化有一定鲁棒性。系统不仅支持单目、双目和 RGB-D 相机，还引入了“地图复用”和“重定位”功能。如果车辆在某个路口迷失了位置，它可以迅速在已有的地图中重新找回自己。

适用场景与局限：虽然 ORB-SLAM2 奠定了视觉 SLAM 的基础框架，但在真实的自动驾驶量产中，纯视觉 SLAM 容易受到光照剧烈变化（如进出隧道）、逆光或无纹理区域（如大白墙）的致命干扰，因此通常需要与其他传感器进行深度融合。

### **2. LOAM：激光 SLAM 的里程碑**

如果说 ORB-SLAM2 是视觉的标杆，那么卡内基梅隆大学（CMU）于 2014 年提出的 LOAM（Lidar Odometry and Mapping） 则是激光 SLAM 走向高精度、实时化阶段的标志性里程碑。

在 LOAM 出现之前，处理庞大的三维激光点云是一件极其耗时的事情，很难在车载计算平台上实时运行。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpC7ScoCXPqE3NpHg7AZvXnTY5nuDIKhk846icGZ2JNicrZn2SJaKUu3zaICZ4I6DYbLH50CKQQrkrzwDicZonvDgbzY0WxZic8Svew/640?wx_fmt=jpeg#imgIndex=3)

图3 | LOAM 系统的双模块解耦设计。左侧为激光里程计（Lidar Odometry），负责高频（如 10Hz）但低精度的快速运动估计；右侧为激光建图（Lidar Mapping），负责低频（如 1Hz）但高精度的全局地图匹配与注册。©【深蓝 AI】编译

核心思路：LOAM 的天才之处在于“解耦”。它把复杂的建图任务拆分成了两个并行的模块：

1. 高频的里程计（Odometry）：提取点云中的边缘特征（线）和平面特征（面），快速计算出两帧点云之间的相对运动，保证车辆在高速行驶时也能跟上节奏。
2. 低频的建图（Mapping）：在后台慢条斯理地将当前点云与全局地图进行极其精细的匹配，消除里程计积累的微小误差。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAATKY8jwSTDLzaiciajeYluOrT3wOhAGqrb55xD13RefvoebQ4V9iczs7By0tPRW6BZibpfZ6uYlYnsiaOytUyGM1iahAsvX8DFibMPI/640?wx_fmt=png&from=appmsg#imgIndex=4)

图4 | LOAM 算法在室外场景的点云建图结果可视化。通过特征提取与匹配，算法能够将离散的激光雷达扫描帧拼接成结构清晰、细节丰富的全局三维点云地图。©【深蓝 AI】编译

代表意义：LOAM 首次证明了仅靠激光雷达就能实现极低漂移的实时建图，深刻影响了后续包括 LeGO-LOAM、A-LOAM 在内的众多优秀衍生算法。

### **3. LIO-SAM：激光与惯导紧耦合的主流路线**

单纯依赖激光雷达也会遇到问题：比如在一条极其笔直、两边全是相同树木的高速公路上，激光雷达前后扫描到的画面几乎一样，算法会陷入“走没走”的错觉（即退化问题）。

为了解决这个问题，引入 IMU（惯性测量单元）成为了必然选择。2020 年提出的 LIO-SAM（Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping） 便是这一路线的集大成者。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpAUbSrCibTJs5qcRLvZQcnK5L5c8O5Trlw6hX2PJMCTe5cqygaruwwZtL4fsIibCemgiabKXxdLSeumpfr9h8bqwVoerxCtdOk5ibo/640?wx_fmt=png&from=appmsg#imgIndex=5)

图5 | LIO-SAM 因子图架构。系统将不同传感器的测量结果转化为因子（Factor），包括 IMU 预积分因子（橙色）、激光里程计因子（绿色）、GPS 因子（黄色）以及回环闭合因子（黑色）。这些因子共同约束着代表机器人状态的节点（蓝色圆圈），通过联合优化求得最优位姿。©【深蓝 AI】编译

核心思路：LIO-SAM 采用了一种叫做“因子图优化（Factor Graph Optimization）”的紧耦合框架。在这个框架里，IMU 不再是简单的辅助，而是与激光雷达处于平等的地位。

- IMU 提供极高频的运动预测，帮助激光雷达去除运动畸变，并在激光雷达失效的瞬间“撑住”局面。
- 激光雷达则定期提供高精度的位置观测，用来纠正 IMU 随时间发散的漂移（Bias）。此外，它还巧妙地将 GPS 信号也作为因子加入图中，彻底消除了全局的绝对误差。

代表意义：LIO-SAM 凭借极高的鲁棒性和出色的计算效率（据作者在项目说明中称，其处理速度可达实时的数倍），成为了目前自动驾驶和移动机器人领域最受推崇的建图与定位方案之一。

***深蓝AI*****3****—****高精地图篇：从静态先验到在线构建**

通过 SLAM 算法，我们得到了一张由点云或特征点组成的“几何地图”。但这还不够，自动驾驶系统看不懂点云，它需要知道哪里是车道线、哪里是人行道。这就需要高精地图。

### **1. 高精地图的多层结构**

高精地图通常被设计为多层金字塔结构，以满足自动驾驶不同模块的需求。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpCNAQ1EHE6VEWe8w3ZHvLJmRp11fHlZfCBNUBxBPvV2cGAVehJ0RErf0NDicpchcYDkiblW1BTQVgwRQM6oRuDgxDDd3Wo2HiaqIk/640?wx_fmt=png&from=appmsg#imgIndex=6)

图6 | 高精地图的多层结构示意图。从底层的普通导航地图（SD map），向上依次叠加几何地图层（Geometric map layer）、语义地图层（Semantic map layer）、地图先验层（Map priors layer）以及动态更新的实时层（Real time layer）。©【深蓝 AI】编译

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpDXYDtOZzf2PBOoRAiczBQ0wGaHEskEZZic6ibOwOmn7ms3icg9EHk3cx8BoIqlnv2KQgyrkwlI40G9y2odsqy96SnaIxVx4wxCRdg/640?wx_fmt=png&from=appmsg#imgIndex=7)

图7 | 高精地图内部的具体内容分层。包含了用于定位的路标（如交通标志、路灯杆）、车道级拓扑结构（如车道线、边界、通行规则），以及道路级模型（如坡度、曲率、交叉路口）。©【深蓝 AI】编译

高精地图通常用于支持厘米级定位，它就像铺在真实道路上的一条“虚拟铁轨”，让自动驾驶车辆在雨雪天气、传感器被遮挡的极端工况下，依然能沿着预设的安全轨迹行驶。

### **2. MapTR：在线向量化建图的新趋势**

传统高精地图的制作成本极其昂贵，需要专门的测绘车（搭载顶级激光雷达和高精度组合导航）满大街跑，然后再经过大量的人工标注。一旦道路施工，地图更新往往滞后，导致自动驾驶系统做出危险决策。

近年来，学术界和工业界开始探索一条新路线：既然车载传感器越来越强，我们能不能在车上实时“画”出高精地图？

2022 年提出的 MapTR（Map Transformer） 便是这一“在线建图”趋势的代表性工作。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpDwquLqyfIv7jkgZmu8kAR0rjeLqBh4y0QXjOtY5lq1ia83rJP6230L42icoTUp7mDq0Vu630DFhIibdNEkJdxHG791IHrQsDuaGM/640?wx_fmt=png&from=appmsg#imgIndex=8)

图8 | MapTR 的整体框架与预测结果。算法将多视角的车载相机图像输入 Transformer 网络，直接在鸟瞰图（BEV）空间下输出结构化的向量地图元素。右侧预测结果清晰地展示了人行横道（紫色）、车道分隔线（黄色）和道路边界（绿色），且与真实情况（Ground-truth）高度吻合。©【深蓝 AI】编译

核心思路：MapTR 摒弃了传统的离线拼接与人工标注流程。它将多路车载相机的图像送入 Transformer 模型中，通过强大的深度学习网络，直接在车辆周围的鸟瞰图（BEV）视角下，实时预测出车道线、道路边界、人行横道等元素的向量化坐标。这种方式将地图元素建模为有序的点集，网络不仅能识别出“这是一条线”，还能输出这条线的几何形状和走向。

代表意义：MapTR 及其后续研究，推动了自动驾驶行业向“重感知、轻地图”甚至“无图化”方向发展。车辆不再死板地依赖云端下发的高精地图，而是拥有了边走边实时构建局部高精地图的能力。

***深蓝AI*****4****—****地图并未消失，只是改变了形态**

回顾 SLAM 与高精地图的技术演进：

- SLAM 从纯视觉（ORB-SLAM2）走向激光雷达（LOAM），再进化为多传感器紧耦合（LIO-SAM），追求的是在任何极端工况下都不迷失方向。
- 高精地图 则从依赖专业车队离线测绘的静态图层，逐渐演变为依靠车端大模型实时生成的在线向量地图（MapTR）。

自动驾驶正在从“重地图依赖”走向“地图与感知联合建模”。但这并不意味着高精地图不重要了，相反，地图的生成和更新方式正在发生革命性的改变。未来的自动驾驶系统，必将是一个拥有强大实时感知能力，同时又能利用众包数据不断自我更新“记忆”的具身智能体。

编辑｜阿豹审核｜阿蓝

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
