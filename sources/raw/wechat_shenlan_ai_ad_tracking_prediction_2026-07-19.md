---
title: 自动驾驶核心算法盘点｜目标跟踪与轨迹预测篇
author: 深蓝AI
date: "2026-07-19 17:32:00"
source: "https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247603090&idx=1&sn=50660dccdee1fe7f77438eb839203156"
---

# 自动驾驶核心算法盘点｜目标跟踪与轨迹预测篇

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAKPXr0kicZddyXdPOg1Jm7tKusPLcWicG0ALpMqpjSHZxxsu45C13rzA4XZ2leKiaxG64fPqc9zIRIj8CR43YYibVy9ic8aRib3LUd8/640?wx_fmt=png&from=appmsg#imgIndex=0)

在之前的[《自动驾驶核心算法盘点》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzY4NjA5NTgyMQ==&action=getalbum&album_id=4596755873481310212#wechat_redirect)系列中，我们盘点了自动驾驶如何通过感知算法“看见”世界，以及如何通过 SLAM 与高精地图“记住”世界。但对于一个成熟的自动驾驶系统来说，仅仅“看见”这一帧画面里有几辆车、几个行人是远远不够的。

本文是该系列的第5篇，聚焦目标跟踪与轨迹预测。

前者负责把每一帧独立的检测结果串联起来，给每个目标发一张“身份证”；后者则负责基于历史轨迹和环境地图，预测目标未来的运动走向。

接下来，我们将为您盘点这两个领域中最具代表性的经典算法，看看自动驾驶是如何从“识别当前状态”走向“理解连续行为”的。

**欢迎关注【深蓝AI】**将持续分享人工智能领域前沿动态👇

***深蓝AI***

**1****—****目标跟踪篇：从 3D 极简基线到多模态融合**

## 在自动驾驶中，最主流的跟踪框架被称为 Tracking-by-Detection（基于检测的跟踪）。简单来说，就是先用 3D 检测算法找出每一帧里的所有目标，然后再用跟踪算法把前后两帧里属于同一个目标的结果连起线来。


**1. AB3DMOT：3D 空间中的极简基线**

早期的 3D 跟踪算法往往极其复杂，试图将 2D 图像特征与 3D 点云特征进行繁琐的匹配。2020 年，卡内基梅隆大学（CMU）的研究团队提出了 AB3DMOT（A Baseline for 3D Multi-Object Tracking），用极简的思路打破了这一局面。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAPlIND5saJCEG7N4uv9zy6sHdBYUgv4Zj7emJiaJyTk4sNB48fAgj9LfptnR3UpGw9aX6xT2eibg8kFt7C2ttkib0w2Uwz17EguI/640?wx_fmt=png&from=appmsg#imgIndex=1)

图1 | AB3DMOT 在 KITTI 数据集上的 3D 目标跟踪结果可视化。图中使用不同颜色的 3D 边界框和 ID 标识了连续帧中被稳定跟踪的车辆目标。©【深蓝 AI】编译

核心思路：AB3DMOT 的哲学是“大道至简”。作者发现，只要 3D 检测器的质量足够好，跟踪算法完全可以非常简单。它摒弃了所有复杂的 2D/3D 特征融合，仅仅使用了两个极其经典的数学工具：

1. 3D 卡尔曼滤波（3D Kalman Filter）：用来根据目标上一秒的 3D 速度和位置，预测它这一秒应该出现在哪里。
2. 匈牙利算法（Hungarian Algorithm）：用来计算 3D 检测框和预测框之间的 3D 交并比（3D IoU），并找出最优的匹配方案。

代表意义：AB3DMOT 证明了在高质量 3D 检测器的加持下，仅靠极其简单的 3D 运动学模型就能实现优秀的在线跟踪，且在 KITTI 数据集上速度高达惊人的 207.4 FPS。它确立了现代自动驾驶实时 3D 多目标跟踪的基础工程范式。


**2. CenterPoint：基于中心点的检测与跟踪一体化**

传统的 3D 检测与跟踪算法通常把目标表示为一个个 3D 边界框（Bounding Box）。但 3D 边界框有朝向角度，在匹配时计算 3D IoU 非常复杂，且容易因为角度预测的微小偏差导致匹配失败。2021 年发表在 CVPR 上的 CenterPoint 提出了一个极其优雅的视角。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpDvNYuTvJMh0IFKfWsqNDT4JfhC6RZkicicAUzWEheCB99EsAic3UkWehn0ZkYfWNVV5kLQdjtdINibJSXiaf2SQDBnzA4c3PONjicZg/640?wx_fmt=png&from=appmsg#imgIndex=2)

图2 | CenterPoint 的两阶段架构。第一阶段将点云转化为鸟瞰图（BEV）特征，通过中心点热力图（Center Heatmap）检测目标，并回归出速度和 3D 尺寸；第二阶段提取目标特征进行进一步的评分和边界框精修。©【深蓝 AI】编译

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpCMFJ50meP6HYQ58OVIZnOw0Ejxk2SGps6EuQ0xNq9aV2yUoOfCt2dhC4fVt8abyPlSeT7keUyoYzl5QjgjVBRw3JyPZ1cBmCw/640?wx_fmt=png&from=appmsg#imgIndex=3)

图3 | 传统 Anchor-based 边界框与 Center-based 中心点表示的对比。将目标简化为中心点后，不仅避开了复杂的朝向角匹配问题，还使得跟踪过程变得极其简单。©【深蓝 AI】编译

核心思路：CenterPoint 认为“万物皆可为点”。它不再直接预测 3D 边界框，而是先在鸟瞰图（BEV）上预测目标的“中心点”，同时附带预测目标的速度。在跟踪阶段，由于目标变成了点，系统只需要拿上一帧的中心点加上预测速度，得到当前帧的预期中心点，然后与当前帧检测到的中心点进行贪心最近邻匹配（Greedy Closest-Point Matching）。完全抛弃了复杂的 3D IoU 计算。

代表意义：CenterPoint 将 3D 检测与跟踪完美统一在了一个框架下，凭借其简洁高效的设计，在 nuScenes 和 Waymo 等大规模自动驾驶基准测试中取得了统治级的 SOTA（State-of-the-Art）表现。


**3. EagerMOT：相机与激光雷达的“急切”融合**

在真实道路上，激光雷达（LiDAR）对远距离目标的点云极其稀疏，容易导致跟踪断裂；而相机虽然能看清远处，却缺乏准确的深度信息。2021 年发表在 ICRA 上的 EagerMOT 提出了一种巧妙的多传感器融合跟踪策略。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpBEYs84XC8bE9jdJzu220uibiaUuxa7wdcw4QaoicLrUcnaicty8zyQI61aSAZ4n0b2iaHe7ticiaPZsNg0J4qgOAvqYC9UkdzY1ib1Kt4/640?wx_fmt=other&from=appmsg#imgIndex=4)

图4 | nuScenes 数据集中的自动驾驶多模态传感器视角。上方为激光雷达点云与 3D 跟踪框的鸟瞰图（BEV），下方为多相机环视图像中的 2D 投影框。EagerMOT 等算法正是利用了这种多模态互补性来提升跟踪的鲁棒性。©【深蓝 AI】编译

核心思路：EagerMOT 的名字来源于它的“急切（Eager）”策略：只要有任何传感器捕捉到了目标，就立刻将其用于跟踪。它设计了两次匹配流程：第一次，用高精度的 3D 激光雷达检测框去匹配 3D 轨迹；第二次，把剩下没匹配上的轨迹投影到 2D 图像平面，拿去和相机的 2D 检测框再匹配一次。

代表意义：EagerMOT 证明了在自动驾驶中，合理利用多模态传感器的互补特性，可以大幅提升跟踪的鲁棒性。它使得系统能够在目标距离极远、激光雷达失效时，依然依靠相机保持对其身份的稳定跟踪。



***深蓝AI***

**2****—****轨迹预测篇：从单体推演到意图驱动**

跟踪解决了“过去和现在”，而预测要解决“未来”。与跟踪不同，轨迹预测天然面临着多模态（Multi-modality）的挑战——在同一个路口，同一辆车完全可能合法地做出左转、直行或右转等多种不同的选择。


**1. Social LSTM：开启交互建模时代**

早期的轨迹预测往往把每个目标孤立看待，但现实中，行人和车辆的运动是相互影响的。2016 年发表在 CVPR 的 Social LSTM 是解决这一问题的先驱性工作。

核心思路：该方法虽然最初针对人群轨迹预测设计，但其核心思想深刻影响了自动驾驶。它为每个目标分配一个长短期记忆网络（LSTM）来处理历史轨迹，同时创新性地提出了“Social Pooling（社交池化）”机制。在每一个时间步，系统不仅看目标自己的状态，还会把周围邻居的隐藏状态聚合过来。

代表意义：它标志着轨迹预测从“单体运动学外推”正式迈入了“考虑群体交互约束”的深度学习时代。自动驾驶系统开始明白：预测一辆车的轨迹，必须同时考虑它旁边车辆的动向。


**2. Social GAN：显式生成多模态未来**

既然未来是不确定的，我们为什么一定要预测出一条“最准”的轨迹呢？2018 年的 Social GAN 引入了生成对抗网络（GAN），试图直接生成多种可能的未来。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpBI0KiaI4fMPAEhKj40DS1xYcKgdozuOGlJH9cCbrInpfO2Zs6sDuEU7OtZpSpoaANpjm4eGp3KCzeo3uX1JzqGCHCoicZLTVDUg/640?wx_fmt=png&from=appmsg#imgIndex=5)

图5 | Social LSTM 提出的 Social Pooling 机制示意图。在拥挤场景下，系统为每个目标分配一个 LSTM，并在每个时间步将邻近目标的隐藏状态（Hidden states）通过空间网格进行池化聚合，从而捕捉复杂的群体交互。©【深蓝 AI】编译

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/943LxrS8cpAiaDkb9AjPFjTX25MWSr8XywvAYElL25VChbiaOXb4cp8icz0WN6jEjwSvBtryDodo9lvFXRHdzxsjf2x8YH8qQfKzchtuIcJAD8/640?wx_fmt=gif&from=appmsg#imgIndex=6)

图6 | 基于生成式对抗网络（GAN）的轨迹预测结果。面对前方路口的多种可能性，模型能够一次性输出多条符合物理约束的预测轨迹（图中不同颜色的线条），显式覆盖了目标的多种未来意图。©【深蓝 AI】编译

核心思路：通过在网络中引入随机噪声，并使用判别器来“挑刺”，Social GAN 能够一次性输出多条不同的预测轨迹。作者还设计了一种特殊的损失函数（Variety Loss），鼓励网络生成的轨迹尽量分散，覆盖左转、直行等各种可能性。

代表意义：它将轨迹预测从“回归一个单一答案”推进到了“生成一组符合常理的未来解”，这与自动驾驶规控模块需要评估多种潜在风险的需求不谋而合。


**3. VectorNet：高精地图的结构化表达**

自动驾驶的轨迹预测不仅要考虑其他车，还要考虑道路结构（高精地图）。在 2020 年之前，很多方法把高精地图和历史轨迹画成一张“鸟瞰图（BEV）”，然后用卷积神经网络（CNN）去提取特征，这既耗时又容易丢失精度。Waymo 团队在 2020 年提出了标志性的 VectorNet。

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpCBMV0DnPzUhibwoKnekpae3rjRwuwePQwJ5eKQQvjAkd3r8RCicXn7Q6GPxhA3L8n9VbQnq3VL3JLgeeY2CRg3oxI0RnfAsWO7Q/640?wx_fmt=png&from=appmsg#imgIndex=7)

图7 | 栅格化表示（左）与向量化表示（右）的对比。传统方法将地图渲染成像素图像，而 VectorNet 直接将车道线、斑马线和历史轨迹建模为由关键点连接而成的向量（Vectors）。©【深蓝 AI】编译

![Image](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAozCGLQFMy0pk7m0pc83AoCbUMsw9SeGAbbSZ9IibX1WzAry0nAjSq0CV2eoeSAGKHU12WgfYQ9cMG8jWvSAPR8It7z4ibEMKYU/640?wx_fmt=png&from=appmsg#imgIndex=8)

图8 | VectorNet 的分层图神经网络架构。系统首先在每个局部元素（如单条车道线或单辆车轨迹）内部构建子图（Polyline subgraphs），提取局部特征；然后再构建一个全局交互图（Global interaction graph），对所有地图元素和交通参与者之间的关系进行高阶建模。©【深蓝 AI】编译

核心思路：VectorNet 彻底抛弃了图像渲染。它把高精地图里的车道线、停止线，以及车辆的历史轨迹，全部看作是一段段由点连成的折线（Polylines）。然后利用图神经网络（GNN）直接在这些几何向量上进行计算。

代表意义：这种向量化表示不仅大幅降低了计算量（参数量和运算量显著下降），还保留了极其精确的道路拓扑结构。VectorNet 几乎成为了后来所有主流自动驾驶轨迹预测算法的底层特征提取标配。


**4. TNT：以目标终点为驱动的预测**

同样在 2020 年，Waymo 团队在 VectorNet 的基础上，进一步提出了 TNT（Target-driveN Trajectory Prediction），为多模态轨迹预测提供了一个极其优雅的工程解法。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/943LxrS8cpBWWgbHQd5bomthZCpX8PayB7KCk58c5F2nX2tTYNCP6OC9Sb9mTwhWcWT4IEzqljfJ06F5vlgFpq9qGfDjiaG4zZuWl62goCew/640?wx_fmt=png&from=appmsg#imgIndex=9)

图9 | TNT 算法三个阶段的可视化示意。左图展示了在路口预测出的多个潜在目标终点（星号）；中图展示了基于终点生成的候选轨迹；右图展示了最终打分筛选出的高概率预测轨迹。©【深蓝 AI】编译

核心思路：TNT 的核心洞见是：车辆未来的轨迹，很大程度上取决于它想去哪个“终点”。因此，TNT 将预测拆成了三步：

1. 预测意图终点：在地图上撒下一堆候选点，预测车辆最可能到达的几个终点（比如左转车道的尽头、直行车道的尽头）。
2. 补全中间轨迹：既然知道了起点和终点，就把中间的行驶路径连起来。
3. 轨迹打分筛选：对这些生成的轨迹进行合理性打分，选出最可能的几条输出。

代表意义：TNT 将原本极难收敛的长时间轨迹回归问题，巧妙地转化为了“终点分类 + 短期路径规划”问题。这种“先猜意图，再画轨迹”的思路，极大地提升了预测的准确性和可解释性，成为工业界广泛借鉴的经典范式。



***深蓝AI***

**3****—****构建连续的时空理解能力**

从目标跟踪到轨迹预测，自动驾驶算法完成了一次认知维度的跨越：

- 目标跟踪（如 AB3DMOT、CenterPoint、EagerMOT）让系统拥有了“短期记忆”，能够在 3D 空间和多模态传感器中，把离散的检测框变成连续的物理实体。
- 轨迹预测（如 Social LSTM、VectorNet、TNT）让系统拥有了“未来想象力”，能够结合道路拓扑和群体交互，推演出交通参与者接下来的意图。

随着端到端大模型的兴起，感知、跟踪、预测和规控的边界正在逐渐模糊。但无论架构如何演进，上述经典算法所沉淀下来的核心思想——3D 极简状态估计、点特征表达、多模态融合、向量化地图约束以及意图驱动的多模态生成，依然是构建自动驾驶“时空理解能力”的基石。

编辑｜阿豹审核｜阿蓝

参考文献

论文

1.Weng, X., et al. (2020). 3D Multi-Object Tracking: A Baseline and New Evaluation Metrics. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

2.Yin, T., et al. (2021). Center-based 3D Object Detection and Tracking. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

3.Kim, A., et al. (2021). EagerMOT: 3D Multi-Object Tracking via Sensor Fusion. IEEE International Conference on Robotics and Automation (ICRA).

4.Alahi, A., et al. (2016). Social LSTM: Human Trajectory Prediction in Crowded Spaces. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

5.Gupta, A., et al. (2018). Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

6.Gao, J., et al. (2020). VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

7.Zhao, H., et al. (2020). TNT: Target-driveN Trajectory Prediction. Conference on Robot Learning (CoRL).

**往期推荐** Recommend [![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/943LxrS8cpAhYq85CXTeKEXodjfiaIUHfDfa8hBib0502WDIBrslJKic68cZC5IiaicIGdXxcCzrBWkfDnacLHu51TWqpBIg0BezPN8zyV3CHEJc/640?wx_fmt=jpeg&from=appmsg#imgIndex=0)](https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247602525&idx=1&sn=179072d10ad35c9c441927d095c3e381&scene=21#wechat_redirect)**近五年谁在 Science Robotics 上发文最多？盘点全球顶尖机器人实验室**[![图片](https://mmbiz.qpic.cn/mmbiz_png/943LxrS8cpAKPXr0kicZddyXdPOg1Jm7tKusPLcWicG0ALpMqpjSHZxxsu45C13rzA4XZ2leKiaxG64fPqc9zIRIj8CR43YYibVy9ic8aRib3LUd8/640?wx_fmt=png&from=appmsg#imgIndex=0)](https://mp.weixin.qq.com/s?__biz=MzY4NjA5NTgyMQ==&mid=2247602190&idx=1&sn=a9e9a29449a395f8c08f54f4c78fed06&scene=21#wechat_redirect)**3D目标检测经典算法全盘点：单目、双目、激光雷达****欢迎关注【深蓝AI】**持续分享人工智能领域前沿动态👇![图片](https://mmbiz.qpic.cn/sz_mmbiz_gif/943LxrS8cpCFreRWsn2fgjfEz7fB26oBpbfOsHK7zRA7xsBRS9mpSIvgQwOETOeicmb4PgKiby0nOGDo9ObI0JrvBflh4oibEdgwTEykKOSQ1w/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=16)
