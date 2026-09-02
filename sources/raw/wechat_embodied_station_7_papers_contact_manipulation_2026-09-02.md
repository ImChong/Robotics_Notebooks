---
title: 7个开源论文一次看完：具身智能机器人最新论文速览
author: 具身智能小站
date: "2026-09-02 14:01:26"
source: "https://mp.weixin.qq.com/s/v2-G3TNZV5e_Uzm0kHPZEA"
---

# 7个开源论文一次看完：具身智能机器人最新论文速览

📅 2026年9月2日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 7 篇 **机器人**与**具身智能开源**论文，覆盖**接触**丰富精密装配、模仿学习**时间鲁棒性**、**神经符号**运动规划、**单目深度**综述、多视角**点云配准**、高精度插入 **benchmark**、**非抓取投掷**，以及低成本 **IMU** 倾角估计。整体来看，这批工作最值得圈内人关注的共同点，是**项目**和**代码**入口相对完整，方便继续复现、评测和二次开发。

**综述主线：****具身智能**正在从“单一大模型策略”走向更工程化的**开源**闭环：**接触**力、时间尺度、符号推理、几何感知、基准硬件和低成本传感器，都在重新进入**机器人**系统设计的核心位置。

01 · arXiv:2609.01596

🔬 ****Facet-0**: A Robotic Foundation Model for Contact-Rich Precise Manipulation**

📌 **Embodied AI · Contact-rich Manipulation · Robotic Foundation Model · RL**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYKEibibTV6vGP8mOCLzBwaJZic70ibSiaJ8mpthKyQmRicY9rFHqibnzdhVowg7UzicUbPWxgfMHrAV1jFqIiaicspJC45N4ibORp3W9GFKs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

✨ 把动作和腕部力共同建模，面向亚毫米级精密装配。

📖 真实机器人装配在亚毫米公差下需要空间精度、柔顺交互和对**接触**失败的鲁棒性。**Facet-0** 提出一个预测并评估动作**接触**后果的机器人基础模型：以联合 **action-wrench** proposal 为核心，将因果力历史与视觉语言语义、运动学状态对齐，并用 **flow matching** 同时生成动作片段和预期未来腕部力曲线。部署 rollout 进一步训练 distributional **Action-Wrench Critic**，区分任务进展相近但**接触**结果不同的动作。该系统基于 **ManuFacet-1K** 训练，在 5 个亚毫米计算机装配任务上达到 **82%** 平均成功率，对比最强 baseline 为 **15%**，同时具备 **0.5 mm** 放置精度和 **50 ms** 指令延迟。

💡 精密操作的瓶颈不只是看懂场景，而是预测**接触**后果。

🔗 项目链接： https://pine-lab-ntu.github.io/facet-0/

🔗 资料来源： https://arxiv.org/pdf/2609.01596

02 · arXiv:2609.01453

🔬 **Does **Imitation Learning** Preserve Temporal Robustness in Dexterous Manipulation? An Expert-Learner Comparison Across Task Execution Speeds**

📌 **Embodied AI · **Imitation Learning** · Dexterous Manipulation · Temporal Robustness**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgub8bWsKHScw5zRw7hn0Tv36KRAIOG7b0EicbsEzaEDV4yQiaAth5EjasEsEbXI7vpiaL1ibajHsHpLf9wn7frzqRicDygeybjBcavTs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

✨ 同样标称成功率下，模仿策略未必继承专家的**时间鲁棒性**。

📖 **灵巧操作**中的模仿学习策略通常评估场景、物体或指令变化下的鲁棒性，但较少检查不同任务执行速度下的表现。论文在 **ParcelStow** 接触丰富任务中比较脚本专家与 **ACT** 学习器：机器人需要获取、重定向并插入包裹，两者在标称速度下都达到 **100%** 成功率；但在示范速度范围的最高速度处，专家成功率为 **84%**，**ACT** 仅为 **53%**。两个不同初始化的 **ACT** policy 都出现类似退化，且最高速度下 47 次失败中的 35 次为**插入错位**。结果说明，标称成功率相同并不意味着学习器保留了专家跨执行速度的性能。

💡 模仿学习评测要看速度维度，否则会高估真实稳定性。

🔗 项目链接： https://github.com/coenwerem/parcelstow

🔗 资料来源： https://arxiv.org/pdf/2609.01453

03 · arXiv:2609.01260

🔬 ****Dual Process** **Motion Planning****

📌 **Embodied AI · **Motion Planning** · Neuro-symbolic AI · **System-1**/**System-2****

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYIc11AP9pWOCWE7DvrGH8sXlyqnW98eow5YLZPqRhAV7MjNoBicgazloQPAbB78y685vuTlKkRW1WGw9iapUicN6HrWbpnwFHHsU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

✨ 用**元认知控制器**协调快直觉与慢推理，提升非线性运动规划。

📖 机器人系统需要兼顾速度、精度和可靠性；经典控制与规划方法有较强保证，但效率和适应性受限，学习方法则能利用经验加速决策。**Dual Process** **Motion Planning** 从**神经符号**视角桥接两者，受 Thinking Fast and Slow 启发，将先进**符号求解器**作为 **System-2**，并引入经验驱动的 **System-1** 模块。**元认知控制器**动态选择何时依赖快速直觉、何时调用更慢但更精确的推理。论文在多类非线性 benchmark 环境中评估，显示该架构在规划效率、准确性和**泛化**性上带来稳定增益，并促进跨任务复用。

💡 运动规划不必在学习和求解器之间二选一，关键是调度机制。

🔗 项目链接： https://github.com/verayannn/System-1-and-System-2-in-Motion-Planning

🔗 资料来源： https://arxiv.org/pdf/2609.01260

04 · arXiv:2609.01172

🔬 ****Monocular Depth Estimation** from a Single Image: Progress and Opportunities**

📌 **Robot Perception · **Monocular Depth Estimation** · **Foundation Models** · **Survey****

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYosdHa6yqdzNPgMaloloHbfEqic4ULD3bzPerjXGyunW2d2PCLEIia1tNMqiaGYWiacmoPDpB2XhUTtAjuGHKxibaibM72uEaAzgCJI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

✨ 系统梳理**单目深度**从早期学习方法到基础模型时代的进展。

📖 **单目深度**估计是计算机视觉中的基础挑战，支撑 3D 重建、机器人、自动驾驶和增强现实等应用。该综述从问题定义出发，区分相对深度与 **metric depth**，梳理十年来影响研究的关键挑战，并整理室内、室外和**合成数据**集。在基础模型时代之前，文章总结了提升准确性、效率和鲁棒性的代表方法；随后转向 foundation-model-based 方法，将其归为判别式与生成式范式，并强调大规模预训练和**合成数据**的重要作用。文章还比较代表模型的定量与定性表现，讨论视频深度估计延展，并覆盖**视觉 SLAM**、内容生成和**机器人感知**等应用。

💡 机器人空间智能的底座，仍离不开可靠且可评测的深度估计。

🔗 项目链接： https://github.com/CVMI-Lab/Depth\_Survey

🔗 资料来源： https://arxiv.org/pdf/2609.01172

05 · arXiv:2609.01089

🔬 **Adaptive Depth-Map-Guided **Bundle Adjustment** for Correspondence-Free Multi-View Point Cloud Registration**

📌 **Robot Perception · Point Cloud Registration · **Bundle Adjustment** · Industrial Robotics**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYmAkjXicicgTfSzns9w56icdKnrZicHdshCMOlOHIIPibCiaNdDMTR8gIcANH45bokIU81ne9fctibpfBSHMzibtXibBosI8l68RXgSvIo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

✨ 绕开显式特征对应，用**分层深度图**做多视角**点云配准**。

📖 不规则**废钢**机器人处理需要密集 3D 测量来替代危险切割工位中的人工目检，重建误差会直接影响尺寸估计、边界几何、预热/切割区域和避碰火炬路径。现有多视角配准通常依赖特征提取和数据关联，但光滑金属表面、重复结构、遮挡和局部重叠会引入错误对应。论文提出 adaptive layered depth-map-guided bundle adjustment，用全局 **2.5-D** 网格表示场景，每个网格自适应维护多个深度假设；原始深度观测直接投影到全局地图形成约束，并用 **softmax** layer assignment 处理冲突深度。非线性最小二乘联合优化传感器位姿和**分层深度图**，在自采工业数据上保持竞争性精度、鲁棒性和低计算成本。

💡 工业**点云配准**的关键，有时是少信任特征对应、多利用深度约束。

🔗 项目链接： https://github.com/YiranZhou-Robotics/ADM-BA.git

🔗 资料来源： https://arxiv.org/pdf/2609.01089

06 · arXiv:2609.00906

🔬 ****Peg-in-Bench**: A Modular Benchmark for High-Precision Robotic Insertion**

📌 **Embodied AI · Robotic Benchmark · Peg-in-hole · High-precision Insertion**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaU5AUUQ7BEVdTFbeyuFSUOf0uRfBWsNgWyZ3Np8aibFLyjL0pd8vwQ7k2NpgEC5ec5WOXaVMzV210Gp1NIkzG3fKKsByD1ibWWU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

✨ 可 **3D 打印**、可重构的插孔基准，专测**高精度插入****泛化**。

📖 **高精度插入**因严格对齐要求和接触丰富交互，一直是机器人操作中的核心挑战。虽然 **peg-in-hole** 常被用于评估，但现有 **benchmark** 多依赖固定任务配置，难以衡量不同插入场景中的鲁棒性和**泛化**。**Peg-in-Bench** 提出可重构插孔基准，由完全可 **3D 打印**的**模块化**组件组成，包含多种 peg 几何、公差等级和可配置底座结构，可组合生成大量插入与装配任务。通过改变物体布局、朝向和任务结构，同时控制物理条件，该基准支持系统化评估未见场景适应能力；项目还提供**场景生成工具**、标准化任务配置、机器可读任务描述和 **STL** 文件。

💡 高精度操作要进步，先得有可复制、可变化的硬件基准。

🔗 项目链接： https://github.com/aistairc/peg-in-bench

🔗 资料来源： https://arxiv.org/pdf/2609.00906

07 · arXiv:2609.00771

🔬 ****Non-Prehensile Throwing**: A Reinforcement Learning Perspective**

📌 **Embodied AI · Reinforcement Learning · Non-prehensile Manipulation · Sim-to-real**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguatMQssbtb6b6TNv1nuWoj7AV5WfPibKD9FIkiaylzDYpUJn9HUYOMlGL4icn7t4zspGgZLOvtyIQjibgVIUXrfZgTXiae2vL6QUktM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

✨ 不抓也能投，**RL** 直接优化关节轨迹扩展机器人可达空间。

📖 机器人投掷能实现快速物体转运，并把可达空间扩展到传统 pick-and-place 之外；相比抓取式投掷，**非抓取投掷**更适合大、重或可变形物体。已有方法常依赖简化接触模型和低维轨迹参数化，限制了解质量和可达范围。论文提出强化学习方法，额外利用**滑动**与**滚动**接触模式，直接优化关节空间轨迹，无需解析接触模型或定制参数化。MDP 被建模为随目标、物体模型和初始构型演化的关节状态动力系统；离线规划 **joint-jerk** 轨迹，并上采样为部署用高速平滑速度指令。仿真中策略在数千配置上达 **99%** 成功率并泛化到未见物体；零样本部署到 **UR5e** 后，可将 **790 g** 重物和 20x20x28 cm 大物体投到 **350 cm** 距离或 180 cm 高度目标，真实成功率 **97%**。

💡 非抓取操作打开的不是一个技能，而是一类新可达空间。

🔗 项目链接： https://abdullah-aist.github.io/NP-Throw/

🔗 资料来源： https://arxiv.org/pdf/2609.00771

**综合观察**

综合来看，这 7 篇论文没有把**具身智能**简单等同于更大的 policy，而是把真实**机器人**会遇到的问题拆得更细：Facet-0 聚焦**接触**后果建模，ParcelStow 追问模仿学习是否保留专家的**时间鲁棒性**，Dual Process Motion Planning 把 System-1 与 System-2 接到非线性规划，Depth Survey 和 ADM-BA 分别补齐**单目深度**与工业**点云配准**，Peg-in-Bench 提供可打印高精度插入基准，NP-Throw 把**非抓取投掷**推向零样本实机。对开发者而言，这批**开源**入口的价值在于：它们能让算法讨论更快落到**可复现**系统、标准化评测和真实部署边界。
