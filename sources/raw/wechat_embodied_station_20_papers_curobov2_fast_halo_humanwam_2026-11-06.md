# 只在虚拟房间练习，人形机器人走进陌生房间能做什么？

> 原始抓取（2026-11-06，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/bS_fZx7sMtiE2B4Yd8NNVA

只在虚拟房间练习，人形机器人走进陌生房间能做什么？
具身智能小站 具身智能小站
 Nov 6, 2026, 7:00 AM

阅读路线

今天更新 20 篇具身智能新论文，覆盖机器人运动与控制、操作策略与数据、交互式世界模型、导航与闭环评测等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进17 篇扫读

先读 cuRoboV2-Fast，拆开看机械臂规划耗时与真机抓放周期。接着看 HALO 怎样把虚拟房间里的训练带到真机，再看 Open-HumanScale 如何检验人类操作视频对陌生场景的帮助。

先看这一篇

1一次抓放省下 3 秒，机械臂快在哪里

cuRoboV2-Fast: Fast Torque-Limited Collision-Free Motion Planning

 

一句话判断：机械臂端着重物搬运，关节受力会随姿态改变。有些路段可以加速，有些路段再快一点就会超出关节力矩上限。cuRoboV2-Fast 按路段调整速度，作者再把真机的规划和执行分别计时，看一次抓放究竟省在哪一步。

你可以带走的两点

• 先测规划器。 在 3 公斤负载的 Franka Panda 离线基准上，同用 RTX 5090，平均规划耗时由 cuRoboV2 的 65.13 毫秒降至 4.43 毫秒。

• 再测真机节拍。 YAM 上，新旧方法分别统计约 50 次和 15 次已完成的搬运，平均抓放周期分别为 5.93 秒和 9.07 秒。把新版的 53 次完成记录拆开看，手臂运动占 63.1%，规划只占 1.0%。

方法亮点：抓起重物后，手臂在不同姿态下可用的加速度不同。作者先用 B 样条优化带载避障路径，再用 TimeOpt 在同一条路上逐段调速，让手臂在力矩余量大的路段加速、接近上限时放慢。真机数据显示，新方法让机械臂在接近速度上限的区间运行得更久，但要进一步缩短作业周期，可靠抓取仍是难点。

论文：https://arxiv.org/pdf/2611.06913

代码：https://github.com/NVlabs/curobo

项目页：https://research.nvidia.com/labs/lpr/curobov2-fast/

另外 2 篇值得跟进

2只在虚拟房间练习，人形机器人也能侧身过门

Perceptive Humanoid Motion from Simulation

 

看点：HALO 在一千个程序生成的住宅里合成动作数据。到了陌生房间，G1 根据相机画面和口头指令规划全身动作，再由底层控制器跟踪，完成过门、钻桌底和推箱等任务。

结果：六种单项技能在六个真实环境各测 10 次，360 次试验平均成功率为 91%。串联技能只在一个真实环境测了 30 次，三种序列的平均成功率为 70%。

论文：https://arxiv.org/pdf/2611.06908

代码：https://github.com/apple-aiml-research/ml-halo

项目页：https://apple-aiml-research.github.io/ml-halo/

3人手操作视频能帮机器人适应新桌面吗

Open-HumanScale: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models

 

看点：Open-HumanScale 从第一人称视频里提取手部动作。作者固定机器人预训练数据和每项任务的示范数，检验加入人类视频后，策略能否适应新的灯光与桌面。

结果：固定 14,618 小时机器人数据，再加入 45,938 小时人类视频。每项用 50 条干净示范微调后，RoboTwin 随机场景成功率从 70.80% 升至 74.36%（50 项仿真任务，各测 100 次）。

论文：https://arxiv.org/pdf/2611.06817

代码：https://github.com/DAGroup-PKU/HumanNet/tree/open-humanscale

项目页：https://dagroup-pku.github.io/open-humanscale/

17 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04给同一个仿真机器人换一双眼睛

Interactive World Simulation with Cosmos for Physical AI

保持物理仿真和策略一致，只换相机画面。在校准用的 32 个起点上，Cosmos 将成功次数从 22 提到 27。把盘子换成红色后，原生渲染反而占优。

论文：https://arxiv.org/pdf/2611.06627

代码：https://github.com/NVIDIA/cosmos-framework

项目页：https://research.nvidia.com/labs/cosmos-lab/cosmos-simulation/

05杯子该放回哪里，由机器人先前的动作决定

Robocasa-Mem: Benchmarking Environment and Process Memory in Long-Horizon Mobile Manipulation

后半程的正确目标由机器人先前的选择决定。九项厨房仿真任务中，记录物体和任务进度的分层策略平均成功率为 24%，作为基线的 π₀.₅ 为 16%。

论文：https://arxiv.org/pdf/2611.06475

资源：https://github.com/SiHhangChen/RoboCasa-Mem

项目页：https://sihhangchen.github.io/RoboCasa-Mem/

06遇到窄道，分开测机器人敢不敢走、能不能过

Beyond Infraction Scores: Failure-Boundary Profiles for Closed-Loop Evaluation of Driving and Robot Navigation Agents

逐步收窄通道，分别记录策略尝试通过和无碰撞通过的概率。在 CARLA 的堵路场景里，擦碰着挤过去有时比停车得分更高。两条曲线能看清总分掩盖的差别。

论文：https://arxiv.org/pdf/2611.06453

资源：https://github.com/speedym/failure-boundary-profiles/tree/v1.0-paper

07让机器人把试错经验写成下次能用的程序

PyRUA-REAL: Robot Exploration and Adaptive Learning through RSI in the Real World

两个智能体分工设计实验和改代码，把真机失误留作下次修改的依据。八项真机任务的 91.1% 是开发期间跨版本的平均成功率，冻结程序后的表现另行测试。

论文：https://arxiv.org/pdf/2611.06768

代码：https://github.com/DAGroup-PKU/PyRUA-Lean/tree/REAL

项目页：https://dagroup-pku.github.io/PyRUA-Lean/real/

08冻结扩散策略，从初始噪声调整动作

Efficient Diffusion Policy Fine Tuning with Latent Noise Representation Bridging

同一动作可能由不同噪声生成，直接给噪声打分容易学偏。LARBRID 让噪声和动作共享价值表示，只调整初始噪声分布，在仿真运动与操作任务上检验适应效果。

论文：https://arxiv.org/pdf/2611.07275

代码：https://github.com/Maincode-Research/LARBRID

09让机械臂提前给自己留一条安全退路

Beyond Terminal Constraints: Reliable Robot Control with Learned Safe Sets

规划器在预测轨迹的中间节点检查能否回到安全集合，求解遇阻时可切换到备用动作。作者也在六轴 Z1 上测试了参考轨迹跟踪，安全结论依赖论文对集合的假设。

论文：https://arxiv.org/pdf/2611.06196

代码：https://github.com/idra-lab/safe-mpc

10给手腕相机补上深度，双手更容易对准

Spatial-UMI-Dream: Scaling World Action Models with Metric Geometry for UMI Manipulation

夹爪靠近目标时，画面变化既来自物体，也来自相机自身。模型用双目深度学距离，同时预测图像、深度和动作。六项双臂真机任务各测 20 次，平均成功率为 85%。

论文：https://arxiv.org/pdf/2611.07229

项目页：https://weiduoyuan.com/spatial-umi-dream/

11夹爪接触物体后，哪种触觉最有用

Which Physical Tactile Cues Matter for Contact-Rich Imitation Learning? A Controlled Real-Robot Study

同一批示范、同一策略骨干下，三项真机任务各测 30 次。仅用视觉成功 52/90 次，加入触觉标记点位移后为 67/90 次，在单独加入的触觉线索中表现最好。

论文：https://arxiv.org/pdf/2611.06673

项目页：https://xuhaoqin02.github.io/lerobot-tactile/

12练过更多物体，策略能把经验用到哪里

Reinforcement Learning Enables Scaling Generalist Robot Policy Improvement

作者在仿真中把训练范围从一项扩大到 144 项，分别测试新场景与新物体。对本已有一定成功率的策略，增加训练物体种类并用价值估计选动作，更容易把收益带到新任务。

论文：https://arxiv.org/pdf/2611.06905

代码：https://github.com/lasgroup/generalist-improvement

项目页：https://lasgroup.github.io/generalist-improvement/

13记忆任务里，什么时候值得再问一次大模型

Can Astra Solve RoboMME without Breaking the Bank?

小模型检查子任务进度，必要时才请 Astra 看历史、派发下一步。RoboMME 的 800 次仿真试验里，每局平均调用 3.63 次，成功率为 79.13%。

论文：https://arxiv.org/pdf/2611.06184

代码：https://github.com/bingaochen/Astra-on-RoboMME

项目页：https://bingaochen.github.io/Astra-on-RoboMME/

14机器人找调料瓶，也能读瓶身上的字

STARE: Scene Text-Aware multimodal Retrieval for Everyday objects

STARE 把指令里的品牌或品名与场景文字、物体外观一起检索，再让机器人去抓。12 种布置共测 100 次，目标进入前五个检索结果且抓取成功的比例为 80%。

论文：https://arxiv.org/pdf/2611.06089

代码：https://github.com/keio-smilab24/STARE

项目页：https://stare-projectpage.pages.dev/

15用 3.49 美元跑完 200 次仿真搬盒评测

Closed-Loop Physics-Based Manipulation Evaluation at a Commodity Price: What 4.4 L4 GPU hours at $3.50 Buys You

在仿真搬盒任务上，8 张 L4 并行跑完 200 次评测，耗时约 38 分钟，按论文单价花费 3.49 美元。作者还剔除了 5 个开局就判成功的场景。

论文：https://arxiv.org/pdf/2611.07441

资源：https://github.com/rohungry/evaluation_harness_wrapper

16路被小车堵住，机器人该不该推

MessyNav: Zero-Shot Navigation via Manipulation in Messy Environments

空推车可以挪，放着笔记本的推车却该绕开。MessyNav 先判断障碍物能否安全移动，再选动作。15 个真机场景中成功 10 次，只用规划器的基线成功 6 次。

论文：https://arxiv.org/pdf/2611.07417

项目页：https://messynav.github.io/

17让机器人反复插线，换个摆法还能成功吗

VABAR: A Variational Automation Benchmark for Agentic Robotics

VABAR 把插线、齿轮啮合等六类操作拆成不同难度的仿真实例。除了成功率，它还记录成功试次的中位周期，便于区分机器人是卡在对位，还是能完成但耗时太长。

论文：https://arxiv.org/pdf/2611.06881

项目页：https://berkeleyautomation.github.io/vabar-website/

18人形机器人会踢球，球速有多准

Playing Soccer Like Humans Do: Learning Versatile and Agile Humanoid Soccer via Latent Motion Priors

用真人足球动捕训练动作先验，让 G1 用同一套策略跑动、带球、停球和射门。迁到真机后，踢球方向仍较准，但球速相对误差从仿真的 8.12% 升至 47.38%。

论文：https://arxiv.org/pdf/2611.06564

项目页：https://soccerbot-humanoid.github.io/

19张手绕开障碍，握拳让机器人群合力推物

Fluidic Body: Swarm Robots as a Fluid Extension of the Hand

张手时，桌面小车群散开绕障。握拳时，小车聚拢推物。17 人参加了虚拟现实对照实验，真机展示另收集了 88 份反馈。附链是实验用的 maru 小车平台。

论文：https://arxiv.org/pdf/2611.06068

项目页：https://github.com/omron-sinicx/maru

20生成机器人视频时，先判断布角会怎样动

ReactWorld: Structuring Interaction Responses for Embodied Video Generation

模型先推断谁在接触物体、哪些局部会动，再用预测位移引导后续画面。在视频生成评测 RBench 上，平均分比 PhysisForcing 高 3.29 分。

论文：https://arxiv.org/pdf/2611.06120

项目页：https://metaengine-world.github.io/ReactWorld/

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment