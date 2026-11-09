让机器人看懂空间，部署时还要多跑一个模型吗？
具身智能小站 具身智能小站
 Nov 9, 2026, 6:00 AM

阅读路线

今天更新 29 篇具身智能新论文，覆盖空间表征与VLA泛化、智能体自我改进与评测、接触操作与触觉学习、人形协作与仿真工具等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进26 篇扫读

先深读SLA的空间训练机制与实验口径，再跟进Flex-RSI的流程自我改进评测和美团MR0的桌面纠错系统；其余论文按各自方向快速浏览，保存需要细读的官方入口。

先看这一篇

1SLA：把空间理解前移到预训练

SLA: Scaling Spatial-Language-Action Model via Continual Pre-training

 

一句话判断：换背景、挪物体之后，机器人往往仍看得见目标，却难以准确操作。SLA把三维空间监督前移到持续预训练，再延续到下游适配，部署时移除空间教师和对齐分支。它适合关注泛化与部署开销的团队，阅读时要分清场景迁移和跨机器人适配的训练条件。

你可以带走的两点

• 把空间学习放在训练侧：空间教师与多层门控贯穿持续预训练和后训练。RoboTwin 2.0用50项任务、每项50条干净示范适配，每项随机场景测试100次，作者报告成功率72.3%，π0.5为46.0%。

• 按对象读真机结果：8种条件、每种50个单独计分对象实例，SLA平均83.8%，π0.5为59.8%。跨机器人条件使用100条Piper加10条Franka示范，属于少样本适配。

方法亮点：SLA将空间对齐从任务后训练扩展到持续预训练，以自适应门控融合多层视觉特征，并利用部分可用的深度和相机标定补充监督。多基准与真机测试显示场景变化下的泛化改善，消融支持两个训练阶段共同对齐；部署时移除空间辅助分支，跨机器人结果则仍依赖少量目标机器人示范。

论文：https://arxiv.org/pdf/2611.08373

代码：https://github.com/OpenHelix-Team/Spatial-Forcing/tree/master/SLA

项目页：https://sla-team.github.io/omnisf_projectpage/

另外 2 篇值得跟进

2Flex-RSI：让智能体改进自己的工作流程

Flex-RSI: Benchmarking Recursive Self-Improvement of Frontier Agents

 

看点：智能体能否在固定预算内改进自己的提示、工具和执行流程？Flex-RSI把方案优化与流程改写分开记录，并用隐藏测试集验收，适合研究自动实验和自我改进的团队。

结果：在24项任务、18个模型中比较固定预算内的改进，采用开发集优化、保留验证集筛选和隐藏测试集计分。模型自身权重保持冻结，改写的是提示、工具和控制流程；收益随任务类型与反馈粒度变化。

论文：https://arxiv.org/pdf/2611.08946

资源：https://github.com/flex-rsi/rsi-bench

项目页：https://flex-rsi.github.io/

3美团MR0：把纠错接进桌面操作

Meituan-Robotics-0: A Vision-Language-Action Foundation Model for Desktop Manipulation

 

看点：桌面操作中的失败恢复，需要比成功示范更直接的反馈。MR0用Rollback纠错配对连接被替换的动作与人工修正，并结合分阶段训练、异步动作合并和事件记忆。

结果：真机每策略每任务测试30次；使用Rollback数据时，CFG-DAgger的网线插接成功率为70%，RECAP为60%。纠错收益来自特定后训练设置，跨任务记忆和中断后恢复主要作定性展示。

论文：https://arxiv.org/pdf/2611.08577

代码：https://github.com/Meituan-Robotics/Meituan-Robotics-0

项目页：https://meituan-robotics.github.io/Meituan-Robotics-0/

26 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04Lodos：加速精细网格接触仿真

Lodos: A Lightweight Simulator Designed for Contact-Rich Interactions with Detailed Meshes

为复杂网格碰撞重做碰撞检测与接触求解，在保持可用物理表现的同时提高并行吞吐。实验包含手内操作与控制迁移，速度优势针对所测精细几何场景，不代表所有仿真负载。

论文：https://arxiv.org/pdf/2611.08872

代码：https://github.com/martius-lab/lodos

05TactileLab：用触觉流学习接触变化

TactileLab: Sim-to-Real Dexterous Robot Manipulation with Efficient Shear-Sensitive Tactile Simulation

以刚体仿真生成触觉流与接触深度，再把真实触觉映射到同一表示，支持双臂跟随和手内旋转。优势是可扩展训练；当前迁移依赖粘着接触假设，突发滑移仍可能导致失效。

论文：https://arxiv.org/pdf/2611.07881

项目页：https://yijionglin.github.io/tactilelab/

06RSIJudge：高分来自能力，还是评分规则

Learning the Task or Learning the Verifier? Rethinking Criteria Exposure in Embodied Self-Improvement

通过控制智能体能否看到成功判据，检验自我改进中的评测泄漏。读取判据能推高原始得分，但优势在忠于指令的新判据和跨任务迁移中缩小；结论来自受控仿真实验。

论文：https://arxiv.org/pdf/2611.08318

代码：https://github.com/DAGroup-PKU/PyRUA-Lean/tree/RSIJudge

项目页：https://dagroup-pku.github.io/PyRUA-Lean/RSIJudge/

07RoboARCade：检验会推理的机器人能否做准

RoboARCade: Benchmarking Agentic Robot Control

在MuJoCo环境里测试智能体的长程控制与自我改进，并区分策略选择、操作精度和误报完成。文中排行榜按子任务得分汇总，不能当成整项任务的完成率；目前主要是仿真。

论文：https://arxiv.org/pdf/2611.08209

资源：https://github.com/Satvik1701/RoboARCade

项目页：https://roboarcade.org/

08VFlash：把视觉轨迹一次成组解码

VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding

利用坐标与轨迹结构，成组预测视觉基元，减少逐坐标自回归开销。论文覆盖轨迹预测与下游操作；约25倍的生成加速来自单次rollout初测，不等于机器人整体快25倍。

论文：https://arxiv.org/pdf/2611.08449

代码：https://github.com/NVlabs/Eagle/tree/main/Embodied

项目页：https://research.nvidia.com/labs/lpr/vflash/

09PAVE：给下一台机器人留下更好的现场

Leave It Better Than Found: Value-Guided Cooperation in Sequential Tasks

先行动的机器人读取后来者的价值函数，规划更合适的交接状态，无需预先联合训练。仿真与真机验证了协作收益，但目前只有两台机器人按固定顺序行动，并共享环境表征。

论文：https://arxiv.org/pdf/2611.08075

代码：https://github.com/proroklab/PAVE

项目页：https://proroklab.github.io/PAVE/

10RoboImagine：同时校正未来画面与任务进展

Learning Robotic World Models with Semantic and Visual Feedback

比较多条候选未来，用语义偏好和视觉误差共同后训练机器人世界模型。评估覆盖真实数据与仿真数据上的视频预测；更好的画质和语义评分，尚未证明能提升闭环规划或真机控制。

论文：https://arxiv.org/pdf/2611.07850

代码：https://github.com/Alexander-wu/RoboImagine

11训练曲线正常，也可能传错了数据

Detecting Silent Faults at the Environment-Trainer Boundary in Reinforcement Learning

在强化学习的环境与训练器之间注入错误，发现回报曲线常常仍然正常。研究建议两侧保存逐步记录并对账，另由环境端记录回合；这检测的是执行一致性，不能替代策略性能评估。

论文：https://arxiv.org/pdf/2611.08144

资源：https://github.com/PlugRL

12ForPC：让人形机器人稳稳推车

ForPC: Force-Aware Whole-Body Control for Precise Humanoid Cart Pushing

通过力感知隐变量与残差全身控制协调双臂、腰和腿，减少双手相互对抗。策略从仿真迁移到G1推车，使用机载观测估计受力；速度误差的主要对比来自仿真测试。

论文：https://arxiv.org/pdf/2611.08664

项目页：https://roboforpc.github.io/

13S2C：先找到目标，再靠力反馈做精细接触

Semantics-to-Contact: A Stagewise Framework for Robust Contact-Rich Manipulation

先用视觉语言模型定位任务相关区域，再以人工参与的残差强化学习完成接触精修。六类真机插接任务检验视觉与位置扰动，训练还需示范、纠错和力反馈，并非零样本操作。

论文：https://arxiv.org/pdf/2611.07532

项目页：https://edem-ai.github.io/S2C/

14统一FIC框架：兼顾跟踪与接触柔顺

A Unified Fractal Impedance Control Framework

把分形阻抗控制整理为统一框架，以H-PMP协调阻抗与导纳行为。Kinova真机比较轨迹跟踪、接触突然消失和拖动示教；结果仅支持所测场景，不能当作普遍安全保证。

论文：https://arxiv.org/pdf/2611.08009

代码：https://github.com/TRL-Sussex/FIC_Interface

15H-TRIP：地图扩展后，批量距离也及时更新

H-TRIP: Exact Many-to-Many Distance Index for Multi-Robot Exploration

用分层增量最短距离索引更新机器人与候选目标之间的距离矩阵，避免新通路出现后沿用过时缓存。仿真测试验证精确性与查询加速；当前限于二维地图中可通行区域单调增加。

论文：https://arxiv.org/pdf/2611.08139

代码：https://github.com/ozaslan/htrip

16Q-OIL：把人工接管变成探索线索

Q-OIL: Optimistic Exploration in Learning from Human Interventions

观察到人常在机器人停滞后才接管，且纠正动作并非最优，Q-OIL用乐观奖励和解耦评价器引导探索。评估覆盖仿真、真实用户及双臂VLA微调，用户研究规模仍有限。

论文：https://arxiv.org/pdf/2611.08576

项目页：https://weirdlabuw.github.io/qoil

17OopsieRecover：让少量纠错跨场景复用

OopsieRecover: Learning Transferable VLA Corrections from Semantic Interventions

从少量语义标注干预中推断纠错意图，将其写成条件触发的空间代价，引导冻结VLA选择动作。实验检验位置与场景迁移；效果依赖目标定位、跟踪以及候选动作是否覆盖修正。

论文：https://arxiv.org/pdf/2611.08462

项目页：https://robin-lab.cs.utexas.edu/oopsierecover/

18GPWM：用三维粒子预测物理变化

GPWM: Modeling the World as 3D Particle Motion

用统一粒子表征描述刚体、布料和颗粒运动，并在生成噪声空间搜索接触计划。实验假设完整且可跟踪的三维状态，规划在仿真中开环执行，尚未打通从相机感知到在线控制的流程。

论文：https://arxiv.org/pdf/2611.08338

项目页：https://robertren1122.github.io/gpwm/

19说“不要做”，VLA听得懂吗

Localizing and Editing the Goal Representation of a Vision-Language-Action Model

对OpenVLA的仿真审计发现，否定指令难以改变执行目标。研究用激活干预与目标探针定位可编辑位置，并提示把目标合规监测与普通失败检测分开；结论限于单模型单场景。

论文：https://arxiv.org/pdf/2611.08895

代码：https://github.com/sudsho/vla-goal-representation

20Social Humanoid：让全身动作回应人

Social Humanoid: Socially Responsive Human-Humanoid Interaction

将人类与人形机器人的动作映射到共享离散表征，由高层生成互动反应、低层跟踪执行。G1已有真机展示，但尚未建模接触力、舒适度及安全约束，互动体验也待用户研究。

论文：https://arxiv.org/pdf/2611.08407

项目页：https://von31.github.io/social-humanoid/

21MOMASPLAT：为移动操作补齐新视角示范

MOMASPLAT: Dynamic Novel-View Demonstration Augmentation for Mobile Manipulation

分别重建静态场景与被操作物体，用动态新视角合成补齐接触后的示范，并同步变换动作。五项真机任务检验底座位置变化下的泛化；遮挡、可变形物体与未建模区域仍是限制。

论文：https://arxiv.org/pdf/2611.08372

项目页：https://robin-lab.cs.utexas.edu/momasplat/

22HOC2R：从人类视频提取接触轨迹

HOC2R: Hand-Object Contact Trajectories for Human-to-Robot Motion Transfer

结合冻结视觉特征与轻量预测头，提取手物接触轨迹供机器人使用。真机评估仅六条序列，较强配置成功三条；已知接触帧的双点定位与自动识别接触是不同能力。

论文：https://arxiv.org/pdf/2611.08920

代码：https://github.com/liszzzzt/HOC2R

23SCALE：带宽有限时，先传有用信息

SCALE: Real-Time Latent Communication for Cyber-Physical Agents

复用连续观测中相似的隐表示，再按语义收益、带宽和截止时间决定消息及token数量。测试覆盖协同感知、操作与实体蜘蛛机器人；场景突变会削弱信息复用带来的收益。

论文：https://arxiv.org/pdf/2611.08390

代码：https://anonymous.4open.science/r/SCALE-4E80/

24HITCH：给视觉语言模型一套全身操作接口

HITCH: Teaching a Vision-Language Model to Operate a Humanoid through a Whole-Body Harness

让模型从图像点取三维位置，再用身体部位指令驱动全身跟踪器，学习可迁移的人形操作技能。测试含基准与真机，但动作执行期间不重新观察，且尚不支持坐下、跨障碍等动作。

论文：https://arxiv.org/pdf/2611.08039

项目页：https://hitch.axell.top/

25TangramBench：先协调零件，再执行装配

It Takes Two to Tangram: Multi-Agent Object Centric Reinforcement Learning for Assembly

把每块拼图视作智能体，用图网络和共享策略协调平面运动，再由真机顺序执行。目标位置事先给定；策略推理快、接触少，但紧密装配的完成率仍落后于联合规划加重规划。

论文：https://arxiv.org/pdf/2611.08814

项目页：https://tangrambench.org/

26dynamicalnodes：从控制框图生成ROS2节点

dynamicalnodes: Block Diagrams to ROS2 Graphs

把控制器、被控对象和观测器写成状态更新与输出函数，先组合仿真，再生成ROS2节点与通信配置。论文用巡航控制展示工作流，适合原型开发，尚不是复杂系统的大规模验证。

论文：https://arxiv.org/pdf/2611.07565

代码：https://github.com/nehalsinghmangat/dynamicalnodes/tree/main

项目页：https://nehalsinghmangat.github.io/dynamicalnodes/tutorials/0_cruise_control/cruise_control_with_dynamicalnodes.html

27形状教师：训练时学几何，部署时只看图像

Aligning Policy Encoders with a Shape Teacher

训练时用Point-MAE的三维形状特征对齐图像编码器，部署仍只接收图像。LIBERO测试显示收敛加快，但鲁棒性改善随扰动类型变化，尚缺跨架构与真机验证。

论文：https://arxiv.org/pdf/2611.08275

项目页：https://miepjemadelon.github.io/aligningpolicieswithashapeteacher/

28康复机器人：从目标肌肉激活反推轨迹

Transformer-Based Trajectory Prescription for sEMG-Optimizing Robot-Mediated Rehabilitation

用Transformer与流匹配，将目标肌电模式映射为接触力轨迹，并测试新用户适配。研究仅用健康参与者的等长交互数据验证可行性，尚无临床疗效证据。

论文：https://arxiv.org/pdf/2611.08295

资源：https://simtk.org/projects/openrobotrehab

29TurtleBot3：把仿真控制器带进课堂真机

Educational robotics beyond ROS: a custom library for direct transfer of Webots controllers in the Turtlebot3

用Python库复刻Webots设备接口，帮助控制逻辑迁移到TurtleBot3真机。证据来自小班教学与学生反馈，主要体现入门便利，不能视为教学成效的对照验证。

论文：https://arxiv.org/pdf/2611.08142

代码：https://github.com/jpperezm/TurtleBot3-NoROS-Library

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment