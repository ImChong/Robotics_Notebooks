# 微信公众号原始抓取：机器人出错之后，怎样把纠错变成下一次进步？｜附代码与项目入口

> 原始抓取落盘（正文纯文本，未下载图片）。来源见 [sources/blogs/wechat_embodied_station_robot_36_papers_2026-10-17.md](../blogs/wechat_embodied_station_robot_36_papers_2026-10-17.md)。

- **原始链接：** https://mp.weixin.qq.com/s/I4XP5JSG9QNmm4hd1nk2JQ
- **抓取方式：** `scripts/fetch_wechat_article.py`（Playwright + 预装 Chromium + MicroMessenger UA，2026-10-17）

---

机器人出错之后，怎样把纠错变成下一次进步？｜附代码与项目入口
Original 具身智能小站 具身智能小站
 Oct 17, 2026, 6:00 AM

阅读路线

今天更新 36 篇具身智能新论文，覆盖接触操作与灵巧手、世界动作模型、人形迁移与机器人学习、导航感知与仿真评测等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进33 篇扫读

先读 PACER，了解部署轨迹与人工纠错如何进入监督后训练；再跟进 CoSMO 的语义安全导航和 CTP 的单示范接触迁移，其余按研究方向快速浏览。

先看这一篇

1让拆解策略从纠错过程里学习

PACER: Process-Aware VLA Policy Learning from Human Corrections for Robotic Disassembly

 

一句话判断：部署轨迹里既有成功，也有接近成功和人工接管。PACER 用过程证据重加权训练动作片段，在保留原生监督损失的同时利用这些差异。它更适合已有 VLA 与纠错采集流程的团队，评测仍限定于主板部件的接近与交接。

你可以带走的两点

• 先判断可模仿性，再按过程加权：把人工纠错、干净示范与自主轨迹分开处理，失败动作保留审计用途，不直接当作模仿目标。

• 认清评测终点与比较成本：UR5e 五类部件各测 20 次，接近与交接成功为 87/100，普通后训练为 54/100、固定几何加权为 79/100；PACER 还需训练并验证多个权重候选。

方法亮点：将过程证据、轨迹角色与可模仿性门控统一到数据侧重加权；通过独立验证选择权重配置，无需新增奖励或价值模型。

论文：https://arxiv.org/pdf/2610.19248

代码：https://github.com/ChangChrisLiu/PACER

项目页：https://pacer-corl.github.io/

另外 2 篇值得跟进

2把语义不确定性转成导航安全距离

CoSMO: Conformal Semantic Mapping for Open-World Safe Robot Navigation

 

看点：CoSMO 将视觉语言特征校准为标签覆盖集合，再按类别扩张障碍区域；面对新增危险类别与延迟反馈，持续更新安全字典和阈值。

结果：在 2,195 个 MP3D 仿真回合中，相比仅几何导航基线，语义碰撞相对减少 61.6%；Go2 两处室内场景的 30 次试验达到 90.73% 长期真实标签覆盖率，覆盖率不等于无碰撞率。

论文：https://arxiv.org/pdf/2610.18321

代码：https://github.com/Yangming911/CoSMO

项目页：https://yangming911.github.io/cosmo-page/

3一段示范同时指定动作与接触方式

Contact Trajectory Prompting: In-Context Transfer of Contact-Rich Behaviors from a Single Demonstration

 

看点：CTP 把单次示范中的运动和接触变化编码为提示，执行时结合在线反馈检索相关片段；在训练外物体上验证擦拭、旋转与插入迁移。

结果：xArm7 上三类任务、九个训练外实例，每实例 12 次测试，平均成功率 78.7%，BPP* 为 65.7%；每次仍需测试物体的一段示范，策略参数保持冻结，插入实例 C 仍只有 33.3%。

论文：https://arxiv.org/pdf/2610.18848

代码：https://github.com/ctp-robotics/CTP

项目页：https://ctp-robotics.github.io/project_page/

33 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04把人类用力经验迁移给灵巧操作

FACT: A Multimodal Force-Action-Track Foundation Model for Human-to-Robot Dexterity

FACT 结合人类触觉数据与机器人示范，先联合预测动作、视频和轨迹，再用未来力预测细化动作，检验人机混合数据的迁移价值。

论文：https://arxiv.org/pdf/2610.19155

项目页：https://fact-1-0.github.io/

05从人类第一视角学习全身移动操作

λ₀: Scaling Open-World Egocentric Human Data for Humanoid Loco-Manipulation

λ₀ 将第一视角视频、身体与手部运动对齐到人形机器人，通过离线修正降低执行偏差，研究包含行走与姿态变化的人类数据迁移。

论文：https://arxiv.org/pdf/2610.18950

代码：https://github.com/LambdaHumanoid/openlambda

项目页：https://lambdahumanoid.github.io/

06把接触与力的演化写进动作表示

Interaction Latents as Actions for Contact-Rich Manipulation

ILA 将运动、接触和施力共同编码成可解码的交互潜变量，用于生成动作与离线改进，在多类策略和真机任务中检验收益。

论文：https://arxiv.org/pdf/2610.18855

项目页：https://mrsecant.github.io/ILA/

07从人类视频生成可回放的双手示范

EgoMatrix: Physically Faithful Dexterous Robot Demonstrations from Egocentric Human Videos

EgoMatrix 将单目第一视角视频重建为手物交互，再经重定向与强化学习生成双手轨迹和同步机器人观测，兼顾接触一致性与视觉数据。

论文：https://arxiv.org/pdf/2610.18822

项目页：https://spikew726.github.io/EgoMatrix/

08执行前先预测动作的接触后果

ForeTac: Predictive Contact Guidance for Generative Robot Policies

ForeTac 预测候选动作造成的触觉变化，再用接触质量目标引导生成策略；在擦拭、刷卡和易碎物抓取等真机任务中检验接触可靠性。

论文：https://arxiv.org/pdf/2610.18118

项目页：https://foretac.github.io/

09没有历史数据，也能回放旧任务经验

Data-Free Continual Learning for World Action Model with Rectified Generative Replay

R²-WAM 让世界动作模型生成旧任务轨迹，再截断完成后的无效片段并过滤动作观测不一致样本，研究持续学习中的遗忘与逆动力学退化。

论文：https://arxiv.org/pdf/2610.18038

项目页：https://kaipengfang.github.io/r2-wam/

10部署后的成功与失败都能提供学习信号

LoopWAM: Closed-Loop Generalist Policy Improvement from Deployment Rollouts

LoopWAM 对各质量轨迹保留视频监督，按质量调节动作模仿；将部署数据纳入下一轮训练，研究通用策略如何持续改进。

论文：https://arxiv.org/pdf/2610.18873

代码：https://github.com/loop-wam/LoopWAM

项目页：https://loop-wam.github.io/

11让任务示范与自由探索各学其长

ForceWAM: Decoupling Future and Response Learning from Demo and Play Data for Contact-Rich Manipulation

ForceWAM 用成功示范预测视频与力矩未来，再用任务及自由探索数据训练逆动力学，执行时结合新力反馈，将预测转成机器人动作。

论文：https://arxiv.org/pdf/2610.18852

项目页：https://force-wam.github.io/

12少量人类视频引导新场景操作

Steertuning Policies from a Few Human Videos

Steertuning 先用第一视角交互预训练，再结合少量新任务示范微调与采样引导，在不同家庭及实验室场景检验人机形态迁移。

论文：https://arxiv.org/pdf/2610.18711

项目页：https://steertuning.github.io/

13不新增执行轨迹，也能比较机器人策略

RoboFQE: Rollout-Free Evaluation of Visual Robot Policies from Offline Trajectories

RoboFQE 结合拟合 Q 评估、预训练视觉编码器与终止成功标注，从离线轨迹估计策略回报；评估策略排序，避免自回归视频误差累积。

论文：https://arxiv.org/pdf/2610.18699

项目页：https://yfdeng.com/robofqe/

14让任务长度成为可控的操作评测变量

LADDER: A Long-Horizon Robot Manipulation Benchmark

LADDER 在同一环境逐步增加子目标，比较底层策略与上层编排；通过仿真和真机观察长任务失败，区分规划能力与实际执行瓶颈。

论文：https://arxiv.org/pdf/2610.18053

项目页：https://ladder.insait.ai/

15让人形控制器适应新的身体

TRACE: Efficient Transfer Learning for Neural Whole-Body Control Across Humanoid Embodiments

TRACE 在输入输出映射加入运动学先验，结合动力学对齐、编码器蒸馏和低秩强化学习，将已有全身控制能力迁移到另一人形平台。

论文：https://arxiv.org/pdf/2610.17999

项目页：https://bfm-trace.github.io/

16只修复长任务中受偏差影响的动作

ReEnvision: Future-Guided Selective Action Repair for Long-Horizon Manipulation

ReEnvision 依据执行反馈更新未来与动作的联合信念，定位尚未执行片段中的受影响动作，保留有效部分，并在严重偏差时整体重规划。

论文：https://arxiv.org/pdf/2610.17977

项目页：https://reenvision.pages.dev/

17让世界模型更可靠地支撑策略强化学习

WoVR+: World Model as Reliable Simulator for Versatile VLA Policy Optimization with RL

WoVR+ 对齐多视角和本体状态接口，以动作块噪声与分阶段想象训练减少误差积累，检验世界模型用于长程 VLA 优化的可靠性。

论文：https://arxiv.org/pdf/2610.17922

项目页：https://youmo445.github.io/WoVR_Plus.github.io/

18为助行机器人补齐无障碍感知数据

NavAble: A Large-Scale Dataset and Synthetic Data Generation Pipeline for Blind Navigation

NavAble 聚焦按钮、扶手等无障碍关键物体，结合真实图像与合成场景补齐类别和视角覆盖，并在助行机器人上验证感知效果。

论文：https://arxiv.org/pdf/2610.19330

项目页：https://huggingface.co/datasets/NavAble/NeurIPS_2026_BLV

19把机器人试错沉淀成可复用代码技能

EmbodiedEvo: Self-Evolving Coding Agent For Physical Intelligence

EmbodiedEvo 把物理交互反馈提炼为代码技能和诊断经验，冻结后交给新场景中的智能体使用，检验跨场景复用与推理成本。

论文：https://arxiv.org/pdf/2610.18867

代码：https://github.com/Einsia/EmbodiedEvo

项目页：https://lab.einsia.ai/embodiedevo

20让机械狗记住探索与重访所需的信息

GLST: Global-Local Spatio-Temporal Memory Decomposition Framework for Online Chained-Goal Navigation

GLST 把空间与时间记忆分解为全局局部、长期短期表示，应对目标在线变化带来的探索和重访冲突，并以 Spot 机械狗验证。

论文：https://arxiv.org/pdf/2610.17927

代码：https://github.com/Yousef-Liang/glst

21让规划与进度判断支撑长序列操作

Let the Mind Conceive, Let the Hand Achieve: Following Multimodal Instructions for Long-Horizon Robotic Manipulation

Orchestra 将指令拆解、空间定位和进度评估与底层动作生成分工，维护子任务共享上下文，检验短轨迹训练后的长序列组合执行。

论文：https://arxiv.org/pdf/2610.18352

资源：https://github.com/youpengwen/Orchestra

项目页：https://youpengwen.github.io/Orchestra

22检验机器人智能体是否真的保留了经验

SEAR: Self-Evolving Agentic Robotics

SEAR 为代码控制、在线决策和自动训练设置对照，分离技能记忆、重复尝试与额外预算的作用，并跨任务、仿真器和真实机器人评估。

论文：https://arxiv.org/pdf/2610.18792

资源：https://github.com/Self-Evolving-Agentic-Robotics/SEAR

项目页：https://sear.bot/

23用扩展行为树组织建筑机器人任务

APTree: Extended Behavior Trees for Adaptive and Scalable Automated Robotic Task Planning in Construction

APTree 以领域语言描述任务，协调多类符号规划器生成模块化行为树，支持局部重规划；安全分析依赖规划器正确性和任务非干扰等假设。

论文：https://arxiv.org/pdf/2610.18523

代码：https://github.com/shshermin/APTree-automated-planning-tree

24让通用灵巧手策略优先练难点

DexMimic: A Generalist Dexterous Hand-Object Tracking Policy via Value-Guided Reset Allocation

DexMimic 将多条手物交互轨迹共同训练，通过价值引导分配重置时刻与起始状态，提高困难阶段的练习效率，并验证真实灵巧操作。

论文：https://arxiv.org/pdf/2610.19089

项目页：https://zhexiluo.github.io/DexMimic/

25让真实操作更快进入可调试仿真

RoboSimStudio: An Interactive Toolkit for Human and Agent Rapid Real2Sim of Robot Manipulation Tasks

RoboSimStudio 将求解器、接触模型和物理参数显式化，支持人类与代码智能体按观察迭代校准仿真，连接视觉判断、力反馈与真实执行。

论文：https://arxiv.org/pdf/2610.18835

项目页：https://robosimstudio.github.io/

26环境变化后，只重算受影响的房间

MIND: Instruction-Conditioned Incremental Grounding for Robots

MIND 将语言指令编译成语义依赖，依据环境事件更新受影响的房间判断，并把可达性与目标选择分开维护，减少重复语义评估。

论文：https://arxiv.org/pdf/2610.19262

代码：https://anonymous.4open.science/r/mind-icra27

项目页：https://mind-icra27.github.io/

27把灵巧手结构与姿态映射一起优化

PACT: Posture-Aligned Co-Design Technique for Hand Configuration and Retargeting in Multi-Finger Imitation Learning

PACT 共同优化机器人手的机械参数与非线性姿态重定向，并纳入装配限制；通过实体手、遥操作和模仿学习验证设计，而非只比较仿真姿态。

论文：https://arxiv.org/pdf/2610.18443

项目页：https://pact-hand-2027.github.io/

28把室内外往返作为连续导航任务

I2O2I-Nav: A Benchmark with a Modular Agent for Long-Horizon Indoor-to-Outdoor-to-Indoor Navigation

I2O2I-Nav 以真实终点衔接室内外导航阶段，结合完成验证、交接与有限恢复；完整任务仍明显难于单阶段，适合定位长程失败环节。

论文：https://arxiv.org/pdf/2610.18307

资源：https://github.com/i2o2i-nav/i2o2i_nav

项目页：https://i2o2i-nav.github.io/

29在视觉受限条件下用声呐建图

RTISSLAM: In-air Imaging Sonar SLAM using a Robust Pose Graph and Occupancy Grid Mapping

RTISSLAM 结合成像声呐、鲁棒位姿图与占据栅格，利用回环校正地图和轮式里程计；当前论文重点结果来自带镜面反射的仿真环境。

论文：https://arxiv.org/pdf/2610.18479

代码：https://github.com/Cosys-Lab/RTISSLAM

30为机器人超声扫描构建可控环境

SonoWorld: Towards Interactive Pose-Controllable Ultrasound World Models

SonoWorld 从临床视频学习超声动态，再用少量带位姿扫描建立六自由度控制；提供交互仿真思路，尚不能等同临床自主扫描能力。

论文：https://arxiv.org/pdf/2610.19327

项目页：https://iclr2027anonymous.github.io/SonoWorld/

31把确定运动与视觉不确定性分开预测

Decoupled Drift-Diffusion World Models for Consistent Future Frame Prediction

D3WM 同时预测下一状态与空间不确定性，再让扩散模型补充遮挡显露等不确定区域，在桌面操作和相机运动场景检验预测质量。

论文：https://arxiv.org/pdf/2610.19212

项目页：https://anton-arapin.github.io/d3wm-site/

32用想象目标补齐策略的组合泛化

Imagining Goals with Geometric Horizon Models for Policy Generalization

TD-Aug 从几何时域模型采样可达未来，把想象结果当作目标增强行为克隆；在导航和操作基准检验技能拼接，重点是离线组合泛化。

论文：https://arxiv.org/pdf/2610.19129

项目页：https://td-aug.github.io/

33区分机器人能力提升与学习机制改进

Who Improves the Robot? Toward Physical Recursive Self-Improvement

Physical RSI 提出分层框架，区分任务内适应、跨任务能力增益与改进过程的递归提升；文献分析尚未确认达到其最高层级的证据。

论文：https://arxiv.org/pdf/2610.18842

资源：https://github.com/SAIL-Research-Lab/Physical-RSI

34在运行约束下协调空地机器人

Collaborative Planning for UAV-UGV Teams under Operational Constraints via Masked Error-bounded Neuro-Symbolic Diffusion

MEND 将无人机与地面车协作规划表示为掩码生成，通过逻辑约束重加权动作分布，并按预测误差安排生成顺序，兼顾协同与执行约束。

论文：https://arxiv.org/pdf/2610.18850

项目页：https://hcrlab.gitlab.io/project/mend/

35让多架飞行器协同搬运并追踪目标

CALM: Decentralized Control of Cooperative Aerial Manipulation with Multi-Agent Reinforcement Learning

CALM 将共享强化学习策略与扰动观测、载荷估计及编队前馈结合，研究多机搬运篮筐并跟踪地面机器人的去中心化控制。

论文：https://arxiv.org/pdf/2610.18425

项目页：https://petreric.github.io/CALM/

36用稀疏观测补全跨形态物体运动

Cepheid: Decoupling Scale from Structure for Cross-Family Motion Completion

Cepheid 将运动残差的结构与尺度分开建模，检验跨几何族的运动补全；机器人抓取柔性物体属于应用验证，重点仍是运动重建。

论文：https://arxiv.org/pdf/2610.18871

项目页：https://su-terry.github.io/cepheid-website/

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment