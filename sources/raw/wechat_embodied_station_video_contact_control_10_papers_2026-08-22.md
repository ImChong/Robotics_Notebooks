---
title: 具身智能又卷到哪了？10 篇开源论文把视频、接触和控制串起来
author: 具身智能小站
date: "2026-08-22 09:00:00"
source: "https://mp.weixin.qq.com/s/EmC4gNgcQdPX34vxy-qSVQ"
---

# 具身智能又卷到哪了？10 篇开源论文把视频、接触和控制串起来

📅 2026年8月22日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 10 篇具身智能开源论文，内容覆盖第一视角手部重建、移动操作、离线强化学习、人形运动、装配感知、灵巧抓取、**ROS 2** 基础栈、潜动作学习与 **VLA** 持续学习。整体来看，这批论文共同追问一个问题：如何把人类视频、仿真结构、接触几何和大模型表征，转化为能在真实机器人上稳定执行的动作能力。

**综述主线：**从「看懂动作」到「生成可执行动作」，再到「跨技能持续适配」，具身智能正在把数据来源、控制接口和策略学习一起重做。

**速览地图**

人类视频到动作数据仿真与控制走向实机接触几何支撑操作泛化**VLA** 与潜动作持续适配

01 · arXiv:2608.20308v1

🔬 ****DreamHand**: Repurposing Video Diffusion Models for Occlusion-Robust Egocentric 3D Hand Motion Recovery**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaJTHRibR4m543icXmD2J1bodTkeRKXyrUq761wZIa6b705VzKyQIWm7hddDVXOBzur0JIPToicuLyGC2dSXBpSYSCibtUxofEAkv0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

📌 **Embodied AI · Egocentric Vision · 3D Hand Recovery · Video Diffusion**

✨ 将视频扩散模型改造成确定性几何编码器，补全遮挡手部轨迹

📖 具身智能想从第一视角视频扩展操作数据，但 3D 手部轨迹因物体遮挡和手部出画而难恢复。**DreamHand** 不把 **VDM** 当像素生成器，而将其重用为确定性几何编码器：单次 clean latent 前向暴露当前观测外的场景内容，再由双向时空解码器恢复连续双手轨迹；Ray-Based Camera Solver 还支持无测试时相机内参配置。五个 egocentric benchmark 上刷新 state of the art，ARCTIC/HOT3D 的 MPJPE-p 分别降低 30%/40%，纳入出画手后收益达 46%-61%。

💡 扩散先验的价值，可能不在生成，而在几何记忆。

🔗 项目链接： https://github.com/ggxxii/dreamhand

🔗 资料来源： https://arxiv.org/pdf/2608.20308v1

02 · arXiv:2608.20251v1

🔬 ****Video2DoorTraversal**: Push Door Traversal via Simulated Door Twins**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubjhHCibwGzqcfddjn0k8C2JfReLR8w78PcJWqN06ALNvYeQbegd7YemvVnrWFn102TgSbIkTKxFIia0vtibfdQxogpw9iaoP7EyRw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

📌 **Loco-Manipulation · **Real-to-Sim-to-Real** · Door Traversal · Mobile Manipulation**

✨ 从单段真实门视频构建仿真门孪生体，训练可上机穿门策略

📖 开门并穿越是长视野 loco-manipulation 任务，需要精准把手交互和底盘、机械臂协同。**Video2DoorTraversal** 提出单视频 **Real-to-Sim-to-Real** 框架：**DoorTwin** 从一段真实 RGB 视频重建实例对齐、带关节且可仿真的门孪生体；仿真闭环 agent 将关节信息转成参数化技能程序，并迭代修复失败 rollout 生成可执行演示。随后 **ArticuACT** 以双深度输入预测底盘、手臂和夹爪命令。系统在五扇真实门上平均成功率 96.57%，结构相近未见门 zero-shot 成功率 80.95%，全流程平均约 13 秒。

💡 门操作的关键，是把一次观察变成可反复试错的仿真资产。

🔗 项目链接： https://video2doortraversal.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.20251v1

03 · arXiv:2608.20208v1

🔬 ****RoMAN-Flow**: Taming Autoregressive Normalizing Flows for Offline Reinforcement Learning in Robotic Manipulation**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubLD19HXuxzB4RR6a9yLECwiaem1jYj00V3A2icU97DicBeOcAlHU1Mibfym6e5P8tMuawYY2Oibo3WRiaKDLo7BY66oXHxfARnSS8KU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

📌 **Offline RL · Normalizing Flow · Robotic Manipulation · Policy Distillation**

✨ 让自回归归一化流进入离线 RL 后训练，并压低部署延迟

📖 离线强化学习希望在不追加环境交互的前提下提升机器人策略，但扩散和 flow matching 策略缺少可处理的似然，限制了 likelihood-based 后训练。**AR-NF** 兼具动作建模能力和精确似然评估，却在策略优化与部署中受制于序列采样开销。**RoMAN-Flow** 用 sampling-free 的 advantage-weighted likelihood 目标提高高优势离线动作的似然，避免优化时从自回归策略采样；部署时再把优化后的自回归策略蒸馏为一步动作生成器。多个仿真操作基准和真实机器人平台上，它保持竞争性策略表现，同时显著降低推理延迟。

💡 可计算似然让离线强化学习有了更直接的抓手。

🔗 项目链接： https://github.com/konnyaku28/RoMAN-Flow

🔗 资料来源： https://arxiv.org/pdf/2608.20208v1

04 · arXiv:2608.20087v1

🔬 **Towards Professional Tennis Styles for Humanoid Robots with Adaptive Motion Planning and Tracking**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguamic3KkY6iaJcZStFuDcgRd0jgkWibV22VMPl0mlek4g1NM8LSSTt1Z4TxLOmbv9UJlDcmVpLCY3UD3jqGFp5bUUeRh71ibtcP7HE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

📌 **Humanoid Robot · Motion Planning · **Sim-to-Real** · Sports Robotics**

✨ 从比赛转播学习职业网球动作风格，并部署到真实人形机器人

📖 人形机器人已能在真实球类运动中展现潜力，但要同时保持职业动作风格和任务表现仍然困难。**AdaPT** 是 Adaptive motion Planning and Tracking 框架，直接从比赛转播视频学习网球发球与回合风格：planner 生成带风格的运动学动作，tracker 尽量少干扰地执行这些动作。面对真实机器人上的 tracking 退化、autoregressive planning 误差累积和噪声感知，方法通过随机执行速度训练提升跟踪鲁棒性，并用 motion-speed adapter 条件化 planner。Unitree G1 实验验证了 sim-to-real gap 缓解效果，策略还部署到约 1.7m 的 Dobot Atom，并展示无动捕的野外发球。

💡 人形运动风格要落地，规划与跟踪必须分工。

🔗 项目链接： https://humanoidtennis.github.io/AdaPT/ ｜ GitHub: https://github.com/noitom-robotics/AdaPT

🔗 资料来源： https://arxiv.org/pdf/2608.20087v1

05 · arXiv:2608.19968v1

🔬 ****PVRA**: A Pointwise Key-point Voting Framework for Robotic Assembly**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZ9WAj1TVynicqA4HJzhHYz2aJ8ibyJrJvkNWYCtrKulL5n7pKPRh49XoXCJMWF3x3MASgWaXNXwgsfZicBBhAcvUu7ltAy9JvQMY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

📌 **Robotic Assembly · RGB-D Perception · Keypoint Voting · 6-DoF Pose**

✨ 用 3D 关键点投票学习装配依赖，输出可执行装配线索

📖 现代计算机视觉已推动装配操作的部分自主化，但渐进式装配不仅要识别物体，还要理解部件之间的装配依赖。**PVRA** 通过相关领域比较分析指出，object-centric perception 需要走向学习 assembly dependencies，才能预测对自主装配有意义的 actionable outputs。论文提出一个 3D keypoint-based 模块化学习框架，从 RGB-D 装配场景中学习依赖关系并推断可执行输出；模型在 assembly pose estimation 数据集上训练与评估，并用面向渐进式装配的增强指标对比 object-centric baseline。

💡 装配感知的下一步，是从看见物体走向理解依赖。

🔗 项目链接： https://github.com/KulunuOS/PVRA

🔗 资料来源： https://arxiv.org/pdf/2608.19968v1

06 · arXiv:2608.19776v1

🔬 ****CoToGrasp**: Contact-Topology-Conditioned Dexterous Grasp Synthesis via Canonical Workspace Learning**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYjPSJOMxFWLLMvJ8fJm0FBdzbcJJt7aH9lfZ3j6WDq3DxAdLicgPktI4d5IiaHHYuribheeVpYrNiaHOrFWLrLB0WKTTkibL2EMakc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

📌 **Dexterous Grasping · **Contact Topology** · Generative Model · Zero-Shot**

✨ 以接触拓扑约束灵巧抓取，让功能意图脱离具体物体几何

📖 现有灵巧抓取规划器多优化物理稳定性，即关注物体能否被抓住，而较少回答如何抓才能服务下游功能任务；按人类抓取 taxonomy 条件化又往往需要昂贵的物体标注数据。**CoToGrasp** 提出接触拓扑条件化的生成框架，合成多样且稳定的灵巧抓取，并完全以 object-agnostic 方式训练。其 feature-based canonical workspace 将局部物体特征投影到统一的 gripper-centric 域中，把功能意图与任意物体几何解耦；通过学习夹爪内在接触流形，模型可在推理时 zero-shot 泛化到未见物体。DexGraspNet 大规模评估显示其达到 state of the art，并在真实机器人上验证物理可行性和运动学可行性。

💡 接触拓扑把能抓推进到为任务而抓。

🔗 项目链接： https://cea-list.github.io/cotograspweb/

🔗 资料来源： https://arxiv.org/pdf/2608.19776v1

07 · arXiv:2608.19759v1

🔬 ****GOAG**: Generative and **Object-Agnostic** Grasp Planner for Dexterous Robotic Manipulation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubKibQB3YxUa6dctNU7ntHVLKFngI8ySiakKIlxTQvxISfBj0rubN6peet4Wwhp03GS0Jaj6OHqLnSe4g36uibLsVbArSicH3dqiaqM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

📌 **Dexterous Manipulation · **Object-Agnostic** Grasping · Generative Model · Contact Geometry**

✨ 只学习夹爪接触流形，在推理时再接入物体特征做泛化抓取

📖 多指抓取是机器人关键技能，但很多深度学习抓取规划器依赖有限且面向特定物体的数据，泛化到新物体时容易失效。**GOAG** 基于一个几何观察：夹爪和物体在相互接触点共享相同表面几何。它学习特定夹爪接触表面分布的紧凑潜表示，从而在不依赖 object-specific training data 的情况下高效采样有效抓取；物体特征只在推理时引入，用于检索与夹爪能力兼容的可接触区域。仿真和真实实验表明其可适配不同夹爪，在 MultiDex 物体上平均成功率 86.93%，生成大量抓取时更快，同时匹配专门在该数据集训练的领先方法表现。

💡 不绑定物体数据，抓取模型才有机会真正泛化。

🔗 项目链接： https://cea-list.github.io/goagweb/

🔗 资料来源： https://arxiv.org/pdf/2608.19759v1

08 · arXiv:2608.19740v1

🔬 **Keeping the **Franka Emika Panda** alive: a **ROS 2** stack with a reliable position interface**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubsq6OyYULlibGTrkSSb8FUjkd3WIldqdqAlsgmWdiayzLkpUdklrqlUsdDXsU1tGL9NEPKpO3E1byMcdSHlAbMiaSpBAkYxJGgSo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

📌 ****ROS 2** · Franka Panda · Position Control · Real-Time Robotics**

✨ 重建 Panda 的 **ROS 2** 位置控制栈，解决抖动和保护停机痛点

📖 这篇工作提供开源软件栈，恢复 **Franka Emika Panda** 的 **ROS 2** 支持，并解决长期存在的外部位置控制接口不可靠问题。论文首先分析不稳定位置控制的根因，指出振动和 protective stops 来自外部控制回路时序与采样抖动，而非机器人本体限制。基于此，系统引入异步硬件接口以解耦实时通信和 **ROS 2** control loop，加入面向较慢指令源的 rate-matching 机制，并采用位置域参考生成策略产生平滑可靠的位置命令。实验显示该架构能更可靠跟踪速度参考、减少官方实现带来的运动伪影，并在两个 Panda 平台上覆盖运动规划、柔顺控制、位置控制操作和触觉遥操作。

💡 机器人基础栈的可靠性，本身就是研究加速器。

🔗 项目链接： https://sites.google.com/view/fer-ros2/

🔗 资料来源： https://arxiv.org/pdf/2608.19740v1

09 · arXiv:2608.19613v1

🔬 **What Matters for **Latent Actions** in Robot Learning**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguav1YTXQKlrUjicySZibAvvdwGmRNlJAiapyAFjtffYSTY96rffgzdgR9lkEtNY1CHYLZ7HVgMiaSKGD1Cb4diaGR831qv81HgEg51c/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

📌 **Robot Learning · Latent Action · **VLM** · Empirical Study**

✨ 系统评测 41 项 **LAM** 设计，回答潜动作到底哪些因素有用

📖 **Latent Action Models** 试图用紧凑 **latent actions** 把大规模无标注视频转化为机器人学习资源，但现有研究在不一致实验设置下分别评估设计选择，难以判断哪些因素真正影响下游操作表现。该论文给出面向 robotic manipulation 的首个综合实证研究，将代表性 **LAM** 方法统一到 autoencoding 框架下，系统考察 41 项设计选择，覆盖 **latent action** 建模范式、学习目标与正则化、以及 **latent action** 集成策略。同时，论文评估四类 **latent action** 质量代理指标与下游性能的相关性。三个常用基准和真实机器人任务的结果表明，用 **latent actions** 微调 **VLM** backbone 能为下游策略学习提供更强初始化。

💡 潜动作研究需要从模型炫技回到可比实验。

🔗 项目链接： https://carldegio.github.io/latent\_action.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.19613v1

10 · arXiv:2608.19589v1

🔬 ****OrthoSkillVLA**: Continual Skill Learning via Gradient-Informed Skill Subspace Adaptation**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZBgh33KqHoOLvoW5moIdwaZmrsm0VER3Pibqu5hOBsMZ7wD6bHS3KX2vgwfdcSY2mGZvBMGeHX0ChkpOoLybJ45tDVicF2Yyf4s/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

📌 ****VLA** · Continual Learning · Skill Adaptation · **MoE****

✨ 用组件级正交子空间和轻量 **MoE**，缓解 **VLA** 连续学技能遗忘

📖 预训练 Vision-Language-Action 模型为机器人学习提供强基础，但顺序适配多种技能会扰动旧技能依赖的表征和速度映射，导致 catastrophic forgetting。已有架构隔离方法会增加推理负担，子空间约束方法又常对整个模型施加统一约束。**OrthoSkillVLA** 先分析 **VLA** 内部组件角色差异：**VLM** 承载广泛语义、易受容量耗尽影响；ActionHead 将语义细化为局部速度模式，对扰动更敏感；最终速度解码器若冻结会形成表达瓶颈，若更新又可能覆盖旧映射。因此论文分别对 **VLM** 和 ActionHead 施加子空间约束，并为输出层引入轻量 feature-aware **MoE** decoder，每个技能分配紧凑 expert，训练免费 router 按特征空间亲和度选择 expert。仿真和真实评估显示该方法能在学习新技能时更好保留旧技能。

💡 **VLA** 持续学习的难点，在于不同模块遗忘方式不同。

🔗 项目链接： https://github.com/Jiaqi-Wangx/OrthoSkillVLA

🔗 资料来源： https://arxiv.org/pdf/2608.19589v1

**综合观察**

这 10 篇论文的共同信号很清楚：机器人学习不再只是在更大的模型上堆数据，而是在补齐动作链路中的关键结构。**DreamHand** 和 **latent action** 研究把人类视频变成可学习动作表征；**Video2DoorTraversal**、**AdaPT** 和 **ROS 2** Panda 栈把仿真、跟踪和控制接口推向真实部署；**CoToGrasp**、**GOAG** 与 **PVRA** 则提醒我们，接触拓扑、装配依赖和物体无关建模仍是操作泛化的关键难题。对圈内人来说，值得关注的不是某个单点指标，而是这些机制是否能被复用到更长视野、更复杂接触、更开放场景的机器人系统里。
