# 【10.5-10.9前沿论文动态】人形/四足44篇

- **作者：** senlanke（具身运控lab）
- **发表日期：** 2026-10-12
- **原文：** <https://mp.weixin.qq.com/s/HPqfC2jkqoL4zPn9c9pGEw>
- **归档依据：** 用 `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）读取公众号正文后提炼条目；此页只保留条目线索与详情节点映射，论文技术事实以各详情节点引用的 arXiv 原文为准。
- **条目数：** 44 个论文条目；同一 arXiv 论文只保留一个 wiki 详情节点（与另一篇周报重复的条目共用同一节点）。

## 条目与详情节点

### 人形机器人（27篇）

#### Dataset-Free Compliant Humanoid Loco-Manipulation with Dynamic Online Posture（arXiv:2610.05678）

- **论文：** [arXiv:2610.05678](https://arxiv.org/abs/2610.05678)
- **机构（周报所述，未逐条核实）：** 加州大学圣迭戈分校、延世大学
- **问题：** 仅给定双手末端目标时，机器人既缺少全身参考动作，也难以兼顾可达性、平衡和接触柔顺性。
- **方法线索：** 依据解析可达性先验在线生成骨盆高度和躯干姿态，再通过策略在环采样修正；
- **详情节点：** [Dataset-Free Compliant Humanoid Loco-Manipulation with Dynamic Online Posture（OCLO）](../../wiki/entities/paper-oclo-online-posture-compliant-loco-manipulation.md)

#### ★ BRACE: Adapting Whole-Body References for Force and Terrain Aware Humanoid Motion Tracking（arXiv:2610.07052）

- **论文：** [arXiv:2610.07052](https://arxiv.org/abs/2610.07052)
- **机构（周报所述，未逐条核实）：** Multiply Labs、布朗大学、新墨西哥大学
- **问题：** 平地动作参考不能直接适用于斜坡、台阶以及双手持续施力场景。
- **方法线索：** 地形模块同步调整落脚点和根节点，力控制模块根据目标手部力计算参考位移，使同一跟踪策略同时适应地形和外部作用力。
- **详情节点：** [BRACE：让全身跟踪同时适应地形与交互力](../../wiki/entities/paper-brace-force-terrain-whole-body-tracking.md)

#### ★ Hierarchical Reinforcement Learning for Collision-Free Locomotion of an Underactuated Biped（arXiv:2610.05855）

- **论文：** [arXiv:2610.05855](https://arxiv.org/abs/2610.05855)
- **机构（周报所述，未逐条核实）：** 印度理工学院曼迪分校
- **问题：** 欠驱动双足机器人需要在保持动态平衡的同时，根据局部障碍观测完成目标导航。
- **方法线索：** 高层SAC根据射线观测输出平面速度和偏航指令，低层策略负责速度跟踪与关节控制，并通过联合训练协调导航和稳定行走。
- **详情节点：** [欠驱动双足分层 RL 无碰撞运动](../../wiki/entities/paper-hrl-underactuated-biped-collision-free.md)

#### ★ I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning（arXiv:2610.06129）

- **论文：** [arXiv:2610.06129](https://arxiv.org/abs/2610.06129)
- **机构（周报所述，未逐条核实）：** 同济大学、清华大学、浙江大学、RoboParty Lab、哈尔滨工业大学、上海科技大学
- **问题：** 现有通用人形行为模型主要生成无接触动作，物体交互仍依赖任务专用参考轨迹或单独训练。
- **方法线索：** 用前向—后向表征和无监督强化学习建模机器人、物体及接触的耦合动力学；
- **详情节点：** [I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning](../../wiki/entities/paper-i-bfm.md)

#### Talk, Render, Act: Integrating Social Gesture and Digital Face with Synchronized Speech for Conversational Humanoid Robot（arXiv:2610.06153）

- **论文：** [arXiv:2610.06153](https://arxiv.org/abs/2610.06153)
- **机构（周报所述，未逐条核实）：** 阿联酋大学
- **问题：** 对话机器人语音、面部动画和身体手势通常独立生成，容易出现语义与时间不同步。
- **方法线索：** 构建机器人可执行的语义动作原子库，由组合规划器选择手势序列，并与语音和数字面部动画统一同步。
- **详情节点：** [TRABot：说、演、动](../../wiki/entities/paper-trabot-talk-render-act.md)

#### R2RI: A Multi-View Event and RGB Dataset for Robot-to-Robot Interaction（arXiv:2610.07117）

- **论文：** [arXiv:2610.07117](https://arxiv.org/abs/2610.07117)
- **机构（周报所述，未逐条核实）：** 佛罗伦萨大学、摩德纳和雷焦艾米利亚大学、锡耶纳大学
- **问题：** 机器人间交互研究缺少同时包含第一视角、第三视角及事件数据的大规模数据集。
- **方法线索：** 同步提供多台人形机器人的自我视角和外部视角RGB、事件流及交互标注，覆盖超过650万帧。
- **详情节点：** [R2RI：机器人间交互的多视角事件与 RGB 数据集](../../wiki/entities/paper-r2ri-robot-interaction-dataset.md)

#### ★ InterMimicGen: Scaling Humanoid Loco-Manipulation through Self-Evolving Motion Imitation（arXiv:2610.06850）

- **论文：** [arXiv:2610.06850](https://arxiv.org/abs/2610.06850)
- **机构（周报所述，未逐条核实）：** 伊利诺伊大学厄巴纳—香槟分校、NVIDIA
- **问题：** 人形移动操作数据分散在不同人体—物体数据源中，动作质量和机器人可执行性差异很大。
- **方法线索：** 统一整理并重定向人体—物体动作，训练通用跟踪器；
- **详情节点：** [InterMimicGen：用自演进动作模仿扩展人形移动操作](../../wiki/entities/paper-intermimicgen.md)

#### ★ What the Elevation Map Cannot See: Semantic-Aware Locomotion and Execution-Aware Navigation for Humanoid Robot（arXiv:2610.07396）

- **论文：** [arXiv:2610.07396](https://arxiv.org/abs/2610.07396)
- **机构（周报所述，未逐条核实）：** 佛罗里达大学
- **问题：** 高程图无法表示湿滑、脆弱、危险区域等语义风险，导航器也通常忽略实际运动控制器对指令的执行误差。
- **方法线索：** 将视觉语言语义危险识别、地形运动和执行反馈闭环结合，使路径规划同时考虑几何可通行性、语义风险和控制器实际能力。
- **详情节点：** [高程图看不到什么：语义感知运动与执行感知导航](../../wiki/entities/paper-semantic-aware-locomotion-execution-aware-nav.md)

#### BiGym 2.0: Benchmarking Learned and Agent-Developed Policies for Humanoid Household Manipulation（arXiv:2610.07594）

- **论文：** [arXiv:2610.07594](https://arxiv.org/abs/2610.07594)
- **机构（周报所述，未逐条核实）：** 帝国理工学院
- **问题：** 缺少覆盖人形全身家庭操作、原生人类示范以及多类学习方法的统一基准。
- **方法线索：** 在Unitree G1上设置20项家庭任务，每项提供60条VR示范，并统一评测模仿学习、VLA、示范强化学习和代码生成智能体。
- **详情节点：** [BiGym 2.0：人形家庭操作基准](../../wiki/entities/paper-bigym2-humanoid-household-benchmark.md)

#### ★ Beyond Retargeting: Low-Latency and Robust Humanoid Whole-Body Teleoperation with Learned Atomic Motion Primitives（arXiv:2610.07891）

- **论文：** [arXiv:2610.07891](https://arxiv.org/abs/2610.07891)
- **机构（周报所述，未逐条核实）：** 香港科技大学、浙江大学、腾讯Robotics X、湖南大学
- **问题：** 直接人体动作重定向容易产生机器人不可执行姿态，且复杂低层跟踪模型会增加遥操作延迟。
- **方法线索：** 先以PPO训练特权混合专家Teacher，再将行为压缩为可部署的原子动作基元，实现低延迟控制、动作恢复和跨操作者泛化。
- **详情节点：** [Beyond Retargeting：学习式原子动作基元的低延迟人形全身遥操作](../../wiki/entities/paper-atomic-motion-primitives-teleoperation.md)

#### AutodidactWAM: Cross-Modal Self-Distillation from Generated Video to Robot Actions（arXiv:2610.08119）

- **论文：** [arXiv:2610.08119](https://arxiv.org/abs/2610.08119)
- **机构（周报所述，未逐条核实）：** 斯科尔科沃科学技术研究院、MWS研发中心
- **问题：** 生成视频能够扩充任务和场景，但缺少与视频对应的可执行机器人动作标签。
- **方法线索：** 从生成视频中提取伪动作监督，再通过跨模态自蒸馏和流模型适配将视觉变化转化为人形机器人动作。
- **详情节点：** [AutodidactWAM：从生成视频到机器人动作的跨模态自蒸馏](../../wiki/entities/paper-autodidactwam.md)

#### iGPC: Generative Motion Priors for Object-Aware Humanoid Interaction（arXiv:2610.08120）

- **论文：** [arXiv:2610.08120](https://arxiv.org/abs/2610.08120)
- **机构（周报所述，未逐条核实）：** 穆罕默德·本·扎耶德人工智能大学、麻省理工学院、Minerva Humanoids
- **问题：** 传统人体动作先验缺少物体和接触条件，无法可靠支持人形机器人在真实场景中的交互。
- **方法线索：** 学习以物体状态和交互目标为条件的生成式动作先验，再由接收深度、激光雷达和本体感知的Student策略转化为实机控制。
- **详情节点：** [iGPC：面向物体感知人形交互的生成式动作先验](../../wiki/entities/paper-igpc-object-aware-humanoid-interaction.md)

#### ★ Humanoid Horizon: Extending Task Horizon in Whole-Body Loco-Manipulation via Parallel Training, Dynamic Starting, and Reward Gating（arXiv:2610.08320）

- **论文：** [arXiv:2610.08320](https://arxiv.org/abs/2610.08320)
- **机构（周报所述，未逐条核实）：** 曼彻斯特大学、中国科学技术大学、X-Humanoid、华威大学、纽卡斯尔大学
- **问题：** 多物体连续搬运中，强化学习容易偏向前期简单阶段，并在训练后期任务时遗忘先前行为。
- **方法线索：** 并行训练各任务阶段，以前序终止状态动态初始化后序任务，并用奖励门控保护已放置物体；
- **详情节点：** [Humanoid Horizon：用并行训练、动态起始与奖励门控扩展全身移动操作时域](../../wiki/entities/paper-humanoid-horizon-long-horizon-loco-manipulation.md)

#### From Legs to Wheels: Embodiment-Aware Human Motion Retargeting for Mobile-Base Humanoids（arXiv:2610.08381）

- **论文：** [arXiv:2610.08381](https://arxiv.org/abs/2610.08381)
- **机构（周报所述，未逐条核实）：** 嘉泉大学
- **问题：** 人体下肢动作无法直接映射到没有腿部关节的轮式人形机器人。
- **方法线索：** 根据机器人形态把人体步行和骨盆运动重新分配给轮式底盘、躯干及双臂，同时保持手部任务空间轨迹和全身协调性。
- **详情节点：** [From Legs to Wheels：面向轮式人形的形态感知人体动作重定向](../../wiki/entities/paper-legs-to-wheels-mobile-base-humanoid-retargeting.md)

#### ★ HULK: Learning Whole-Body Forceful Loco-Manipulation for Humanoids（arXiv:2610.08970）

- **论文：** [arXiv:2610.08970](https://arxiv.org/abs/2610.08970)
- **机构（周报所述，未逐条核实）：** Amazon、密歇根大学、加州大学洛杉矶分校、加州理工学院
- **问题：** 搬运大型重物会改变质心并持续向上肢施加载荷，使平衡与动作跟踪相互冲突。
- **方法线索：** 用负载动力学MPC指导强化学习，分别训练手腕受力和抱持重物行走Teacher，再蒸馏为统一策略；
- **详情节点：** [HULK：重载人形机器人的全身移动操作](../../wiki/entities/paper-hulk-forceful-loco-manipulation.md)

#### MimicX: Policy-in-the-Loop Supervision Refinement for Video-Driven Humanoid Motion Tracking（arXiv:2610.09055）

- **论文：** [arXiv:2610.09055](https://arxiv.org/abs/2610.09055)
- **机构（周报所述，未逐条核实）：** 香港科技大学（广州）
- **问题：** 从视频重建的动作视觉上合理，但可能包含持续不可执行的转场、碰撞或局部姿态。
- **方法线索：** 根据策略执行失败定位困难时段和身体区域，联合调整跟踪目标与重置课程，再通过反复实体验证保留真正改善可执行性的修改。
- **详情节点：** [MimicX（策略在环的视频驱动人形动作跟踪监督精炼）](../../wiki/entities/paper-mimicx.md)

#### Workhorse: Learning Robust Whole-Body Humanoid Loco-Manipulation from Human Data（arXiv:2610.09117）

- **论文：** [arXiv:2610.09117](https://arxiv.org/abs/2610.09117)
- **机构（周报所述，未逐条核实）：** 加州大学伯克利分校
- **问题：** 人类示范不能直接解决人形机器人平衡、接触误差和任务失败后的自主恢复。
- **方法线索：** 从人体动作学习任务意图和全身协调，再以强化学习获得满足机器人动力学约束的执行与恢复能力，实现搬箱、分拣和提箱攀爬等任务。
- **详情节点：** [Workhorse：从人类数据学习稳健全身移动操作](../../wiki/entities/workhorse-humanoid-loco-manipulation.md)

#### Precise SE(3) End-Effector Tracking in Whole-Body Humanoid Control（arXiv:2610.09479）

- **论文：** [arXiv:2610.09479](https://arxiv.org/abs/2610.09479)
- **机构（周报所述，未逐条核实）：** 加州大学伯克利分校、Amazon FAR、卡内基梅隆大学
- **问题：** 全身运动策略通常难以同时实现精确末端位置、姿态跟踪和接触柔顺性。
- **方法线索：** 将几何导纳控制与残差强化学习结合，由解析控制器提供SE(3)结构和柔顺响应，学习残差补偿全身动力学误差。
- **详情节点：** [ResGAC：全身人形控制中的精确 SE(3) 末端执行器跟踪](../../wiki/entities/paper-resgac-precise-se3-end-effector-tracking.md)

#### Benchmarking Behavioral Steerability in Behavior Foundation Models（arXiv:2610.10198）

- **论文：** [arXiv:2610.10198](https://arxiv.org/abs/2610.10198)
- **机构（周报所述，未逐条核实）：** 浙江大学、宇树科技、新加坡国立大学、中南大学及浙江大学—宇树具身智能联合实验室
- **问题：** 现有评测只关注人形行为能否生成，无法判断模型是否准确执行速度、方向、顺序和身体部位等复合要求。
- **方法线索：** 提出RoboSteer基准，以条件操控、约束操控和组合操控三个层级评测9种行为基础模型，包含大规模多模态动作任务。
- **详情节点：** [RoboSteer：行为基础模型可操控性基准](../../wiki/entities/paper-robosteer-behavior-steerability-benchmark.md)

#### ★ HuMBLE: Human Motion-Driven Behavior Learning for Embodied Locomotion（arXiv:2610.10489）

- **论文：** [arXiv:2610.10489](https://arxiv.org/abs/2610.10489)
- **机构（周报所述，未逐条核实）：** RAI Institute、Boston Dynamics、卡内基梅隆大学
- **问题：** 命令跟踪策略步态机械，而严格模仿人体动作的策略又难以响应数据分布外的速度与方向指令。
- **方法线索：** 先用RL训练参考动作Teacher，再蒸馏为仅依赖本体感知和躯干平面速度的轻量策略；
- **详情节点：** [HuMBLE：人类动作驱动的具身 locomotion](../../wiki/entities/paper-humble-human-motion-driven-behavior.md)

#### TAPNAV: Humanoid Navigation through Tactile Active Perception（arXiv:2610.10748）

- **论文：** [arXiv:2610.10748](https://arxiv.org/abs/2610.10748)
- **机构（周报所述，未逐条核实）：** 佐治亚理工学院、哈维穆德学院、Walton High School
- **问题：** 在黑暗、烟雾或摄像头失效时，人形机器人无法仅靠视觉确认障碍与通道结构。
- **方法线索：** 通过里程计、IMU和接触构建位姿置信分布，并在不确定区域主动执行触探动作；
- **详情节点：** [TAPNAV：通过触觉主动感知实现人形机器人导航](../../wiki/entities/paper-tapnav-humanoid-tactile-navigation.md)

#### Being-M0.7: A Latent World-Action Model for Humanoid Robots（arXiv:2610.11283）

- **论文：** [arXiv:2610.11283](https://arxiv.org/abs/2610.11283)
- **机构（周报所述，未逐条核实）：** 智在无界（BeingBeyond）
- **问题：** 人形移动操作数据昂贵，而大量人类数据往往只有视频或动作之一，不能直接生成机器人控制命令。
- **方法线索：** 利用超过一万小时的视频、动作及配对数据联合预测未来视觉latent和人体运动；
- **详情节点：** [Being-M0.7: A Latent World-Action Model for Humanoid Robots（人形机器人潜空间世界–动作模型）](../../wiki/entities/paper-being-m07-humanoid-latent-wam.md)

#### ★ DAMP: Humanoid Locomotion via Denoised Belief Learning and Adversarial Motion Priors（arXiv:2610.11505）

- **论文：** [arXiv:2610.11505](https://arxiv.org/abs/2610.11505)
- **机构（周报所述，未逐条核实）：** 山东大学、山东省人形机器人重点实验室
- **问题：** 没有外部地形感知时，人形机器人难以在复杂地形上同时保持稳定、目标一致和自然动作。
- **方法线索：** 循环网络从本体历史中隐式推断特权状态和任务latent，并以对抗动作先验约束运动风格；
- **详情节点：** [DAMP：去噪信念学习 + 对抗动作先验的人形运动控制](../../wiki/entities/paper-damp-humanoid-denoised-belief-amp.md)

#### Humanoid World Action Model With Joint State–Action Generation（arXiv:2610.12026）

- **论文：** [arXiv:2610.12026](https://arxiv.org/abs/2610.12026)
- **机构（周报所述，未逐条核实）：** 逐际动力（LimX Dynamics）、南方科技大学
- **问题：** 高层策略输出的参考动作经过平衡控制、动力学和接触后会产生执行偏差，普通世界动作模型没有显式建模这一差距。
- **方法线索：** 同时生成参考动作和执行后的本体状态，并用策略、正向动力学和逆向动力学三条条件路径连接动作、实际身体响应与未来视觉结果。
- **详情节点：** [HWAM：联合状态–动作生成的人形世界动作模型](../../wiki/entities/paper-hwam-humanoid-world-action-model.md)

#### VioLA: Learning Generalist Humanoid Control Policies from Human Data（arXiv:2610.12435）

- **论文：** [arXiv:2610.12435](https://arxiv.org/abs/2610.12435)
- **机构（周报所述，未逐条核实）：** Vesoma、苏黎世联邦理工学院、马克斯·普朗克智能系统研究所、图宾根大学、ELLIS Institute
- **问题：** 人形关节动作维度高且强耦合，机器人示范数量又远少于人类视频和动作数据。
- **方法线索：** 策略预测身体与手部动作latent而非机器人关节命令，通过共享编码空间对齐人类和机器人动作；
- **详情节点：** [VioLA：从人类数据学习人形通用控制](../../wiki/entities/paper-viola-human-data-control.md)

#### DASH: A da Vinci Adapter for Serial-link and Humanoid Robots as an Accessible Platform for Surgical Robotics Research（arXiv:2610.05792）

- **论文：** [arXiv:2610.05792](https://arxiv.org/abs/2610.05792)
- **机构（周报所述，未逐条核实）：** 加州大学圣迭戈分校
- **问题：** 达芬奇手术器械高度依赖专用平台，难以用于通用机械臂和人形机器人研究。
- **方法线索：** 设计兼容Classic和Xi器械的适配器，保留原生锁止机构，通过无线识别自动加载器械特定运动学及耦合矩阵，无需改装器械。
- **详情节点：** [DASH：面向串联机械臂与人形机器人的达芬奇器械适配器](../../wiki/entities/paper-dash-da-vinci-adapter.md)

#### PhoneBot: A Low-Cost Open Humanoid Robot Platform Reusing Smartphones（arXiv:2610.08737）

- **论文：** [arXiv:2610.08737](https://arxiv.org/abs/2610.08737)
- **机构（周报所述，未逐条核实）：** 加州大学洛杉矶分校RoMeLa实验室
- **问题：** 小型人形平台仍需要独立计算、视觉、通信和交互硬件，导致成本及复现门槛较高。
- **方法线索：** 将普通智能手机作为躯干计算与感知核心，结合13个执行器及开源机械、电气和软件设计，形成面向教学和快速原型开发的平台。
- **详情节点：** [PhoneBot：复用智能手机的低成本开源人形平台](../../wiki/entities/paper-phonebot-smartphone-humanoid.md)

### 四足机器人（13篇）

#### ★ Transporting Unsecured Stacked Payloads with a Quadrupedal Robot via Multi-Objective Reinforcement Learning（arXiv:2610.05819）

- **论文：** [arXiv:2610.05819](https://arxiv.org/abs/2610.05819)
- **机构（周报所述，未逐条核实）：** 富士通
- **问题：** 机器人自身稳定并不代表未固定箱体不会滑动或倾倒，速度跟踪与载荷稳定之间存在动态冲突。
- **方法线索：** 训练由目标权重条件化的多目标基础策略，再由本体感知权重调节器在线改变运动和载荷稳定优先级；
- **详情节点：** [PAMORT：用多目标强化学习运输未固定的堆叠载荷](../../wiki/entities/paper-pamort-stacked-payload-quadruped-morl.md)

#### Towards Quadruped-Provided Localization and Active Tracking for Micro-UAVs（arXiv:2610.06215）

- **论文：** [arXiv:2610.06215](https://arxiv.org/abs/2610.06215)
- **机构（周报所述，未逐条核实）：** 哥本哈根信息技术大学、Helix Lab、诺和诺德
- **问题：** 微型无人机无法携带完整定位传感器和计算平台。
- **方法线索：** 由带机械臂相机的四足机器人检测无人机AprilTag，并与自身定位组合成共享地图坐标；
- **详情节点：** [四足机器人为微型无人机提供定位与主动跟踪](../../wiki/entities/paper-quadruped-uav-localization-active-tracking.md)

#### Robust Nonprehensile Object Transport with Quadruped Robots（arXiv:2610.07245）

- **论文：** [arXiv:2610.07245](https://arxiv.org/abs/2610.07245)
- **机构（周报所述，未逐条核实）：** 慕尼黑工业大学、坦佩雷大学、那不勒斯费德里科二世大学、穆罕默德·本·扎耶德人工智能大学
- **问题：** 背部托盘上的无约束物体对惯量误差和机器人加速度高度敏感，容易滑落。
- **方法线索：** 通过不确定性感知轨迹优化寻找闭环敏感度较低的物体运动，再用同时预测机器人与载荷质心的耦合凸MPC和全身QP跟踪。
- **详情节点：** [四足机器人鲁棒非抓取式物体运输](../../wiki/entities/paper-quadruped-nonprehensile-object-transport.md)

#### Feeling Through the Load: Compliant Quadruped Locomotion under Payload Interactions（arXiv:2610.08637）

- **论文：** [arXiv:2610.08637](https://arxiv.org/abs/2610.08637)
- **机构（周报所述，未逐条核实）：** Fujitsu Research of America、卡内基梅隆大学
- **问题：** 人直接推动未固定载荷时，机器人需要区分载荷自然滑动与有意引导力，并安全改变运动。
- **方法线索：** 训练载荷感知柔顺运动策略和因果力估计器，通过估计器在环数据聚合与微调，仅用机器人自身传感器估计载荷作用力并产生导纳式响应。
- **详情节点：** [Feeling Through the Load：载荷交互下的四足柔顺运动](../../wiki/entities/paper-feeling-through-the-load-compliant-quadruped.md)

#### Magnet-Aware Control of Legged Robots（arXiv:2610.08653）

- **论文：** [arXiv:2610.08653](https://arxiv.org/abs/2610.08653)
- **机构（周报所述，未逐条核实）：** 慕尼黑工业大学、欧洲核子研究中心
- **问题：** 强磁场会破坏传感器并对机器人铁磁部件产生随位姿变化的力和力矩，普通扰动控制器难以补偿。
- **方法线索：** 构建MuJoCo磁力插件与逆向磁场估计器，并在MPC和全身控制中显式预测磁扰动，提高四足机器人在强磁场区域的稳定工作范围。
- **详情节点：** [腿式机器人的磁场感知控制](../../wiki/entities/paper-magnet-aware-legged-control.md)

#### ★ Teaching a Robot Dog New Tricks: Diverse Quadruped Skills via Combined Reinforcement and Imitation Learning with Adversarial Task Selection（arXiv:2610.10601）

- **论文：** [arXiv:2610.10601](https://arxiv.org/abs/2610.10601)
- **机构（周报所述，未逐条核实）：** 加州大学圣克鲁兹分校研究团队
- **问题：** 在一个策略中同时学习行走、挖掘、跳跃等差异显著的任务会产生样本效率低和梯度冲突。
- **方法线索：** 先训练8个任务Teacher，再以RL与模仿学习联合目标蒸馏Student；
- **详情节点：** [四足多技能：RL+模仿联合蒸馏与对抗任务选择](../../wiki/entities/paper-quadruped-skills-adversarial-task-selection.md)

#### Borrowed Eyes: Markerless Nano-UAV Flight with an Active Quadruped Observer（arXiv:2610.09967）

- **论文：** [arXiv:2610.09967](https://arxiv.org/abs/2610.09967)
- **机构（周报所述，未逐条核实）：** 哥本哈根信息技术大学、Helix Lab、诺和诺德
- **问题：** 无人机携带标记会增加重量和阻力，而快速运动与遮挡又容易使外部观察丢失目标。
- **方法线索：** 用四足机器人RGB-D相机完成无标记三维定位，并以非线性MPC联合调整机身和机械臂视角；
- **详情节点：** [Borrowed Eyes：四足主动观察者支持的无标记纳米无人机飞行](../../wiki/entities/paper-borrowed-eyes-quadruped-nano-uav.md)

#### ★ Energy-Efficient Gait Adaptation via Hierarchical Reinforcement Learning for Quadrupedal Locomotion Across Diverse Terrains（arXiv:2610.10297）

- **论文：** [arXiv:2610.10297](https://arxiv.org/abs/2610.10297)
- **机构（周报所述，未逐条核实）：** ITMO大学、BE2R实验室
- **问题：** 固定步态难以同时适应速度和地形变化，直接最小化能耗又可能损害运动稳定性。
- **方法线索：** 高频策略执行关节控制，低频策略根据地形和状态调整步态参数，并显式优化运输成本；
- **详情节点：** [分层强化学习节能步态适配（四足多地形）](../../wiki/entities/paper-hrl-energy-efficient-gait-adaptation-quadruped.md)

#### ★ CMP-IRRT*: A Perception-Assisted Height-Adaptive Planner for Quadruped Robots（arXiv:2610.10470）

- **论文：** [arXiv:2610.10470](https://arxiv.org/abs/2610.10470)
- **机构（周报所述，未逐条核实）：** 中山大学深圳校区
- **问题：** 二维规划器通常把所有障碍都视为不可穿越，无法利用四足机器人跨越低矮物体的能力。
- **方法线索：** 从俯视RGB估计地面相对高度，高障碍绕行、低障碍允许穿越；
- **详情节点：** [CMP-IRRT*：感知辅助的四足高度自适应规划器](../../wiki/entities/paper-cmp-irrt-star-height-adaptive-planner.md)

#### Higher-Order Morphology Priors for Quadruped Reinforcement Learning Under Actuator Degradation（arXiv:2610.10934）

- **论文：** [arXiv:2610.10934](https://arxiv.org/abs/2610.10934)
- **机构（周报所述，未逐条核实）：** 普林斯顿大学、Edison Academy Magnet School、South Brunswick High School
- **问题：** 某些关节驱动力下降后，其余关节必须形成跨肢体补偿，普通图策略只建模成对连接。
- **方法线索：** 把Unitree Go1表示为包含节点、边和面单元的胞腔复形，使用Hodge消息传递显式编码肢体及身体级高阶机械关系。
- **详情节点：** [高阶形态先验：Go1 执行器退化下的 Hodge 消息传递策略](../../wiki/entities/paper-hodge-morphology-priors-quadruped-degradation.md)

#### RAGNAROK: Radar-Aided Gravity-Normalized Alignment for Robust Open Keyframe-based Radar-Visual-Kinematic-Inertial SLAM（arXiv:2610.11531）

- **论文：** [arXiv:2610.11531](https://arxiv.org/abs/2610.11531)
- **机构（周报所述，未逐条核实）：** 首尔大学、慕尼黑工业大学、苏黎世联邦理工学院
- **问题：** 四足机器人在打滑、冲击、低光、强眩光和重复纹理环境中，单一视觉或本体里程计容易失效。
- **方法线索：** 融合打滑与滚动接触感知的腿速、运动学雷达因子、图像退化增强、B样条雷达本体骨干及在线外参标定，并动态调整各模态权重。
- **详情节点：** [RAGNAROK：四足的雷达–视觉–运动学–惯性 SLAM](../../wiki/entities/paper-ragnarok-radar-visual-kinematic-inertial-slam.md)

#### Walking on Roofs: Exploring the Potential of Walking Robots for Construction Work on Roofs（arXiv:2610.12272）

- **论文：** [arXiv:2610.12272](https://arxiv.org/abs/2610.12272)
- **机构（周报所述，未逐条核实）：** 卡尔斯鲁厄理工学院
- **问题：** 商业四足机器人及标准球形足端是否适合倾斜屋顶尚缺少系统验证。
- **方法线索：** 搭建可调坡度屋顶试验台，量化Unitree Go2的滑移、起身和趴下能力，发现滑移随坡度近似二次增长且25度已频繁失效。
- **详情节点：** [Walking on Roofs：商用四足能在屋顶上干活吗](../../wiki/entities/paper-walking-on-roofs-quadruped-go2.md)

#### ★ Toward Lunar Legged Robots: Field Deployment Lessons at LUNA（arXiv:2610.12276）

- **论文：** [arXiv:2610.12276](https://arxiv.org/abs/2610.12276)
- **机构（周报所述，未逐条核实）：** 苏黎世联邦理工学院机器人系统实验室
- **问题：** 月壤沉陷、打滑、扬尘和极端照明对四足机器人运动、感知和任务运行的影响缺少长时实地数据。
- **方法线索：** 在LUNA月面模拟环境部署ANYmal-D和Magnecko，系统记录月壤接触、视觉惯性导航及操作问题，明确提出月壤感知运动、抗照明感知和任务级验证需求。
- **详情节点：** [Toward Lunar Legged Robots：LUNA 月面模拟场的部署教训](../../wiki/entities/paper-toward-lunar-legged-robots-luna-field.md)

### 轮足与通用腿式机器人（4篇）

#### ★ RACER: Residual-Adaptive Closed-Loop Estimation for Sampling-Based Planning in Wheeled-Quadruped Racing（arXiv:2610.07409）

- **论文：** [arXiv:2610.07409](https://arxiv.org/abs/2610.07409)
- **机构（周报所述，未逐条核实）：** 加州大学伯克利分校、加州理工学院
- **问题：** 高速转弯时，简单运动学模型无法预测RL速度跟踪器受轮胎打滑、延迟和执行器饱和影响后的真实闭环行为。
- **方法线索：** 在MPPI名义独轮车模型上增加学习式闭环残差；
- **详情节点：** [RACER：用残差闭环模型让 MPPI 驾驭轮足高速过弯](../../wiki/entities/paper-racer-residual-adaptive-wheeled-quadruped-racing.md)

#### ★ Model-Based Geometry-Aware Generative Optimization for Constrained Locomotion Planning（arXiv:2610.07772）

- **论文：** [arXiv:2610.07772](https://arxiv.org/abs/2610.07772)
- **机构（周报所述，未逐条核实）：** 卡内基梅隆大学、密歇根大学
- **问题：** 腿式机器人在强接触、碰撞及运动学约束下，传统优化容易陷入局部解，纯生成模型又难以严格满足约束。
- **方法线索：** 将任务相关的活跃约束几何引入模型式扩散生成过程，在生成候选轨迹时沿约束流形优化，可用于四足和人形受约束运动规划。
- **详情节点：** [2GO：几何感知的模型式生成优化做受约束腿式规划](../../wiki/entities/paper-2go-geometry-aware-generative-locomotion-planning.md)

#### ★ LLA-MPPI: Rapidly Adaptive Whole-Body Control of Legged Robots with GPU-Accelerated Parallel Simulations（arXiv:2610.10465）

- **论文：** [arXiv:2610.10465](https://arxiv.org/abs/2610.10465)
- **机构（周报所述，未逐条核实）：** 卡内基梅隆大学、麻省理工学院
- **问题：** 载荷、摩擦、坡度和执行器故障会在线改变接触动力学，而固定模型控制器和离线训练策略难以覆盖未知变化。
- **方法线索：** 并行运行大量不同物理参数和结构假设的接触仿真器，以历史误差选择当前模型，再由MPPI进行前瞻控制；
- **详情节点：** [LLA-MPPI：用 GPU 并行仿真做腿式全身控制的在线自适应](../../wiki/entities/paper-lla-mppi-legged-adaptive-wbc.md)

#### ClimbLab: MATLAB Simulation Platform for Legged Climbing Robotics（arXiv:2610.09315）

- **论文：** [arXiv:2610.09315](https://arxiv.org/abs/2610.09315)
- **机构（周报所述，未逐条核实）：** 东北大学（日本）空间机器人实验室
- **问题：** 腿式攀爬研究缺少便于教学、动力学分析和控制原型验证的轻量开源仿真工具。
- **方法线索：** 在MATLAB中集成攀爬机器人建模、接触、环境、控制和可视化功能，不依赖额外商业工具箱，并提供完整开源实现。
- **详情节点：** [ClimbLab：腿式攀爬机器人的 MATLAB 仿真平台](../../wiki/entities/paper-climblab-matlab-legged-climbing-sim.md)

## 说明

- ★ 为周报标注的“与表示与强化学习、视觉运动控制、Teacher–Student、复杂地形运动及项目相关方向高度相关”条目。
- 周报机构信息仅作线索；各详情节点的机构 tag 以 arXiv 页、项目页或仓库为准，注册表缺失的机构已略过。
