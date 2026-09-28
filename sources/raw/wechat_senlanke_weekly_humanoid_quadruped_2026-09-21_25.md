# 【9.21-9.25前沿论文动态】人形/四足35篇

Original senlanke 

 具身运控lab 

★ 表示与强化学习、视觉运动控制、Teacher–Student、复杂地形运动高度相关。

## 1. ★ FootQuery: Future-Touchdown-Guided Retrieval from Depth History for Perceptive Humanoid Locomotion

中文标题：FootQuery：未来落脚点引导的深度历史检索式人形感知运动

机构：清华大学智能产业研究院、北京科技大学、南洋理工大学、清华大学电子工程系

论文链接：https://arxiv.org/abs/2609.21447

解决的问题：脚即将落地时，相关台阶或平台可能已经离开当前相机视野，机器人无法从当前深度图判断落脚位置。

创新点：根据本体状态预测左右脚下一落点及不确定性，再以每只脚的预测落点从历史深度帧中检索相关区域；用实际接触点投影监督检索位置，并加入力辅助课程和台阶中线接触约束。G1完成室内外楼梯、平台和沟隙连续穿越。

相关性：深度视觉、历史信息、落脚预测、复杂地形。非常适合参考其“按未来落脚需求查询历史视觉”的结构。

## 2. ★ REDACT: Robust Perceptive Locomotion under Unseen Visual Corruption

中文标题：REDACT：面向未知视觉损坏的鲁棒感知运动控制

机构：南安普顿大学

论文链接：https://arxiv.org/abs/2609.25450

解决的问题：深度策略遇到训练中未覆盖的黑洞、无效测距、遮挡、噪声和视觉干扰时，会输出不可预测动作；单纯加入已知噪声增强无法覆盖未知损坏。

创新点：采用Teacher–Student框架，将持续特征遮蔽、改进视觉编码器和共识门控结合起来；门控仅使用干净观测进行近似共形校准，在不知道损坏类型的情况下判断哪些深度特征仍可信。只用干净仿真深度训练，也能迁移到结构化环境和森林场景。

相关性：Teacher–Student、深度黑洞与未知噪声、视觉运动控制；是本周与你的真机深度问题最直接相关的论文。

## 3. ★ Echo in the Steps: Learning Perceptive Humanoid Parkour with Gated Memory

中文标题：脚步回响：基于门控记忆的人形机器人感知式跑酷学习

机构：清华大学

论文链接：https://arxiv.org/abs/2609.28960

解决的问题：稀疏踏脚石和狭窄支撑面要求机器人记住已经离开视野的落脚信息，同时保证高速运动中左右脚交替落点。

创新点：用显著性先验筛选重要深度区域，再通过门控记忆跨帧保存落脚相关特征；加入交替步态损失和对称正则，降低策略固定使用同一条腿的倾向。

相关性：深度历史编码、左右腿交替、稀疏落脚地形。其交替损失与你当前“总是同一条腿先抬”直接相关。

## 4. ★ STRIDER: Stepping-Enabled Multi-Gait Hierarchical 3D Loco-Manipulation Framework for Humanoid Robots

中文标题：STRIDER：支持精准踏步与多步态的人形机器人分层三维移动操作框架

机构：北京人形机器人创新中心（X-Humanoid）

论文链接：https://arxiv.org/abs/2609.23483

解决的问题：速度指令策略无法精确控制每只脚的三维落点，而单独训练的精准踏步策略难以与自然行走及上肢操作统一。

创新点：结合AMP自然行走专家、三维落脚专家和笛卡尔上肢控制；提出LD-PPO，同时进行在线RL、DAgger动作重建和Teacher条件latent对齐，把异构专家蒸馏为统一Student。

相关性：Teacher–Student、latent蒸馏、精准落脚、多专家融合。

## 5. ★ PLAT: Sparse Timed Keyframe Motion Tracking for Humanoid Control via Privileged Latent Transition Learning

中文标题：PLAT：通过特权隐变量转移学习实现人形机器人稀疏定时关键帧跟踪

机构：武汉大学、BeingBeyond、北京大学

论文链接：https://arxiv.org/abs/2609.25754

解决的问题：传统动作跟踪需要逐帧稠密参考，无法作为接受稀疏目标的高层运动控制器。

创新点：先训练稠密动作专家，再利用稠密目标序列作为特权信息，通过DAgger学习latent转移先验；最后用强化学习只修正latent转移，而不是直接叠加动作残差。部署时仅需稀疏关键帧及到达时间。

相关性：特权信息、latent residual、DAgger蒸馏；与MoRE式“修改latent而非动作”高度一致。

## 6. ★ UniPoint: Unified Point-Level Sensor Fusion for Humanoid Locomotion Across Challenging Terrains

中文标题：UniPoint：面向复杂地形人形运动的统一点级传感器融合

机构：浙江大学、云深处科技、浙江省增材制造技术与装备重点实验室

论文链接：https://arxiv.org/abs/2609.23666

解决的问题：单个前向深度相机覆盖有限，高程图在激烈运动中会漂移，并容易漏掉细杆和薄墙；增加相机又会提高图像编码成本。

创新点：把360度激光雷达和两台深度相机提前融合为机身坐标系点集，再体素化为固定数量token；采用线性自注意力和本体查询交叉注意力编码，使计算量不随传感器数量增长，并通过传感退化注入训练单一全地形策略。

相关性：多传感器地形latent、复杂地形、感知失效鲁棒性。

## 7. HOTICE: Whole-Body Humanoid Object Transportation in Cluttered Environments

中文标题：HOTICE：拥挤环境中的人形机器人全身物体运输

机构：南加州大学

论文链接：https://arxiv.org/abs/2609.25363

解决的问题：搬运大型物体时，机器人本体和载荷都必须避障，同时上、下半身动作空间高度耦合。

创新点：分别为机器人和物体建立解耦势场；使用上下半身双智能体RL并通过共享状态和奖励保持协调；再将多个特权场景Teacher蒸馏成统一Student，实现未见拥挤场景的G1实机运输。

相关性：多Actor强化学习、Teacher–Student、复杂环境避障。

## 8. HIGenNTO: Scalable Humanoid Interaction Generation via Noise-Space Trajectory Optimization

中文标题：HIGenNTO：通过噪声空间轨迹优化扩展人形交互动作生成

机构：卡内基梅隆大学、庆应AI研究中心、庆应义塾大学

论文链接：https://arxiv.org/abs/2609.22611

解决的问题：接触丰富型人形交互动作容易受到人体遮挡和动作重定向误差影响，难以获得稳定、物理可执行的参考。

创新点：不直接优化动作序列，而是优化预训练文本动作模型的初始噪声，使生成结果同时满足接触、避碰、支撑和场景约束；生成的动作既可训练跟踪器，也可训练只依赖机载深度的视觉策略。

## 9. Learning Distance-Conditioned Object Transport for Humanoid Loco-Manipulation from a Single Motion Clip

中文标题：从单段动作学习距离条件化的人形机器人移动运输

机构：韩国电子技术研究院、首尔大学、高丽大学

论文链接：https://arxiv.org/abs/2609.21467

解决的问题：单段示范中的中间位置只是经过状态，策略无法把它们识别成“应在此停止”的目标位置。

创新点：把原示范的终止片段移动到不同运输距离处，构造新的参考；冻结的跟踪Teacher执行这些参考，再按实际到达位置重新标注并蒸馏成无需参考动作的策略，最后通过RL增强鲁棒性。

## 10. LIMBO: Learning and Internalizing Model-Free Barrier Objectives for Agile and Safe Whole-Body Control

中文标题：LIMBO：面向敏捷安全全身控制的无模型屏障目标学习与内化

机构：亚马逊、华盛顿大学、西北大学、加州理工学院

论文链接：https://arxiv.org/abs/2609.22075

解决的问题：高自由度人形机器人的避碰和动态平衡很难建立可复用的解析安全证书，在线安全过滤又会增加控制开销。

创新点：围绕冻结控制器的残差动作学习状态—动作控制屏障函数，并在可恢复边界附近进行风险引导采样；训练任务策略时让屏障函数提供动作级安全反馈，把安全结构直接内化进策略。

## 11. Whole-Body UMI: Transferring UMI Manipulation Skills to Humanoid Whole-Body Manipulation via Real-Time Motion Generation

中文标题：Whole-Body UMI：通过实时动作生成将UMI操作技能迁移到人形全身操作

机构：浙江大学、香港具身智能实验室、Mondo Robotics、香港中文大学

论文链接：https://arxiv.org/abs/2609.22829

解决的问题：UMI只采集末端轨迹，无法唯一确定人形机器人的腿、躯干和手臂如何协调。

创新点：把任务语义学习和全身协调学习解耦：扩散策略从UMI数据预测末端轨迹，独立训练的实时动作生成器再把末端轨迹转换为全身参考，并通过异步闭环层级在G1上执行。

## 12. Opt2VLA: Force-Aware Vision-Language-Action for Contact-Rich Humanoid Whole-Body Manipulation

中文标题：Opt2VLA：面向接触丰富型人形全身操作的力感知视觉—语言—动作模型

机构：佐治亚理工学院

论文链接：https://arxiv.org/abs/2609.23968

解决的问题：只预测位置或关节目标的VLA无法区分几何动作相似、但接触力要求不同的任务。

创新点：让多任务VLA同时输出运动目标和连续接触力参考，再由任务专属RL全身控制器跟踪；使用带显式力约束的全身轨迹优化自动产生动力学可行监督数据。

## 13. Smoothness as a Constraint for Stable Humanoid Locomotion

中文标题：将平滑性作为稳定人形运动的显式约束

机构：瑞典厄勒布鲁大学

论文链接：https://arxiv.org/abs/2609.24552

解决的问题：把动作平滑性写成奖励会与速度跟踪竞争，而且统一约束全身会使下肢反应迟缓或上身晃动。

创新点：提出DeCap约束强化学习，分别为上下半身设置物理运动约束，并以有界屏障惩罚在接近约束边界前介入；同一组约束可迁移到不同地形，不需要重新调整平滑奖励。

## 14. PredActor: Predictive Action Diffusion for Steerable Onboard Humanoid Control

中文标题：PredActor：面向机载可控人形控制的预测式动作扩散策略

机构：哈尔滨工业大学、上海创智学院、RoboParty Lab、清华大学、上海交通大学

论文链接：https://arxiv.org/abs/2609.24840

解决的问题：参考动作生成器与跟踪器分离时，生成动作可能超出跟踪能力；纯动作扩散又缺少可供测试时引导的未来状态。

创新点：单个策略联合生成可执行动作和内部未来状态轨迹；用无分类器引导选择文本行为，用分类器引导修改预测状态，实际只执行动作。通过滚动去噪将Jetson Orin NX推理时间压到20毫秒以内。

## 15. Brace Yourself: Task-Conditioned Environmental Bracing for Forceful Humanoid Manipulation

中文标题：撑住自己：面向强力人形操作的任务条件化环境支撑

机构：昆士兰科技大学、澳大利亚协作机器人中心

论文链接：https://arxiv.org/abs/2609.25486

解决的问题：人形机器人执行推压等强力操作时，末端反作用力会破坏全身平衡。

创新点：先根据工作手的任务区域和目标力优化另一只手的支撑位姿，再由两个同步RL策略执行支撑和操作；G1可持续产生最高60 N接触力，而无支撑基线仅为13.5 N。

## 16. FRAMES: Failure Recovery And Monitoring of Embodied Skills for Humanoid Loco-Manipulation

中文标题：FRAMES：人形移动操作技能的失败监测与恢复

机构：杜克大学

论文链接：https://arxiv.org/abs/2609.22538

解决的问题：语言规划器即使选择了正确技能，也无法保证机器人在接近、抓取、搬运和放置阶段正确执行。

创新点：以VLM监控器融合多视角时序图像、机器人状态和接触证据；检测失败后停止当前技能，并向恢复智能体返回结构化原因，同时利用记忆模块复用历史恢复经验。

## 17. PRIMO: Prior-Informed Odometry from Human-Motion Tracking for Humanoid Robots

中文标题：PRIMO：基于人体动作跟踪先验的人形机器人本体里程计

机构：武汉大学、智元机器人

论文链接：https://arxiv.org/abs/2609.23610

解决的问题：只用某一控制策略的数据训练里程计，会过拟合该策略的运动分布；无约束网络在Sim-to-Real误差下也容易产生不合理估计。

创新点：通过机器人跟踪大量重定向人体动作生成更广运动分布，再以物理和左右对称先验约束速度及旋转预测，同时保留原始传感上下文支路。

## 18. MATE: Multi-Agent Virtual Teleoperation Platform for Humanoid Collaboration Data Collection

中文标题：MATE：面向人形机器人协作数据采集的多智能体虚拟遥操作平台

机构：上海科技大学、南洋理工大学

论文链接：https://arxiv.org/abs/2609.26520

解决的问题：多个人形协作数据需要多台实机、多人同步操作和频繁复位，采集成本很高。

创新点：允许异地操作者在同一物理仿真环境中同步控制多个人形；提出执行对齐交互采样，重点抽取任务推进和接触切换片段，构建24.1小时、2500条协作轨迹，并实现虚拟示范到实机零样本迁移。

## 19. Sample, Simulate, Select: Physics-in-the-Loop Text-to-Motion for Humanoids Without Training

中文标题：采样、仿真、筛选：无需训练的物理闭环人形文本动作生成

机构：波恩大学、Lamarr机器学习与人工智能研究所

论文链接：https://arxiv.org/abs/2609.26420

解决的问题：文本动作模型生成的人体动作看起来合理，但经过重定向后未必能被机器人稳定执行。

创新点：每条文本生成多个候选动作，将其全部重定向并交给真实部署用跟踪器在刚体仿真中执行，按实际动力学表现筛选最佳候选；不训练新的生成器或控制器。

## 20. Banana Kick: Response-Informed Skill Evolution for Humanoid Soccer

中文标题：香蕉球：基于响应信息的人形足球技能演化

机构：卡内基梅隆大学、德克萨斯大学阿灵顿分校、通用汽车

论文链接：https://arxiv.org/abs/2609.27269

解决的问题：普通射门策略附近几乎不产生旋转球，旋转奖励在当前策略分布内缺少有效梯度，RL难以探索出弧线球接触方式。

创新点：提出RISE，根据缓存轨迹中物理响应对候选目标变化进行排序，只接受确实提高旋转且保持射门可靠性的更新，逐步把普通射门先验推进到新的足球接触区域。

## 21. DAVIS: A Depth-Only End-to-End Active-Vision Framework for Humanoid Soccer Skills

中文标题：DAVIS：仅使用深度图的人形足球端到端主动视觉框架

机构：松延动力、清华大学

论文链接：https://arxiv.org/abs/2609.28175

解决的问题：机器人接近、踢击和恢复时头部视角剧烈变化，足球容易离开视野；额外目标检测和规划模块会割裂感知—动作闭环。

创新点：只输入头部深度图、本体历史和可选任务命令，直接输出25自由度PD目标；通过训练期几何辅助任务、真实值到预测值退火、课程学习和AMP先验联合学习头部主动转向与全身足球技能。

相关性：端到端深度视觉、主动感知、直接关节控制。

## 22. ForgetMimic: Motion Unlearning for Reinforcement Learning Humanoid Control

中文标题：ForgetMimic：面向强化学习人形控制的动作遗忘

机构：北京理工大学、弗吉尼亚大学、山东大学、东北大学

论文链接：https://arxiv.org/abs/2609.28378

解决的问题：动作模仿策略训练完成后，缺少只删除恶意、低质量或受版权保护动作，同时保留其他技能的方法。

创新点：提出动作级策略遗忘，使指定动作的跟踪能力定向退化，同时约束未删除动作的性能保持；并针对人形策略中导致遗忘失败的两个训练机制进行修正。

## 23. Humanoid Locomotion with a Fly-Inspired Recurrent Controller

中文标题：采用果蝇启发循环控制器的人形机器人运动

机构：香港科技大学、Zenbot、南洋理工大学、香港理工大学

论文链接：https://arxiv.org/abs/2609.27001

解决的问题：生物神经回路启发控制器究竟依靠哪些信息维持运动，通常缺少可追踪的机制分析。

创新点：将3609个连续神经状态连接到G1仿真本体，利用状态重置和路径替换实验定位行为来源；结果表明持续运动主要依赖本体—指令输入和循环电机状态。

## 24. Learning Expressive Humanoid Locomotion from Monocular Runway Videos for Robot Fashion Shows

中文标题：从单目走秀视频学习表现型人形机器人行走

机构：维尔纽斯大学、肯特州立大学

论文链接：https://arxiv.org/abs/2609.27003

解决的问题：常规人形策略强调稳定和速度，无法复现模特走秀中的窄步宽、姿态和全身风格。

创新点：建立单目视频人体动作恢复、机器人重定向、动作修正、策略训练和实机部署的完整流程，在Booster K1上执行不同走秀步态。

## 25. EmoPose: Vision-Language Model Guided Emotion-Aware Gesture Generation for Humanoid Robots

中文标题：EmoPose：视觉语言模型引导的人形机器人情感手势生成

机构：香港科技大学（广州）、山东大学、RoboScience

论文链接：https://arxiv.org/abs/2609.23414

解决的问题：开放式语言交互需要灵活理解情绪和语境，但基础模型直接输出关节命令难以保证可执行性。

创新点：VLM只选择手势类别、动作版本、强度和语音触发点；机器人本地动作库负责生成、验证和调度14自由度轨迹，使语义规划与底层安全执行分离。

## 26. TactileStep: Sole Tactile Learning for Regulating Foot-Terrain Interaction in Humanoid Locomotion

中文标题：TactileStep：利用足底触觉学习调节人形机器人足地交互

机构：清华大学

论文链接：https://arxiv.org/abs/2609.28959

解决的问题：机器人即使成功越障，也可能出现硬着陆、踩边和支撑面积不足，而普通深度策略无法直接感知足底压力分布。

创新点：将仿真接触与实机压力鞋垫对齐，使用触觉和运动状态识别摆动、预着陆、落地和支撑阶段，再施加分阶段接触奖励；落地峰值力最高降低48.8%。

## 1. ★ SABER: Learning Attention-based Semantic Affordance for Legged Locomotion

中文标题：SABER：面向腿式运动的注意力语义可供性学习

机构：新加坡科技研究局A*STAR先进智能与计算研究所、南洋理工大学

论文链接：https://arxiv.org/abs/2609.21572

解决的问题：仅根据地形几何，机器人可能把管道、草地或易碎箱体当作可踩踏区域。

创新点：将三维几何和语义接触代价编码到统一地形图，在交叉注意力logit中加入与足端距离相关的有符号语义偏置，使危险区域只在仍会影响下一落脚点时改变注意力。Unitree B2完成室内外语义避踩。

相关性：强化学习、感知落脚、语义地形。

## 2. Duty Factor Predicts Robust Constrained Quadrupedal Locomotion Across Gait Types

中文标题：占空比对不同步态下四足受限运动鲁棒性的预测作用

机构：迈阿密大学、卡内基梅隆大学

论文链接：https://arxiv.org/abs/2609.22073

解决的问题：步行或小跑等步态名称不能充分解释机器人在窄梁和扰动环境中的稳定性。

创新点：在轨迹优化加LQR、学习控制器和质心MPC三种框架中统一分析步态参数，发现占空比比名义步态类别更能预测误差收敛；进一步让策略按地形宽度选择占空比。

## 3. MimicAgent: Quadruped Skills via Prompt-to-Trajectory Generation

中文标题：MimicAgent：通过提示词到轨迹生成学习四足技能

机构：卡内基梅隆大学

论文链接：https://arxiv.org/abs/2609.24145

解决的问题：训练新的四足技能通常需要人工制作参考动作或重新设计奖励。

创新点：让大模型根据自然语言和技能上下文自动生成粗略轨迹，再将其作为示例引导强化学习的参考目标；把技能设计转化为“提示词—轨迹—策略”的自动流程。

## 4. ★ SG-CPG: Severity-Gated Central Pattern Generators for Adaptive Quadruped Locomotion under Continuous Actuator Degradation

中文标题：SG-CPG：面向执行器连续退化的严重度门控四足中央模式发生器

机构：普渡大学

解决的问题：执行器性能通常是逐渐下降，而现有容错控制常把关节简单分成正常或失效两种状态。

创新点：冻结健康CPG策略，增加全腿协调残差门和弱腿振幅门，两者都由故障严重度连续调节；Go2在最高93%小腿力矩退化下仍完成多数实机运动测试。

相关性：强化学习残差、连续故障适应。

## 5. An Analysis of Streaming Deep Reinforcement Learning for Adaptive Continual Learning in Robotics

中文标题：流式深度强化学习用于机器人自适应持续学习的分析

机构：耶鲁大学

论文链接：https://arxiv.org/abs/2609.28807

解决的问题：预训练四足策略遇到未建模的机器人变化、环境变化或目标变化时，离线数据扩充无法提供即时适应。

创新点：每次只使用最新交互样本进行流式RL更新，并分析优化器和策略可塑性保持机制；四足实验中相较预训练策略，部分变化条件下成功率最高提高90%。

## 6. FlyCNS: Connectome-Grounded Information Organization for Communication-Constrained Embodied Control

中文标题：FlyCNS：基于神经连接组先验的受限通信具身控制信息组织

机构：印第安纳大学布卢明顿分校

论文链接：https://arxiv.org/abs/2609.28816

解决的问题：分布式腿部控制中，并非所有局部传感信息都能持续传到中央控制器，需要决定哪些计算留在腿部、哪些信息值得发送。

创新点：借鉴果蝇脑—神经索连接组，设置局部感知运动模块和独立上下行通信路径；信息内容、发送时机和运动策略由RL联合学习，在仅使用约21%通信量时仍保持接近完整通信策略的跟踪性能。

## 1. Online Sim-to-Real Adaptation via Closed-Loop System Modeling

中文标题：基于闭环系统建模的在线Sim-to-Real适应

机构：杜克大学

论文链接：https://arxiv.org/abs/2609.28878

解决的问题：策略迁移到实机后可能保持稳定，却因残余动力学误差产生持续速度或轨迹跟踪偏差。

创新点：把机器人和已部署策略整体视为闭环动力系统，直接学习“任务指令—实际响应”关系；在线只优化输入给原控制器的参考命令，不修改策略参数，也不辨识完整物理模型。在双足速度跟踪和移动操作硬件上验证。

## 2. When to Waddle: A Comparative Study of Bipedal Torso-Stabilization on Low-Friction Surfaces

中文标题：何时应该摇摆行走：低摩擦表面双足躯干稳定策略对比研究

机构：卡内基梅隆大学、纽约大学

论文链接：https://arxiv.org/abs/2609.21185

解决的问题：低摩擦地面限制可用地面反力，直立步态容易打滑，但躯干侧移和质心高度应如何选择尚不明确。

创新点：在五执行器双足平台上比较直立步态与企鹅式“躯干移到支撑腿上方”步态，并系统改变质心高度和摩擦系数；发现低摩擦时高质心企鹅步态速度和能效更好，而较高摩擦时低质心配置更优。

## 3. Spiderbot: An Open-Source Energy-Efficient Hexapod with Passive Gravity Compensation

中文标题：Spiderbot：采用被动重力补偿的开源节能六足机器人

机构：印度博拉理工学院皮拉尼分校果阿校区

论文链接：https://arxiv.org/abs/2609.26989

解决的问题：常见三自由度腿增加重量和持续支撑力矩，不利于低成本、长续航多机器人研究。

创新点：每条腿采用两自由度四连杆和被动弹簧支撑体重，站立功耗降至1.5 W；整机成本低于400美元，并开源CAD、装配资料、mjlab强化学习训练代码、部署代码和模型，在斜坡、粗糙地形及台阶上完成Sim-to-Real。