# 【9.28–10.2 前沿论文动态】人形/四足49篇

- **作者：** senlanke（具身运控lab）
- **发表日期：** 2026-10-06
- **原文：** <https://mp.weixin.qq.com/s/JTzpow1Ls1T4CCHSvh5mQA>
- **归档依据：** 用户提供的 PDF《【9.28–10.2 前沿论文动态】人形/四足49篇》。此页提炼文章条目与独立详情节点映射；论文技术事实仍以 arXiv 原文为准。
- **条目数：** 53 个论文条目；同一 arXiv 论文只保留一个 wiki 详情节点。

## 条目与详情节点

### WB-WAM: Heterogeneous Body-Hand Pre-training for Humanoid Loco- Manipulation（arXiv:2609.34199）

- **论文：** [arXiv:2609.34199](https://arxiv.org/abs/2609.34199)
- **问题：** 人体视频、动作捕捉与机器人数据对身体和手部的动作表示不统一
- **方法线索：** 构建统一的身体—手部物理动作表示，依次完成异构预训练、中间训练和实机任务适配
- **详情节点：** [WB-WAM：面向人形移动操作的异构身体—手部联合预训练](../../wiki/entities/paper-wb-wam.md)

### GAE: General Action Expert for Real-Time Humanoid Teleoperation（arXiv:2609.34233）

- **论文：** [arXiv:2609.34233](https://arxiv.org/abs/2609.34233)
- **问题：** 人体动作含噪且与机器人动力学不匹配，通信和执行延迟还会破坏人机同步
- **方法线索：** 先由特权生成策略产生机器人可执行轨迹，再蒸馏给部署策略；同时通过延迟条件化预测补 偿遥操作延迟
- **详情节点：** [GAE：面向实时人形遥操作的通用动作专家](../../wiki/entities/paper-gae-general-action-expert.md)

### Model-Informed Safe Reinforcement Learning for Bipedal Locomotion via Step-to-Step Prediction（arXiv:2609.34486）

- **论文：** [arXiv:2609.34486](https://arxiv.org/abs/2609.34486)
- **问题：** 纯强化学习步行策略缺少可解释的安全约束，受扰时可能给出危险落脚点
- **方法线索：** 从ALIP模型推导离散指数控制障碍函数，同时用于训练奖励塑形和运行时落脚动作过滤
- **详情节点：** [基于步间预测的模型引导安全双足运动强化学习](../../wiki/entities/paper-model-informed-safe-reinforcement-learning.md)

### HOI-Retarget: Contact-Centric Retargeting for Human-Object Interaction（arXiv:2609.34674）

- **论文：** [arXiv:2609.34674](https://arxiv.org/abs/2609.34674)
- **问题：** 只匹配人体和机器人的关节或末端位置，不能保证手部接触物体上功能相同的位置
- **方法线索：** 把物体坐标系中的接触点作为直接优化目标，同时约束身体跟踪、足部支撑和平滑性
- **详情节点：** [HOI-Retarget：面向人—物交互的接触中心动作重定向](../../wiki/entities/paper-hoi-retarget.md)

### DexWeave: Learning Dexterous Humanoid Loco-Manipulation from Human Demonstrations（arXiv:2609.34724）

- **论文：** [arXiv:2609.34724](https://arxiv.org/abs/2609.34724)
- **问题：** 分别处理身体、手部和物体轨迹会破坏三者之间的协调与接触关系
- **方法线索：** 先分别初始化身体和手部动作，再进行交互一致的联合细化，并训练解剖结构感知的全身控 制策略
- **详情节点：** [DexWeave：从人体示范学习灵巧人形移动操作](../../wiki/entities/paper-dexweave-humanoid-loco-manipulation.md)

### CoHuB: A Simulation Benchmark for Multi-Humanoid Collaboration（arXiv:2609.34782）

- **论文：** [arXiv:2609.34782](https://arxiv.org/abs/2609.34782)
- **问题：** 现有人形基准主要评估单机器人能力，缺少第一视角条件下的多机器人协作任务
- **方法线索：** 提供10项双人或三人协作任务，并通过多操作者VR遥操作采集同步示范
- **详情节点：** [CoHuB：多个人形机器人协作仿真基准](../../wiki/entities/paper-cohub.md)

### Passive-Dynamic-Walking-Inspired Dynamics Guidance for Energy- Efficient Humanoid Locomotion（arXiv:2609.35935）

- **论文：** [arXiv:2609.35935](https://arxiv.org/abs/2609.35935)
- **问题：** 仅惩罚力矩和功率并不能直接帮助策略发现具有经济性的步间动力学协调
- **方法线索：** 训练早期使用倾斜重力构造类似下坡的动力学条件，再逐渐恢复正常重力，使策略先发现节 能步态、再适应平地
- **详情节点：** [受被动动态行走启发的节能人形运动动力学引导](../../wiki/entities/paper-passive-dynamic-walking-inspired-dynamics.md)

### Uni-VLaT: Whole-Body Tactile Adaptation of VLA Policies for Humanoid Loco-Manipulation（arXiv:2609.35450）

- **论文：** [arXiv:2609.35450](https://arxiv.org/abs/2609.35450)
- **问题：** 纯视觉VLA难以判断遮挡后的接触状态、物体滑移及身体与环境的接触
- **方法线索：** 把时空触觉token接入预训练VLA，并联合预测未来触觉、本体状态和视觉信息
- **详情节点：** [Uni-VLaT：人形移动操作VLA策略的全身触觉适配](../../wiki/entities/paper-uni-vlat.md)

### Humanoid Loco-Manipulation With Discrete VLA Model（arXiv:2609.35709）

- **论文：** [arXiv:2609.35709](https://arxiv.org/abs/2609.35709)
- **问题：** 人形机器人腿、躯干、双臂和手部动作异构，难以使用统一的VLA动作表示
- **方法线索：** 分别构建末端、身体、手部和运动学动作tokenizer，再把离散动作token直接并入语言模型词 表
- **详情节点：** [基于离散视觉—语言—动作模型的人形移动操作](../../wiki/entities/paper-holo-m.md)

### KPI: A Promptable Kernel for Physical Interaction on Humanoids（arXiv:2609.36151）

- **论文：** [arXiv:2609.36151](https://arxiv.org/abs/2609.36151)
- **问题：** 开门、书写和挤压搬运等任务对轨迹、力和柔顺性的要求不同
- **方法线索：** 把高层任务转化为动作参考和“交互契约”，统一指定增益、前馈力及接触约束，并在线适配
- **详情节点：** [KPI：可提示的人形机器人物理交互控制内核](../../wiki/entities/paper-kpi.md)

### EquivDP3: A SIM(3)-Invariant Point-Cloud Encoder for Data-Efficient Humanoid Loco-Manipulation（arXiv:2609.36575）

- **论文：** [arXiv:2609.36575](https://arxiv.org/abs/2609.36575)
- **问题：** 普通点云策略不能自然泛化到物体的旋转、平移和尺度变化，少量示范下容易过拟合
- **方法线索：** 为扩散策略加入SIM(3)等变点云编码器，高层以6 Hz生成全身指令，底层以冻结的运动策略和 微分逆运动学执行
- **详情节点：** [EquivDP3：面向数据高效人形移动操作的SIM(3)不变点云编码器](../../wiki/entities/paper-equivdp3.md)

### OTRetarget: Joint Robot and Object Motion Retargeting via Optimal Transport（arXiv:2609.36602）

- **论文：** [arXiv:2609.36602](https://arxiv.org/abs/2609.36602)
- **问题：** 固定物体轨迹后只调整机器人动作，可能造成手部脱离物体或足部接触失真
- **方法线索：** 通过熵正则最优传输，同时优化机器人和多个物体的轨迹及其表面交互关系
- **详情节点：** [OTRetarget：基于最优传输的机器人与物体运动联合重定向](../../wiki/entities/paper-otretarget.md)

### Track-and-Complete: Learning Humanoid Skills from a Single Failed Human Video（arXiv:2609.36924）

- **论文：** [arXiv:2609.36924](https://arxiv.org/abs/2609.36924)
- **问题：** 视频模仿通常要求成功示范，但失败视频在失败前仍包含有用动作并隐含任务目标
- **方法线索：** 识别不可挽回点，先模仿失败前的有效动作，再通过任务完成奖励学习后续动作
- **详情节点：** [跟踪并完成：从单段失败人体视频学习人形技能](../../wiki/entities/paper-track-and-complete.md)

### Learning Expressive and Compositional Motion Representation via Spectral Skills（arXiv:2609.37677）

- **论文：** [arXiv:2609.37677](https://arxiv.org/abs/2609.37677)
- **问题：** 高层规划器需要容易预测、准确执行且可组合的低维动作接口
- **方法线索：** 学习谱技能latent，使冻结的控制器能够连续串联技能，并通过latent加减组合转向等行为
- **详情节点：** [基于谱技能的表达性与可组合动作表征学习](../../wiki/entities/paper-spectral-skills-motion-representation.md)

### EgoHumanoid-V2: Human-to-Humanoid Transfer of Coordinated Whole-Body Skills for Loco-Manipulation（arXiv:2609.37181）

- **论文：** [arXiv:2609.37181](https://arxiv.org/abs/2609.37181)
- **问题：** 第一视角人体示范中的视觉、身体动作和接触关系与机器人本体不一致
- **方法线索：** 通过视觉对齐、运动学修正和动力学精炼，把无需机器人参与采集的人体示范转化为可训练 数据
- **详情节点：** [EgoHumanoid-V2：移动操作协调全身技能的人到人形迁移](../../wiki/entities/paper-egohumanoid-v2.md)

### EgoAlign: Bridging the Human-Humanoid Gap for Long-Range Loco- Manipulation（arXiv:2609.38046）

- **论文：** [arXiv:2609.38046](https://arxiv.org/abs/2609.38046)
- **问题：** 长距离人体示范缺少机器人状态，身体尺度和控制响应差异会导致示范不可执行
- **方法线索：** 利用目标机器人仿真反馈校正落脚、上身交互尺度和控制器误差，再通过因果回放恢复机器 人状态与动作token
- **详情节点：** [EgoAlign：弥合长距离移动操作中的人与人形机器人差距](../../wiki/entities/paper-egoalign.md)

### CrossBFM: Distilling a Shared Latent Behavior Space Across Humanoid Embodiments（arXiv:2609.38087）

- **论文：** [arXiv:2609.38087](https://arxiv.org/abs/2609.38087)
- **问题：** 不同机器人独立训练得到的行为latent不对齐，无法跨本体复用
- **方法线索：** 利用动作重定向的逐帧对应关系，把行为基础模型的latent蒸馏到无本体专属参数的统一编码 器
- **详情节点：** [CrossBFM：跨人形机器人本体蒸馏共享潜在行为空间](../../wiki/entities/paper-crossbfm-shared-latent-behavior.md)

### Counterfactual Video Generation Enables Scalable Humanoid Loco- Manipulation（arXiv:2609.38172）

- **论文：** [arXiv:2609.38172](https://arxiv.org/abs/2609.38172)
- **问题：** 少量真实视频无法覆盖不同物体外观、尺寸和摆放方式
- **方法线索：** 从真实视频生成更换物体后的反事实交互视频，再重建并重定向为机器人—物体训练轨迹
- **详情节点：** [反事实视频生成实现可扩展人形移动操作学习](../../wiki/entities/paper-prism-real2sim2real.md)

### GestAdapt: Workspace-Conditioned Co-Speech Gesture Generation for Humanoid Robots（arXiv:2609.38400）

- **论文：** [arXiv:2609.38400](https://arxiv.org/abs/2609.38400)
- **问题：** 靠近墙壁或物体时，普通语音手势可能超出机器人可用空间
- **方法线索：** 把手腕允许工作空间作为生成条件，直接生成语义匹配且空间可行的手势
- **详情节点：** [GestAdapt：工作空间条件化的人形伴随语音手势生成](../../wiki/entities/paper-gestadapt.md)

### Dense Temporal Motion Retargeting for Legged Robots（arXiv:2609.38617）

- **论文：** [arXiv:2609.38617](https://arxiv.org/abs/2609.38617)
- **问题：** 跳跃等动作的时序与控制耦合，统一缩放整段动作会破坏不需要调整的部分
- **方法线索：** 用GPU并行采样MPC联合优化每个控制步的时间和动作，只局部改变必要片段
- **详情节点：** [面向腿式机器人的稠密时间动作重定向](../../wiki/entities/paper-dense-temporal-motion-retargeting.md)

### CEER2: Directional and Tunable End-Effector and Root Compliance for Humanoid Loco-Manipulation（arXiv:2609.38709）

- **论文：** [arXiv:2609.38709](https://arxiv.org/abs/2609.38709)
- **问题：** 接触任务要求机器人在部分方向柔顺、其他方向保持精度，同时控制躯干对外力的响 应
- **方法线索：** 分层强化学习控制器调制冻结的全身跟踪策略，分别设置末端方向柔顺性和根部抗扰、阻尼 跟随模式
- **详情节点：** [CEER²：人形移动操作中的方向可调末端与根部柔顺控制](../../wiki/entities/paper-ceer2-directional-compliance.md)

### Locomotion-Grounded Humanoid Soccer: Task-Gated Reinforcement Learning of a Multi-Directional Kicking Library（arXiv:2609.38852）

- **论文：** [arXiv:2609.38852](https://arxiv.org/abs/2609.38852)
- **问题：** 现有足球策略围绕单段踢球参考训练，基础运动能力弱且技能切换复杂
- **方法线索：** 先训练通用速度指令运动策略，再把七种踢球技能作为任务门控层叠加，所有技能均返回同 一可控运动状态
- **详情节点：** [以运动为基础的人形足球：多方向踢球库的任务门控强化学习](../../wiki/entities/paper-locomotion-grounded-humanoid-soccer.md)

### NEXUS: Perceptive Whole-Body Control for Terrain-Adaptive Teleoperation（arXiv:2609.39000）

- **论文：** [arXiv:2609.39000](https://arxiv.org/abs/2609.39000)
- **问题：** 操作者在平地运动时，机器人可能位于台阶、斜坡或平台边缘，直接复制人体动作会产 生错误接触
- **方法线索：** 生成同一动作在不同地形上的近1000小时配对数据，通过Teacher–Student训练地形感知全身控 制器
- **详情节点：** [NEXUS：面向地形自适应遥操作的感知式全身控制](../../wiki/entities/paper-nexus-terrain-adaptive-teleoperation.md)

### OccluDex: Hierarchical 3D Visuo-Tactile Representation Learning for Egocentric Dexterous Manipulation under Self-Occlusion（arXiv:2609.39017）

- **论文：** [arXiv:2609.39017](https://arxiv.org/abs/2609.39017)
- **问题：** 操作手会遮挡物体表面与接触区域，导致第一视角视觉状态估计失效
- **方法线索：** 分层融合局部触觉接触与全局三维视觉几何，在遮挡过程中维持物体和接触状态表征
- **详情节点：** [OccluDex：自遮挡第一视角灵巧操作的分层三维视触觉表征学习](../../wiki/entities/paper-occludex.md)

### RoboAssist: Interactive Human-Humanoid Planning for Long-Horizon Surgical Assistance（arXiv:2609.39384）

- **论文：** [arXiv:2609.39384](https://arxiv.org/abs/2609.39384)
- **问题：** 手术流程和人员请求动态变化，完整重新规划延迟高且难以持续维持安全约束
- **方法线索：** 分开维护人类流程和机器人任务，只重规划受变化影响的任务后缀，并统一监督导航、交接 和全身执行
- **详情节点：** [RoboAssist：面向长时手术辅助的交互式人—人形规划](../../wiki/entities/paper-roboassist.md)

### IronMind: Scaling Humanoid Dexterous Manipulation via Camera- Space Ego-Centric Pretraining（arXiv:2609.39403）

- **论文：** [arXiv:2609.39403](https://arxiv.org/abs/2609.39403)
- **问题：** 人体手部与机器人末端结构不同，普通第一视角视频还缺少可靠躯干运动
- **方法线索：** 在相机坐标中对齐人类与机器人动作，对超过一万小时异构数据进行质量加权预训练
- **详情节点：** [IronMind：通过相机空间第一视角预训练扩展人形灵巧操作](../../wiki/entities/paper-ironmind.md)

### ECHO-G: Embodied Co-speech Humanoid Motion Generation（arXiv:2609.39575）

- **论文：** [arXiv:2609.39575](https://arxiv.org/abs/2609.39575)
- **问题：** 伴随语音动作需要同时匹配语音韵律、文本语义和机器人本体约束
- **方法线索：** 以音频帧特征和文本token共同条件化流匹配Transformer，直接生成机器人全身动作
- **详情节点：** [ECHO-G：具身人形机器人伴随语音动作生成](../../wiki/entities/paper-echo-g-cospeech-humanoid.md)

### Towards a General Humanoid Loco-Manipulation Model via Egocentric Whole-Body Human Data Pretraining（arXiv:2610.00438）

- **论文：** [arXiv:2610.00438](https://arxiv.org/abs/2610.00438)
- **问题：** 机器人全身操作数据昂贵，普通第一视角视频缺少可直接执行的机器人动作
- **方法线索：** 构建500小时HumanVerse-500数据集，通过互联网数据、人体数据和机器人数据三阶段训练全 身VLA模型
- **详情节点：** [通过第一视角全身人体数据预训练通用人形移动操作模型](../../wiki/entities/paper-lambda0-egocentric-human-pretraining.md)

### Toward Humanoid Robots in Construction: A Teleoperation Feasibility Study（arXiv:2610.00718）

- **论文：** [arXiv:2610.00718](https://arxiv.org/abs/2610.00718)
- **问题：** 完全自主施工尚不成熟，需要评估人形遥操作能否完成真实建筑任务
- **方法线索：** 结合XR上身控制与脚踏式行走控制，在工具运输和墙面涂刷任务中进行量化验证
- **详情节点：** [迈向建筑施工人形机器人：遥操作可行性研究](../../wiki/entities/paper-toward-humanoid-robots-in-construction.md)

### Reactive Humanoid Multi-Contact Using Learned Stability Models（arXiv:2610.00823）

- **论文：** [arXiv:2610.00823](https://arxiv.org/abs/2610.00823)
- **问题：** 机器人受推后仅靠双脚无法恢复时，需要快速选择墙面等环境中的手部支撑位置
- **方法线索：** 学习冲击后的压力中心可控区域，快速评分候选手部接触点并执行支撑
- **详情节点：** [基于学习式稳定性模型的人形机器人反应式多点接触](../../wiki/entities/paper-reactive-humanoid-multi-contact-using.md)

### MASkillBlender: Decentralized Whole-Body Coordination for Multi- Humanoid Loco-Manipulation via Skill Blending（arXiv:2610.01102）

- **论文：** [arXiv:2610.01102](https://arxiv.org/abs/2610.01102)
- **问题：** 多机器人全身协同的联合动作空间很大，逐任务设计参考动作和奖励成本高
- **方法线索：** 高层多智能体策略只学习如何混合预训练的单机器人技能，通过局部观测进行分散控制
- **详情节点：** [MASkillBlender：通过技能混合实现多个人形机器人分散式全身协同](../../wiki/entities/paper-maskillblender.md)

### Continue, Abort, or Fall: Viability-Aware Policy Selection for Safe Humanoid Acrobatics（arXiv:2610.01397）

- **论文：** [arXiv:2610.01397](https://arxiv.org/abs/2610.01397)
- **问题：** 翻腾等动作失控后，单一跟踪策略无法判断应该继续、主动终止还是保护性跌倒
- **方法线索：** 分别训练任务、终止和跌倒策略，再用短时域可行性预测器逐控制周期选择策略
- **详情节点：** [继续、终止还是受控跌倒：面向安全人形特技的可行性策略选择](../../wiki/entities/paper-continue-abort-or-fall.md)

### HumanoidToolBench: Benchmarking Humanoid Tool Use from Selection to Mobile Execution（arXiv:2610.02089）

- **论文：** [arXiv:2610.02089](https://arxiv.org/abs/2610.02089)
- **问题：** 现有工具使用基准多局限于桌面机械臂，缺少移动接近和全身执行
- **方法线索：** 提供55种工具、分层任务结构和约3100段人体示范，并在Unitree G1上验证
- **详情节点：** [HumanoidToolBench：从工具选择到移动执行的人形工具使用基准](../../wiki/entities/paper-humanoidtoolbench.md)

### InterEvolve: Test-Time Evolution of Reward Programs for Humanoid Loco-Manipulation（arXiv:2610.02196）

- **论文：** [arXiv:2610.02196](https://arxiv.org/abs/2610.02196)
- **问题：** 预训练控制器面对新任务时，现有技能难以由固定奖励和规划程序自动组合
- **方法线索：** 通过物体感知行为基础模型把新奖励转化为动作；大模型修改分阶段奖励程序，数值优化器 调整奖励常数，并根据执行结果持续迭代
- **详情节点：** [InterEvolve：人形移动操作奖励程序的测试时演化](../../wiki/entities/paper-interevolve.md)

### Filter-Aware Fine-Tuning for Safe Humanoid Whole-Body Tracking（arXiv:2610.02341）

- **论文：** [arXiv:2610.02341](https://arxiv.org/abs/2610.02341)
- **问题：** 运行时安全过滤器会修改动作并改变状态分布，与原跟踪策略形成动力学、目标和信息 不匹配
- **方法线索：** CoFiT在安全过滤器启用状态下微调跟踪策略，并输入约束信息和干预历史，使策略主动减少 过滤器修正
- **详情节点：** [面向安全人形全身跟踪的过滤器感知微调](../../wiki/entities/paper-filter-aware-fine-tuning-for.md)

### Beyond Reward Hacking: Proxy Divergence Across Four Layers of a Staged Humanoid Learning Pipeline（arXiv:2610.03196）

- **论文：** [arXiv:2610.03196](https://arxiv.org/abs/2610.03196)
- **问题：** 人形强化学习的失败不仅来自奖励设计，还可能来自课程门控、评估指标和参考动作
- **方法线索：** 统一分析四类代理目标的偏离条件，并提出L1代价、峰值门控、异步相位评估和可行性优先 参考等修正方法
- **详情节点：** [超越奖励投机：分阶段人形学习流程四层代理目标偏离](../../wiki/entities/paper-beyond-reward-hacking.md)

### KungfuAthleteBot: Learning High-Dynamic Humanoid Motion from Video with Unified Robust Recovery（arXiv:2610.03388）

- **论文：** [arXiv:2610.03388](https://arxiv.org/abs/2610.03388)
- **问题：** 视频动作存在漂浮、穿地和抖动，没有执行力信息，也不包含失败恢复示范
- **方法线索：** 构建武术动作数据集，以物理引导的抛物线修正处理腾空和落地轨迹，并把动作跟踪与失败 恢复统一到同一控制框架。
- **详情节点：** [KungfuAthleteBot：从视频学习具备统一鲁棒恢复能力的高动态人形动作](../../wiki/entities/paper-kungfuathlete-humanoid-martial-arts-tracking.md)

### Proprioceptive Force Estimation for Quadruped Locomotion and Human-Robot Interaction（arXiv:2609.34222）

- **论文：** [arXiv:2609.34222](https://arxiv.org/abs/2609.34222)
- **问题：** 负载力需要被策略补偿，而牵引力又可作为人类给机器人的运动指令
- **方法线索：** 从本体历史联合估计三轴外力、速度和latent环境信息，同一估计结果用于负载补偿和牵引导 航
- **详情节点：** [面向四足运动与人机交互的本体力估计](../../wiki/entities/paper-proprioceptive-force-estimation-for-quadruped.md)

### Terrain-Aware Autonomous Planetary Exploration for Exteroceptive- Proprioceptive Mapping with Quadruped Scouts（arXiv:2609.35493）

- **论文：** [arXiv:2609.35493](https://arxiv.org/abs/2609.35493)
- **问题：** 几何地图无法反映月面类地形实际引起的滑移、稳定性下降和能耗变化
- **方法线索：** 把RGB-D高程、几何可通行性和本体交互指标写入多层地图，并据此选择探索目标
- **详情节点：** [基于四足侦察机器人的外感知—本体感知地形自主行星探索](../../wiki/entities/paper-terrain-aware-autonomous-planetary-exploration.md)

### ChronoSRL: Temporal Geometry for Self-Supervised Reinforcement Learning（arXiv:2609.36238）

- **论文：** [arXiv:2609.36238](https://arxiv.org/abs/2609.36238)
- **问题：** 空间上相近的目标可能因障碍和机器人能力而需要很长时间到达
- **方法线索：** 让状态—动作与目标的latent距离直接表示预计到达时间，并同时建模到达概率和停留时间
- **详情节点：** [ChronoSRL：自监督强化学习的时间几何表征](../../wiki/entities/paper-chronosrl.md)

### Inferring Soil Friction Angle from Robot Foot-Ground Force Histories: A Bayesian Inverse Approach to Proprioceptive Soil Sensing（arXiv:2609.36582）

- **论文：** [arXiv:2609.36582](https://arxiv.org/abs/2609.36582)
- **问题：** 四足机器人在沙土环境中需要估计土壤强度，但专用测量设备会增加负担
- **方法线索：** 用物质点法生成腿—土壤交互样本，再通过高斯过程代理模型和贝叶斯反演估计土壤内摩擦 角
- **详情节点：** [从机器人足—地力历史推断土壤内摩擦角](../../wiki/entities/paper-inferring-soil-friction-angle-from-robot-foot-ground-force-histories.md)

### Predictive Safety Curricula for Robust Legged Locomotion（arXiv:2609.37070）

- **论文：** [arXiv:2609.37070](https://arxiv.org/abs/2609.37070)
- **问题：** 平均性能优秀的运动策略仍可能存在低概率碰撞，普通课程无法主动增加稀有危险经 验
- **方法线索：** 使用安全Critic预测未来安全代价，据此重采样地形和历史随机事件；ANYmal-D胫部碰撞率 降低63%
- **详情节点：** [面向鲁棒腿式运动的预测式安全课程学习](../../wiki/entities/paper-predictive-safety-curricula-for-robust.md)

### Experience-Driven Continual Learning of Terrain Traversability for Quadruped Robots（arXiv:2609.39755）

- **论文：** [arXiv:2609.39755](https://arxiv.org/abs/2609.39755)
- **问题：** 地形外观无法直接判断滑移、足端冲击和能耗，持续更新模型又容易遗忘旧地形
- **方法线索：** 从接触前图像预测五类本体交互指标，并通过证据回归、经验回放和验证门控控制模型更 新
- **详情节点：** [基于经验驱动持续学习的四足地形可通行性预测](../../wiki/entities/paper-experience-driven-continual-learning-of.md)

### Experience-Based Feasibility-Aware Generative Adversarial Imitation from Observation under Embodiment Mismatch（arXiv:2610.01171）

- **论文：** [arXiv:2610.01171](https://arxiv.org/abs/2610.01171)
- **问题：** 人体状态轨迹可能超出四足机器人的动力学能力，直接模仿会让不可行动作误导策略
- **方法线索：** 从机器人当前经验中估计示范转移的可行性，并随策略能力提升动态扩大可行区域
- **详情节点：** [本体不匹配条件下基于经验可行性感知的生成式对抗观察模仿](../../wiki/entities/paper-experience-based-feasibility-aware-generative.md)

### PROMO: Preference-Conditioned Multi-Objective Reinforcement Learning for Quadrupedal Robots（arXiv:2610.01260）

- **论文：** [arXiv:2610.01260](https://arxiv.org/abs/2610.01260)
- **问题：** 速度跟踪、稳定性和能耗的权衡在训练后被固定，部署时无法切换
- **方法线索：** 把语义化目标偏好作为策略输入，使单一策略在运行时连续调整跟踪、稳定和能耗目标
- **详情节点：** [PROMO：偏好条件化的四足机器人多目标强化学习](../../wiki/entities/paper-promo.md)

### ReCo: Response-Consistent Locomotion with Policy-Aware MPC for Legged Manipulation（arXiv:2610.01612）

- **论文：** [arXiv:2610.01612](https://arxiv.org/abs/2610.01612)
- **问题：** RL运动策略对速度指令的响应随步态相位、接触和负载变化，MPC难以准确预测
- **方法线索：** 训练具有一致命令响应的运动策略，再辨识其闭环响应模型，供MPC联合规划底盘和机械臂 动作
- **详情节点：** [ReCo：采用策略感知MPC的响应一致腿式移动操作](../../wiki/entities/paper-reco.md)

### Bidirectional Voronoi-Biased Exploration Curriculum for Reinforcement Learning（arXiv:2610.03395）

- **论文：** [arXiv:2610.03395](https://arxiv.org/abs/2610.03395)
- **问题：** 四足爬箱等长时稀疏奖励任务很难从初始状态探索到目标
- **方法线索：** 分别从目标向外扩展起始状态、从初始分布向外扩展目标，两端同时生长并训练同一目标条 件策略
- **详情节点：** [面向强化学习的双向Voronoi偏置探索课程](../../wiki/entities/paper-bidirectional-voronoi-biased-exploration-curriculum.md)

### CrowdOcc: Monocular Semantic Scene Completion for Quadruped Robots in Crowded Indoor Environments（arXiv:2610.03031）

- **论文：** [arXiv:2610.03031](https://arxiv.org/abs/2610.03031)
- **问题：** 人群遮挡会破坏室内静态几何重建，并导致人体占据位置不完整或错误
- **方法线索：** 发布2.51万帧四足机器人室内数据集，融合表面法向几何与稀疏人—人、人—场景交互完成三 维语义占据预测
- **详情节点：** [CrowdOcc：面向拥挤室内环境四足机器人的单目语义场景补全](../../wiki/entities/paper-crowdocc.md)

### TACET: Context-Appropriate Acoustic-Social Navigation for Quadrupeds（arXiv:2610.03828）

- **论文：** [arXiv:2610.03828](https://arxiv.org/abs/2610.03828)
- **问题：** 医院和办公室中的四足机器人不仅要避让人员，还要根据场景控制行走噪声
- **方法线索：** 视觉语言模型输出包含步态、速度和社交代价的行为token，同时控制导航代价图和低噪声运 动策略
- **详情节点：** [TACET：面向四足机器人的情境适配声学—社会导航](../../wiki/entities/paper-tacet.md)

### Around the World: Unified Learned Locomotion on a 270 g Continuous-Rotation Quadruped（arXiv:2610.02728）

- **论文：** [arXiv:2610.02728](https://arxiv.org/abs/2610.02728)
- **问题：** 亚千克四足平台上的闭环学习式运动仍较少，连续旋转腿还具有普通关节腿没有的姿态 空间
- **方法线索：** 采用连续关节环面参考和重力条件变换，以单个小型策略统一完成正立行走、倒立行走和落 地恢复。
- **详情节点：** [环行世界：270克连续旋转四足机器人的统一学习式运动](../../wiki/entities/paper-around-the-world.md)

### LocoWM: High-Precision Locomotion through World-Model-Guided Residual Adaptation（arXiv:2609.39179）

- **论文：** [arXiv:2609.39179](https://arxiv.org/abs/2609.39179)
- **问题：** 普通残差控制只能在误差已经出现后进行修正，无法提前补偿未来偏差
- **方法线索：** 冻结基础策略后，用动作条件世界模型预测未来物理状态，再由残差适配器根据预测序列提 前修正动作；在Go2-W轮腿机器人上验证地形调平、加速度补偿和抗扰恢复
- **详情节点：** [LocoWM：基于世界模型引导残差适配的高精度运动控制](../../wiki/entities/paper-locowm.md)

### Multifunctional Locomotion Control of Multi-Jointed BURs with Swimming and Gait Capabilities（arXiv:2609.37086）

- **论文：** [arXiv:2609.37086](https://arxiv.org/abs/2609.37086)
- **问题：** 兼具鳍游和腿式步行能力的机器人通常依靠人工状态机切换
- **方法线索：** 设计四条四轴腿鳍，并通过势函数融合多模态传感信息，使同一控制器在三种游泳和步行行 为之间连续转换
- **详情节点：** [具备游泳与步行能力的多关节仿生水下机器人多功能运动控制](../../wiki/entities/paper-multifunctional-locomotion-control-of-multi.md)

### Terrain-Dependent Intra-Cycle Leg Timing for Effective Locomotion on Granular Slopes（arXiv:2610.04144）

- **论文：** [arXiv:2610.04144](https://arxiv.org/abs/2610.04144)
- **问题：** 六足机器人在颗粒斜坡上容易下滑，但不同坡度所需的周期内腿部速度分配尚不明确
- **方法线索：** 调整Buehler时钟慢速相位的位置，使推进剪切力与足够的地形法向支撑在时间上重合，并用 颗粒阻力模型解释最佳时序随坡度变化的原因。 ·
- **详情节点：** [面向颗粒斜坡有效运动的地形依赖周期内腿部时序](../../wiki/entities/paper-terrain-dependent-intra-cycle-leg.md)
