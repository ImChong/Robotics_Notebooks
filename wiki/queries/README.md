# wiki/queries/

## 这个目录是什么

`wiki/queries/` 存放 **Query 操作的独立产物**。

根据 Karpathy LLM Wiki 模式：

> "Answers can become new wiki pages, enabling knowledge to compound."

当一次 Query 操作产生了跨多个 wiki 页面的综合分析、对比洞见，或者某个具有独立认知价值的结论，这个产物应写回 wiki 而不是留在聊天记录里。

---

## 什么样的内容放在这里

```mermaid
flowchart LR
  Q["新问题 / 工程决策提问"] --> S["检索 wiki 与 sources 现有页面"]
  S --> M["跨 2+ 页面精读与综合分析"]
  M --> W["沉淀为 wiki/queries/ 独立页面"]
  W --> R["顶部注明触发来源 + 关联页面 + 参考来源"]
```

**放这里的条件（满足任一）：**
- 精读了 2 个以上 wiki 页面，综合出新的连接或对比
- 问题本身有较高的通用性，其他人将来也会问到
- 结论不适合直接归入某个已有类别（concepts/methods/tasks/comparisons）

**不放这里的内容：**
- 只是某个概念的补充 → 直接更新对应的 `wiki/concepts/` 或 `wiki/methods/` 页面
- 系统性的方法对比 → 放 `wiki/comparisons/`
- 新的概念 → 放 `wiki/concepts/`

---

## 页面格式要求

每个 query 产物页面，必须在顶部注明触发来源：

```markdown
> **Query 产物**：本页由以下问题触发：「<问题一句话>」
> 综合来源：[页面A](../concepts/xxx.md)、[页面B](../methods/yyy.md)
```

然后按照正常 wiki 页面标准写内容，包括：
- 核心结论
- 关联页面
- 参考来源

---

## 当前页面列表

（每次新增 query 产物后，在此更新）

| 文件 | 触发问题 | 综合来源 |
|------|---------|---------|
| [mpc-wbc-integration](../concepts/mpc-wbc-integration.md) | MPC 和 WBC 在人形机器人 locomotion 里是怎么配合工作的？ | MPC、WBC、Locomotion、Optimal Control |
| [rl-algorithm-selection](./rl-algorithm-selection.md) | 在足式/人形机器人里，PPO / SAC / TD3 怎么选？ | RL、Policy Optimization、Locomotion、Sim2Real |
| [sim2real-checklist](./sim2real-checklist.md) | 从仿真到真机部署，有哪些必须检查的工程事项？（含快速部署检查） | Sim2Real、Domain Randomization、SysID、Privileged Training |
| [control-architecture-comparison](./control-architecture-comparison.md) | 人形机器人的主流控制架构有哪些，各有什么优劣？ | WBC vs RL、MPC-WBC、RL、IL、TSID |
| [humanoid-hardware-selection](./humanoid-hardware-selection.md) | 做人形机器人运动控制研究，该选哪个硬件平台？ | Locomotion、Sim2Real、Loco-Manipulation |
| [g1-video-dance-gvhmr-gmr-beyondmimic-multi-motion](./g1-video-dance-gvhmr-gmr-beyondmimic-multi-motion.md) | 想把短视频里的舞蹈搬到 Unitree G1，从视频到追踪策略每一步该用什么、命令是什么、哪里容易踩坑？ | GVHMR、GMR、BeyondMimic、unitree_rl_mjlab、mjlab |
| [wbc-implementation-guide](./wbc-implementation-guide.md) | 如何从零搭建一个 WBC 控制器？ | WBC、TSID、HQP、Centroidal Dynamics、Contact Estimation |
| [locomotion-reward-design-guide](./locomotion-reward-design-guide.md) | 怎么设计 locomotion RL 的奖励函数？ | RL、Locomotion、Reward Design、Curriculum Learning |
| [humanoid-rl-cookbook](./humanoid-rl-cookbook.md) | 从零训练人形机器人 RL 策略的完整 checklist？ | RL、Sim2Real、Privileged Training、Curriculum、Deployment |
| [pinocchio-quick-start](./pinocchio-quick-start.md) | 用 Pinocchio 做机器人动力学计算的最小可运行示例？ | Pinocchio、WBC、Kinematics、Dynamics |
| [urdf-link-inertia-real-robot-check](./urdf-link-inertia-real-robot-check.md) | URDF 中的机器人连杆惯量如何与真机对比检查？ | URDF、连杆 vs 转子惯量、SysID、重力补偿、Pinocchio |
| [mpc-solver-selection](./mpc-solver-selection.md) | 机器人 MPC 求解器怎么选：OSQP vs qpOASES vs Acados vs FORCES Pro？ | MPC、QP Solver、Optimization、Acados |
| [reward-design-guide](./reward-design-guide.md) | 从零设计 locomotion RL 的 reward 函数？核心原则和常见陷阱？ | Reward、Curriculum、Locomotion、PPO |
| [sim2real-gap-reduction](./sim2real-gap-reduction.md) | sim2real transfer 失败的根因分类与对应缩减策略？ | Sim2Real、DR、ActuatorNet、Privileged Training |
| [hardware-comparison](./hardware-comparison.md) | 主流人形机器人平台在硬件能力上有何差异？如何根据任务选择平台？ | Humanoid、Actuator、Locomotion、WBC vs RL |
| [rl-hyperparameter-guide](./rl-hyperparameter-guide.md) | 训练腿式机器人 locomotion 策略时，PPO/SAC 的关键超参数如何调节？ | RL、PPO、SAC、Locomotion、Reward Design |
| [when-to-use-wbc-vs-rl](./when-to-use-wbc-vs-rl.md) | 面对具体机器人控制任务，应该选择 WBC、RL，还是两者结合？ | WBC、RL、Locomotion、Decision、Architecture |
| [il-for-manipulation](./il-for-manipulation.md) | 做机器人操作用模仿学习还是 RL？怎么收集数据？ | Manipulation、Imitation Learning、Behavior Cloning、DAgger、RL |
| [vla-deployment-guide](./vla-deployment-guide.md) | 如何在真机上部署 VLA 策略？推理延迟怎么控制？ | VLA、Foundation Policy、Manipulation、Loco-Manipulation、Deployment |
| [foundation-policy-for-humanoids](./foundation-policy-for-humanoids.md) | 人形机器人 foundation policy 现在适合什么，不适合什么？ | Foundation Policy、VLA、Loco-Manipulation、Manipulation、Locomotion |
| [open-source-motion-control-projects](./open-source-motion-control-projects.md) | 飞书公开文档《开源运动控制项目》里抽出了哪些可复用的方法模式？ | RL、IL、Model-Based RL、Locomotion、Manipulation、Motion Retargeting |
| [hmi-opensource-projects-coverage](./hmi-opensource-projects-coverage.md) | 开源项目主表里的项目，在本库分别对应哪一页、该怎么读？（166 项快照；上游已 586） | Humanoid Motion Intelligence、开源运动控制项目摘要、上游开源主表 |
| [china-domestic-opensource-424-coverage](./china-domestic-opensource-424-coverage.md) | 国内 76 家机构 424 个开源项目，在本库分别对应哪一页？ | 国内开源全景技术地图、Humanoid Motion Intelligence |
| [hmi-papers-coverage](./hmi-papers-coverage.md) | 论文与项目总索引里的 191 篇论文，在本库是否都有独立详情节点？ | Humanoid Motion Intelligence、上游论文总索引、开源主表导读 |
| [robot-policy-debug-playbook](./robot-policy-debug-playbook.md) | RL 策略在仿真中好但真机差，如何系统排查？ | Sim2Real、Domain Randomization、Privileged Training、Deployment |
| [simulator-selection-guide](./simulator-selection-guide.md) | MuJoCo vs Isaac Lab vs Genesis，做 locomotion RL 选哪个？ | Locomotion、Sim2Real、RL、Humanoid Robot |
| [demo-data-collection-guide](./demo-data-collection-guide.md) | 用模仿学习做操作，怎么高效收集人类演示数据？ | Teleoperation、Imitation Learning、Behavior Cloning、Bimanual Manipulation |
| [ppo-vs-sac-for-robots](./ppo-vs-sac-for-robots.md) | 机器人 RL 用 PPO 还是 SAC？有什么实践区别？ | Policy Optimization、RL、Locomotion、GAE |
| [grasp-policy-selection](./grasp-policy-selection.md) | 真机抓取系统该用几何启发式 / 检测式 grasp pose / 还是端到端策略？开放场景 vs 已知物体怎么选？ | Grasp Pose Estimation、AnyGrasp、ContactNet、Manipulation、Visual Servoing、Contact-Rich Manipulation |
| [humanoid-motion-tracking-method-selection](./humanoid-motion-tracking-method-selection.md) | 人形运动跟踪与风格先验方法怎么选、怎么组合？ | DeepMimic、BeyondMimic、AMP/ADD/SMP、Any2Track、AMS、GentleHumanoid、ASE、GenMo |
| [manipulation-vla-architecture-selection](./manipulation-vla-architecture-selection.md) | 灵巧操作里 VLA / 视频-动作 / 世界模型 / 开源策略怎么选？ | VLA、mimic-video、DeFI、DWM、CLAW、π₀、π0.7、STAR-VLA、Pelican |
| [humanoid-contact-character-control-guide](./humanoid-contact-character-control-guide.md) | 接触丰富人形控制与角色化表演路线怎么选？ | GentleHumanoid、HTTD、Being-H07、Disney OLAF、HiPAN、Zest、EFGCL |
| [dexterous-manipulation-data-pipeline](./dexterous-manipulation-data-pipeline.md) | 灵巧操作的数据标注、手部感知、GAE 与执行器建模怎么配？ | Auto-labeling、WiLoR、GAE、Actuator Network、Tactile Impedance |
| [object-detection-model-selection](./object-detection-model-selection.md) | 机器人感知栈选单阶段还是两阶段检测器？机载实时 vs 服务器侧怎么选？ | Object Detection、Vision Backbones、YOLO、ResNet、Manipulation、Humanoid Soccer |
| [perception-backbone-selection](./perception-backbone-selection.md) | 机器人感知栈该用分类骨干 / 检测头 / 还是通用预训练表征？选型第一刀砍在哪？ | Vision Backbones、Visual Representation for Policy、CNN vs ViT、Object Detection、VLA、DINOv2/R3M/VC-1 |
| [real-robot-policy-autoresearch-harness](./real-robot-policy-autoresearch-harness.md) | 如何把真机策略开发改造成 coding agent 可编排的自改进闭环？前提、范式选型与 scaling 怎么读？ | ENPIRE、Behavior Cloning、Reinforcement Learning、Simulation-Evaluation Infrastructure、Data Flywheel、Embodied Scaling Laws |
| [numerical-optimization-method-selection](./numerical-optimization-method-selection.md) | 机器人数值优化方法（LQR/iLQR、拟牛顿、罚/障碍/增广拉格朗日、凸松弛）该按什么标准选、怎么分层组合？ | LQR/iLQR、Quasi-Newton BFGS、Penalty/Barrier/Augmented Lagrangian、Convex Relaxation、Optimization Software Selection |
| [go2-3d-semantic-mapping-sam-pipeline](./go2-3d-semantic-mapping-sam-pipeline.md) | GO2 三维语义建图有哪些资料？运动中点云差怎么排？SAM 如何落到 3D？CMU 相关工作如何区分？ | point_lio_unilidar、FAST-LIO、LIO-SAM、autonomy_stack_go2、DualMap、OVO、OV-SAM3D、FindAnything、CMU MSCV Semantic 3D Mapping |
| [slam-second-spring-embodied](./slam-second-spring-embodied.md) | 具身时代 SLAM 的精华与糟粕是什么？VLA naive scaling 能否当两年终局？ | VLA、状态估计枢纽、LIO/VIO 选型、具身 Infra 2026 全景 |
| [embodied-six-routes-holes](./embodied-six-routes-holes.md) | 六条主流路线各自卡在哪？还该不该用「端到端 vs 模块化」当第一刀？ | 五大范式、五层模型族、VLA、RL、SmoothRL、Gemini ER、Figure |

**对比页（comparisons/）补充**：AMP 先验变体见 [amp-add-smp-motion-prior-variants](../comparisons/amp-add-smp-motion-prior-variants.md)。

---

## 和 wiki/concepts/mpc-wbc-integration.md 的说明

该页面是早期的 Query 产物（MPC 与 WBC 如何协同工作），但创建时 queries/ 目录尚未存在，因此留在了 `wiki/concepts/`。
后续新的 query 产物统一放入本目录。
| [actuator-drive-chain-selection-loop](./actuator-drive-chain-selection-loop.md) | 策略（RL/MPC）算出来的关节力矩指令，最终由什么样的电子硬件驱动链落地——从画驱动板、写 FOC 固件、给执行器建模辨识摩擦，到用实时总线把它们闭环起来，中间分几层、每层选什么工具/方案、数据手册标称参数和真机实测曲线为什么对不上、执行器建到多真才够、总线跑得快是不是就等于闭环带宽高？ | KiCad（开源 EDA）、Altium Designer（商用 EDA）、SimpleFOC（开源 FOC 驱动固件）、NeuralActuator（神经执行器建模）、BAM（执行器摩擦辨识）、Implicit/Explicit 执行器建模、EtherCAT 协议基础。它是具身大模型分类学选型闭环（选哪一类策略）与具身大模型评测基准选型闭环（怎么证明它）的**硬件侧姊妹链**——回答「策略的力矩指令能不能被真实驱动链忠实执行」。 |
| [clf-cbf-in-wbc](./clf-cbf-in-wbc.md) | CLF 和 CBF 如何在 WBC/MPC 中联合使用保证稳定性与安全性？ | Control Lyapunov Function、Control Barrier Function、Whole-Body Control、CLF vs CBF 对比 |
| [contact-rich-manipulation-guide](./contact-rich-manipulation-guide.md) | 做接触丰富的操作任务（装配/拧螺丝），有哪些实践要点？ | Manipulation、Contact-Rich Manipulation、Bimanual Manipulation |
| [contact-wrench-closed-loop](./contact-wrench-closed-loop.md) | 我要做接触丰富的操作，从『手碰到东西』到『稳定用力完成任务』中间到底有哪几层，每一层各自解决什么、怎么互相约束、哪一层做不好会怎样失败？ | Contact Estimation、Force Control Basics、Hybrid Force-Position Control、Impedance Control、Visuo-Tactile Fusion。 |
| [cross-embodiment-transfer-strategy](./cross-embodiment-transfer-strategy.md) | 同一段人形运动控制能力要搬到一台新机器人，单具身重训、Any2Any 高效后训练、多具身联合训练这三条路怎么选、各踩什么坑？ | Whole-Body Tracking Pipeline、Any2Any、SONIC、Motion Retargeting Pipeline、Behavior Foundation Model |
| [dexterous-data-collection-guide](./dexterous-data-collection-guide.md) | 如何采集灵巧手操作的专家数据？有哪些主流的遥操作方案？ | Allegro Hand、Demo Data Collection |
| [domain-randomization-guide](./domain-randomization-guide.md) | 做 sim2real，如何设计 domain randomization 的参数与分布范围？ | Domain Randomization、Sim2Real、Sim2Real 工程 Checklist |
| [embodied-data-collection-five-routes-landscape](./embodied-data-collection-five-routes-landscape.md) | 具身智能数据采集有哪些技术路线？各路线装备分支与代表案例是什么？ | 见页内「关联页面」与「参考来源」 |
| [embodied-eval-benchmark-selection-loop](./embodied-eval-benchmark-selection-loop.md) | 我训了一个具身大模型，接下来到底怎么『测/证明它』——从『大脑看懂没』到『真机能不能做成』中间分几层评测、每层测什么、用哪个代表性基准、指标要复现性还是真实代表性、看过程指标还是结果指标、哪一层评测结论最容易骗人？ | RoboBench（MLLM 具身大脑评测）、ESI-Bench（具身空间智能）、Daily-Omni（日常 AV 时序对齐）、EWMBench（世界模型生成评测）、WorldScore（开放域世界生成）、HarnessEval-W（交互式世界 agentic 评测）、GigaWorld-1（策略评估器）、仿真评测基础设施。它是具身大模型分类学选型闭环（选哪一类模型）的姊妹链——回答「选完之后怎么评测/证明它」。 |
| [embodied-fm-taxonomy-loop](./embodied-fm-taxonomy-loop.md) | 我要给一个具身系统选模型，从『看懂场景』到『在真机上稳定完成任务』中间到底分几层——VLM / VLN / VLA / VLX / 世界模型各自解决什么、各家族的 I/O 边界与数据需求/泛化能力/实时性/闭环稳定性怎么互相约束、哪一层选错会怎样失败？ | 五大具身模型分类对比、VLA 方法页、VLN 任务页、World Action Models、生成式世界模型、统一多模态 token。 |
| [embodied-six-routes-survey-lineage](./embodied-six-routes-survey-lineage.md) | 模块化、技能编排、IL、RL、VLA、世界模型六条路线在吵什么？有没有一条按综述补课的阅读顺序？ | 五大范式、六条路线的窟窿、VLA、生成式世界模型、深蓝 2026-09-20 公众号归档。 |
| [ethercat-master-optimization](./ethercat-master-optimization.md) | 为什么我的 EtherCAT 主站跑不到 1kHz？丢包和抖动怎么处理？分布式时钟 (DC) 怎么配置？ | Real-time Middleware、Humanoid Know-how |
| [field-robotics-troubleshooting](./field-robotics-troubleshooting.md) | 机器人在实验室跑得很稳，一带到户外草地或碎石路就各种摔，传感器也频繁报错，怎么排查？ | Know-how、System ID |
| [hardware-abstraction-layer](./hardware-abstraction-layer.md) | 如何写一套代码，既能跑在仿真里，又能跑在不同品牌的机器人硬件上？硬件抽象层该怎么分层？ | 运动控制主路线、Humanoid Know-how |
| [humanoid-battery-thermal-management](./humanoid-battery-thermal-management.md) | 人形机器人在做高动态动作时电池掉电极快，且关节电机经常过热停机，怎么排查和优化？ | Know-how、Humanoid Hardware |
| [humanoid-joint-motor-topology-selection](./humanoid-joint-motor-topology-selection.md) | 无框力矩、空心杯、轴向磁通，谁才是人形机器人关节电机的首选？ | 见页内「关联页面」与「参考来源」 |
| [humanoid-robot-data-collection-landscape](./humanoid-robot-data-collection-landscape.md) | 人形机器人训练数据怎么采？Substack 里提到的项目/论文各自是什么？ | LeoInAI Substack、Scanford 论文摘录 |
| [humanoid-soccer-skill-learning-method-selection](./humanoid-soccer-skill-learning-method-selection.md) | 人形机器人踢球/射门技能学习，工程上该走 PAiD 式感知–动作渐进融合，还是 RoboNaldo 式 motion scaffold + 课程 RL，怎么选、怎么组合？ | PAiD Framework、RoboNaldo、Humanoid Soccer |
| [humanoid-training-data-pipeline](./humanoid-training-data-pipeline.md) | 我要训练一个人形全身策略，从**原始动作捕捉 / 人体视频**到能喂进 **RL/IL** 的训练输入，参考运动来源、重定向方案、训练范式三层各该怎么选？ | 人形参考运动与操作数据集选型、Motion Data Quality、Motion Retargeting |
| [humanoid-voice-interaction-pipeline](./humanoid-voice-interaction-pipeline.md) | 人形机器人的语音交互要做成可打断的闭环，工程上从唤醒/ASR、LLM 理解规划、技能或 VLN 执行到 TTS 反馈，每一环该怎么选型、会踩哪些坑、出问题先查什么？ | 见页内「关联页面」与「参考来源」 |
| [legged-humanoid-rl-pd-gain-setting](./legged-humanoid-rl-pd-gain-setting.md) | 人形或腿足式机器人强化学习里，底层关节 Kp/Kd（或仿真里的 stiffness/damping）一般怎么设、和什么量绑定？ | legged_gym、Sim2Real、Sim2Real Checklist、Domain Randomization 指南、开源实现索引、RL+PD 接口论文索引 |
| [lightnav-0-blog-references-deep-read](./lightnav-0-blog-references-deep-read.md) | 深读 [LightNav-0 Tech Blog](https://www.lightorigins.com/en/blog/lightnav-0) 引用的所有相关项目和论文。 | lightorigins-3blogs-technology-map、paper-lightnav-0、spatial-reasoning-benchmarks-technology-map 及博客正文 ingest 节点。 |
| [locomotion-failure-modes](./locomotion-failure-modes.md) | locomotion RL 训练时常见的失败模式有哪些？怎么诊断？ | Reward Shaping Guide、Robot Policy Debug Playbook、Legged Gym |
| [motion-retargeting-three-routes-landscape](./motion-retargeting-three-routes-landscape.md) | 运动重定向有哪几条技术路线？深蓝 ICRA 2026 长文里每篇论文/项目分别是什么？ | 深蓝公众号归档 |
| [mpc-tuning-guide](./mpc-tuning-guide.md) | 我的线性 MPC 跟踪很不稳，Base 晃动严重，应该先调哪个参数？Q 和 R 怎么给值？ | MPC 核心页、Optimal Control |
| [multimodal-fusion-tricks](./multimodal-fusion-tricks.md) | 我的机器人有视觉、触觉和 IMU，怎么把这些数据喂给神经网络效果最好？模态间的权重怎么分配？ | Cross-modal Attention、Tactile Sensing |
| [optimization-software-selection](./optimization-software-selection.md) | 机器人优化软件怎么选：QP / NLP / 锥规划求解器与建模框架（CasADi、Drake、Acados、OSQP 等）对照。 | 见页内「关联页面」与「参考来源」 |
| [real-time-control-middleware-guide](./real-time-control-middleware-guide.md) | 为什么我的机器人策略在仿真里很完美，上了真机却经常疯狂抽搐？Linux 实时性怎么配置？ | ROS 2 vs LCM 对比、Sim2Real 工程 Checklist |
| [reward-shaping-guide](./reward-shaping-guide.md) | 训练 locomotion RL 策略时，奖励函数怎么设计？有哪些调教技巧？ | Reward Design、Humanoid RL Cookbook、Legged Gym |
| [rl-sim2real-deployment-conditions](./rl-sim2real-deployment-conditions.md) | 强化学习真机部署时，机器人在仿真里究竟学会了什么？为什么维度对齐仍会翻车？ | Sim2Real、RL 策略真机调试 Playbook、Sim2Real 闭环误差分层工程、RMA；叙事骨架编译自 深蓝具身智能 2026-09-18 公众号文。 |
| [robot-learning-three-eras-narrative](./robot-learning-three-eras-narrative.md) | 围绕 Ted Xiao 访谈编译稿中出现的机器人学习话题，能否用一手文献串联成可维护的知识索引？ | Foundation Policy、VLA、Imitation Learning、Reinforcement Learning、Embodied Scaling Laws、以及 `ted_xiao_embodied_three_eras_primary_refs.md` 中的论文指针。 |
| [robot-perception-stack-selection-loop](./robot-perception-stack-selection-loop.md) | 机器人策略/操作/导航要消费的视觉感知信号，到底从哪来、怎么一层层选出来——从选传感与标定（RGB / RGB-D / LiDAR、内外参标定、深度精度 vs 成本），到 2D 检测/分割选型（单阶段 YOLO vs 两阶段 vs DETR、闭集 vs 开放词汇、实时机载 vs 服务器侧），再到 2D→3D 提升与语义建图（深度融合、点云语义、在线 vs 离线、稠密 vs 稀疏），最后到下游策略怎么消费感知输出（坐标后处理、感知频率 vs 控制频率对齐）？中间分几层、每层选什么、精度和时延/算力怎么取舍、什么时候 2D 框就够、什么时候非上 3D 语义几何不可？ | Ultralytics YOLO（单阶段实时检测）、RF-DETR（端到端 DETR）、YOLO 奠基论文、Segment Anything（可提示分割）、SAM2（视频可提示分割）、FindAnything（对象级开放词汇 3D 语义建图）、CMU MSCV 语义 3D 建图、OV-SAM3D（开放词汇 3D 分割）。它位于策略的**输入端**，是执行器驱动链选型闭环（策略**输出端**「力矩指令怎么被硬件执行」）的**镜像姊妹链**，也与具身大模型分类学选型闭环（选哪一类策略）、具身大模型评测基准选型闭环（怎么证明它）互补——回答「策略要消费的感知信号从哪来、怎么选这条感知栈」。 |
| [rsi-four-tier-five-pushes](./rsi-four-tier-five-pushes.md) | RSI 从 EURISKO 到 GPT-5.6 RSI Index 究竟走到了哪一层？会自我改进是否等于智能爆炸？ | 递归自改进、Awesome RSI、RSI Survey 实体、AI Auto-Research、Motus2；叙事骨架编译自 Datawhale 2026 RSI 科普综述（赵志民）；机制 taxonomy 与验证层级对齐 arXiv:2607.07663。 |
| [sim2real-closed-loop-engineering](./sim2real-closed-loop-engineering.md) | 为什么说 Sim2Real 不是训完之后才考虑的事？工程上如何从系统辨识做到在线适应与安全部署？ | Sim2Real、System Identification、Domain Randomization、Privileged Training、RMA、Sim2Real Checklist、Gap 缩减指南；叙述骨架编译自 深蓝具身智能公众号文（2026-07-28）。 |
| [simulation-physics-fidelity](./simulation-physics-fidelity.md) | 我要在仿真里训一个腿式/操作策略，**从几何/URDF 精度到执行器模型这几层物理保真度，分别该投到什么程度、各自补的是哪一种 sim2real gap、代价是什么？** | Physics Fidelity ↔ Sim2Real Gap、Sim2Real、Contact Dynamics |
| [soccer-visual-field-localization-pipeline](./soccer-visual-field-localization-pipeline.md) | 足球机器人只有相机，如何从场地白线一路做到抗抖的 (x,y,θ) 位姿？检测、线匹配、EKF 三步怎么串、各环怎么选型、坑在哪？ | 场线/交点检测（课程 6.4）、线匹配定位（课程 7.2）、线特征 EKF 融合（课程 7.3）。 |
| [table-tennis-hierarchical-skill-learning-guide](./table-tennis-hierarchical-skill-learning-guide.md) | 物理乒乓球动画里技能 mode collapse 严重、又要支持双智能体对打与 VR 人–机，该走单 ASE 还是分层专家 + 策略？ | Table Tennis Strategy & Skill Learning、SMPLOlympics、ASE |
| [tactile-feedback-in-rl](./tactile-feedback-in-rl.md) | 在强化学习抓取和操作任务中，如何有效地加入触觉反馈？怎么解决视触觉多模态融合以及 Sim2Real 差距的问题？ | Tactile Sensing、Contact-Rich Manipulation |
| [vla-with-low-level-controller](./vla-with-low-level-controller.md) | VLA 如何与低级关节控制器（MPC/WBC）融合？有哪些架构？ | VLA、Whole-Body Control、MPC、VLA 真机部署指南 |
| [wbc-tuning-guide](./wbc-tuning-guide.md) | WBC QP 求解器怎么调参？权重矩阵、约束松弛、热启动有哪些技巧？ | Whole-Body Control、TSID、WBC Implementation Guide |
| [wheeled-dual-arm-humanoid-urdf-workflow](./wheeled-dual-arm-humanoid-urdf-workflow.md) | 输出轮式双臂人形机器人的 URDF 的基本流程与工具链是什么？ | Zane Hub 原文 PDF 归档、URDF 概念、URDF 惯量真机核查 与下列 ROS 规范。 |
