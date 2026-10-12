# 【10.5-10.9前沿论文动态】Manipulation34篇

- **作者：** senlanke（具身运控lab）
- **发表日期：** 2026-10-12
- **原文：** <https://mp.weixin.qq.com/s/oPAUE5EvKeOfGU-Vjsynrw>
- **归档依据：** 用 `scripts/fetch_wechat_article.py`（Chromium + 微信 UA）读取公众号正文后提炼条目；此页只保留条目线索与详情节点映射，论文技术事实以各详情节点引用的 arXiv 原文为准。
- **条目数：** 34 个论文条目；同一 arXiv 论文只保留一个 wiki 详情节点（与另一篇周报重复的条目共用同一节点）。

## 条目与详情节点

### VLA / 通用操作策略（4篇）

#### ViDAL: A Visual Dynamics-Grounded Action Latent Space for Vision-Language-Action Models（arXiv:2610.08150）

- **论文：** [arXiv:2610.08150](https://arxiv.org/abs/2610.08150)
- **机构（周报所述，未逐条核实）：** 中国科学院相关研究团队等，具体联合机构待核实。
- **问题：** VLA 的动作表示主要从机器人示范中学习，动作特征与环境实际发生的视觉变化之间缺少明确联系。
- **方法线索：** ViDAL 将视觉动力学引入动作潜在空间学习，使动作表示能够反映动作引起的场景变化。
- **详情节点：** [ViDAL：视觉动力学锚定的 VLA 动作潜空间](../../wiki/entities/paper-vidal-visual-dynamics-action-latent-vla.md)

#### EgoLAP: Learning from Egocentric Human Data through Language-Action Reasoning（arXiv:2610.08726）

- **论文：** [arXiv:2610.08726](https://arxiv.org/abs/2610.08726)
- **机构（周报所述，未逐条核实）：** 普林斯顿大学等。
- **问题：** 人类第一视角视频包含丰富的操作经验，但缺少机器人关节动作标签。
- **方法线索：** EgoLAP 引入语言与动作推理，将人类第一视角数据中的操作信息用于机器人学习。
- **详情节点：** [EgoLAP：用语言动作推理从第一视角人类数据学机器人](../../wiki/entities/paper-egolap-egocentric-language-action-reasoning.md)

#### VLA-ZO: Fast Zeroth-Order Adaptation for Vision-Language-Action Models（arXiv:2610.06271）

- **论文：** [arXiv:2610.06271](https://arxiv.org/abs/2610.06271)
- **机构（周报所述，未逐条核实）：** Taesik Gong 研究团队，具体机构待核实。
- **问题：** VLA 部署到新相机视角或新环境后，需要适应视觉分布变化。
- **方法线索：** VLA-ZO 冻结视觉语言部分，仅对动作侧进行零阶优化，并复用视觉语言模型的条件特征，避免每次参数扰动都重新计算大型 Backbone。
- **详情节点：** [VLA-ZO：VLA 的快速零阶适应](../../wiki/entities/paper-vla-zo-zeroth-order-adaptation.md)

#### PLaW-VLA: Predictive Latent World Modeling for Vision-Language-Action Policies（arXiv:2610.12285）

- **论文：** [arXiv:2610.12285](https://arxiv.org/abs/2610.12285)
- **机构（周报所述，未逐条核实）：** 作者联合研究团队，具体机构待核实。
- **问题：** VLA 主要依赖当前视觉信息生成动作，对动作执行后的环境变化缺少显式预测。
- **方法线索：** 将预测式潜在世界建模引入 VLA，使策略学习同时利用未来状态的潜在表示。
- **详情节点：** [PLaW-VLA：预测式潜在世界建模的 VLA 策略](../../wiki/entities/paper-plaw-vla-predictive-latent-world-model-vla.md)

### WAM / World Model 与操作预测（7篇）

#### OpenWAM: An Open Framework for Composable World-Action Models（arXiv:2610.07922）

- **论文：** [arXiv:2610.07922](https://arxiv.org/abs/2610.07922)
- **机构（周报所述，未逐条核实）：** 斯坦福大学，李飞飞、Jiajun Wu 等参与。
- **问题：** 现有 WAM 同时改变视频生成 Backbone、动作网络、模态交互方式和推理流程，很难判断性能提升来自哪一个设计。
- **方法线索：** OpenWAM 从 Wan2.2-5B 出发，使用超过 10,000 小时机器人相关视频进行因果视频预训练，再通过 Mixture-of-Transformers 连接视频与动作分支。
- **详情节点：** [OpenWAM（Stanford）：可组合世界–动作模型框架](../../wiki/entities/paper-openwam-composable-world-action-models.md)

#### Cross-Embodiment Robot Foundation World Models with Latent Actions（arXiv:2610.10846）

- **论文：** [arXiv:2610.10846](https://arxiv.org/abs/2610.10846)
- **机构（周报所述，未逐条核实）：** 斯坦福大学、Meta 相关研究团队等，李飞飞、Jiajun Wu、Franziska Meier 参与。
- **问题：** 不同机器人具有不同的关节结构和动作空间。
- **方法线索：** 提出 LAC-WM（潜在动作条件世界模型），通过学习共享的潜在动作空间，统一不同机器人本体的动作表示。
- **详情节点：** [LAC-WM：基于潜在动作的跨本体机器人基础世界模型](../../wiki/entities/paper-lac-wm-cross-embodiment-latent-action-world-model.md)

#### MiniWAM: Learning Compact Future Targets for Efficient World-Action Modeling（arXiv:2610.12194）

- **论文：** [arXiv:2610.12194](https://arxiv.org/abs/2610.12194)
- **机构（周报所述，未逐条核实）：** Guillaume Sartoretti 等研究团队，具体机构待核实。
- **问题：** WAM 通常预测高维未来视觉特征，导致大量训练计算被用于重建与机器人控制关系不大的视觉信息。
- **方法线索：** 提出 PRISM（通过逆向时空建模学习预测表示），结合逆动力学监督与特征重建，学习更紧凑的未来状态表示。
- **详情节点：** [MiniWAM：用紧凑未来目标做高效世界–动作建模](../../wiki/entities/paper-miniwam-compact-future-targets.md)

#### RealtimeWAM: How Fast Can I Run My World Action Model?（arXiv:2610.10079）

- **论文：** [arXiv:2610.10079](https://arxiv.org/abs/2610.10079)
- **机构（周报所述，未逐条核实）：** Zhi Wang 等研究团队，具体机构待核实。
- **问题：** WAM 同时承担未来世界预测和机器人动作生成，视频生成过程带来的计算延迟限制了模型在真实机器人闭环控制中的应用。
- **方法线索：** RealtimeWAM 将实时执行效率作为主要研究目标，围绕世界预测与动作生成之间的计算开销开展优化。
- **详情节点：** [RealtimeWAM：免训练的世界–动作模型实时推理](../../wiki/entities/paper-realtimewam-training-free-wam-acceleration.md)

#### Long-WAM: Scaling the Context of World-Action Models（arXiv:2610.10528）

- **论文：** [arXiv:2610.10528](https://arxiv.org/abs/2610.10528)
- **机构（周报所述，未逐条核实）：** 多机构联合研究团队，具体机构待核实。
- **问题：** 短时间窗口难以提供长程机器人操作所需的完整历史信息。
- **方法线索：** Long-WAM 研究扩大 WAM 的时间上下文，使世界预测与动作生成能够利用更长时间的历史信息。
- **详情节点：** [Long-WAM：Scaling the Context of World-Action Models](../../wiki/entities/paper-long-wam-scaling-context.md)

#### UNITAS: A 3D-Native World Action Model for Embodied Manipulation（arXiv:2610.12099）

- **论文：** [arXiv:2610.12099](https://arxiv.org/abs/2610.12099)
- **机构（周报所述，未逐条核实）：** Kui Jia 等联合研究团队，具体机构待核实。
- **问题：** 基于二维视频生成的 WAM 需要间接学习三维几何关系。
- **方法线索：** UNITAS 采用原生三维世界动作建模路线，将三维空间信息作为世界预测与动作学习的重要表示。
- **详情节点：** [UNITAS](../../wiki/entities/paper-unitas-3d-native-wam.md)

#### WAM-Cache: Staleness-Bounded KV Reuse for Efficient World Action Models（arXiv:2610.11401）

- **论文：** [arXiv:2610.11401](https://arxiv.org/abs/2610.11401)
- **机构（周报所述，未逐条核实）：** Kai Ding 等研究团队，具体机构待核实。
- **问题：** WAM 在连续控制过程中会反复处理相似的历史视觉和动作上下文，导致大量重复计算。
- **方法线索：** WAM-Cache 将缓存陈旧度纳入 KV Cache 复用机制，在减少重复计算的同时限制过时信息对动作预测的影响。
- **详情节点：** [WAM-Cache](../../wiki/entities/paper-wam-cache.md)

### 在线/离线强化学习与后训练（3篇）

#### Many Ways to Succeed: Diversity-Driven RL Fine-Tuning for VLA Generalization（arXiv:2610.09943）

- **论文：** [arXiv:2610.09943](https://arxiv.org/abs/2610.09943)
- **机构（周报所述，未逐条核实）：** 南京大学相关研究团队等。
- **问题：** RL Fine-Tuning 能提高 VLA 在训练任务上的成功率，但策略可能逐渐集中到少数成功动作模式。
- **方法线索：** 提出 DRIVE，将成功轨迹的行为多样性加入强化学习目标。
- **详情节点：** [DRIVE（多样性驱动的 VLA 强化微调）](../../wiki/entities/paper-drive-diversity-rl-vla.md)

#### EigenDEXplore: Structured Exploration for Dexterous Manipulation with Human Priors（arXiv:2610.07681）

- **论文：** [arXiv:2610.07681](https://arxiv.org/abs/2610.07681)
- **机构（周报所述，未逐条核实）：** 斯坦福大学、哥伦比亚大学相关研究团队等。
- **问题：** 多指灵巧手的动作空间维度高，直接在所有关节上进行随机探索，容易产生大量无效动作，降低强化学习的样本效率。
- **方法线索：** EigenDEXplore 将人类操作先验引入灵巧手探索过程，通过结构化探索减少高维动作空间中的无效搜索。
- **详情节点：** [EigenDEXplore](../../wiki/entities/paper-eigendexplore.md)

#### PathTime-VLA: Path-Time Decoupling for Factorized Post-Training of Vision-Language-Action Policies（arXiv:2610.11771）

- **论文：** [arXiv:2610.11771](https://arxiv.org/abs/2610.11771)
- **机构（周报所述，未逐条核实）：** 浙江大学相关研究团队等。
- **问题：** VLA 动作轨迹同时包含空间路径和执行时间信息。
- **方法线索：** PathTime-VLA 将空间路径与时间参数分开建模，并据此设计分解式后训练方法。
- **详情节点：** [PathTime-VLA](../../wiki/entities/paper-pathtime-vla.md)

### 模仿学习、视觉运动策略与生成式动作策略（3篇）

#### Immiscible Diffusion Policy: Preserving Multimodal Robot Actions through Label-Free Noise Assignment（arXiv:2610.09369）

- **论文：** [arXiv:2610.09369](https://arxiv.org/abs/2610.09369)
- **机构（周报所述，未逐条核实）：** 加州大学伯克利分校相关研究团队等。
- **问题：** 同一机器人状态可能对应多种合理动作，例如从不同方向抓取同一物体。
- **方法线索：** 提出无标签噪声分配机制，改善扩散模型训练时噪声与动作样本之间的对应关系，重点是减少不同动作模式在生成过程中相互干扰，从而保留更多有效的操作行为。
- **详情节点：** [Immiscible Diffusion Policy](../../wiki/entities/paper-immiscible-diffusion-policy.md)

#### ESP: Energy-Score Policy for One-Step Multimodal Action Generation（arXiv:2610.07696）

- **论文：** [arXiv:2610.07696](https://arxiv.org/abs/2610.07696)
- **机构（周报所述，未逐条核实）：** Yasuyuki Matsushita 等研究团队，具体机构待核实。
- **问题：** Diffusion Policy 能生成多种动作，但通常需要多步迭代去噪。
- **方法线索：** ESP 研究基于能量评分的单步多模态动作生成，希望在减少生成步骤的同时保留动作分布的多模态性质。
- **详情节点：** [ESP（Energy-Score Policy）](../../wiki/entities/paper-esp-energy-score-policy.md)

#### RoboPace: Contact-Aware Time-Optimal Retiming for Action-Chunk Policies（arXiv:2610.09696）

- **论文：** [arXiv:2610.09696](https://arxiv.org/abs/2610.09696)
- **机构（周报所述，未逐条核实）：** Kei Ota 等研究团队，具体机构待核实。
- **问题：** Action Chunk 通常包含固定的动作时间安排，但机器人在自由空间移动和与物体接触时，对速度和执行精度的要求不同。
- **方法线索：** RoboPace 将接触状态纳入动作块时间重规划，在保留动作路径的基础上调整执行时间。
- **详情节点：** [RoboPace](../../wiki/entities/paper-robopace.md)

### 双臂操作（2篇）

#### Arm-wise Compositional Generalization in Dual-Arm Vision-Language-Action Models（arXiv:2610.06184）

- **论文：** [arXiv:2610.06184](https://arxiv.org/abs/2610.06184)
- **机构（周报所述，未逐条核实）：** Huchuan Lu 等研究团队，具体联合机构待核实。
- **问题：** 双臂机器人可能已经分别学会多个单臂技能，但当任务要求重新分配左右臂角色、改变技能执行顺序或增加同步约束时，策略往往无法直接组合已有能力。
- **方法线索：** 提出 ACG-Bench，包含 8 类任务、23 组任务条件，重点测试双臂技能重排序、同步执行和跨任务组合泛化。
- **详情节点：** [AE-VLA 与 ACG-Bench](../../wiki/entities/paper-ae-vla-arm-wise-compositional.md)

#### CoRE: Learning Collaboration-Role Experts for Decentralized Collaborative Manipulation with One Policy（arXiv:2610.07752）

- **论文：** [arXiv:2610.07752](https://arxiv.org/abs/2610.07752)
- **机构（周报所述，未逐条核实）：** Weiming Zhi 等研究团队，具体机构待核实。
- **问题：** 多机器人协同操作需要不同执行单元承担不同角色。
- **方法线索：** CoRE 引入协作角色专家，使多个机器人能够通过同一策略学习不同协作角色。
- **详情节点：** [CoRE（Collaboration-Role Experts）](../../wiki/entities/paper-core-collaboration-role-experts.md)

### 灵巧操作（2篇）

#### OmniHOI: Dexterous Hand-Object Interaction from Monocular Human Video（arXiv:2610.10855）

- **论文：** [arXiv:2610.10855](https://arxiv.org/abs/2610.10855)
- **机构（周报所述，未逐条核实）：** Yao Mu 等研究团队，具体机构待核实。
- **问题：** 人类操作视频包含大量灵巧手动作，但单目视频缺少完整三维信息，手与物体之间还存在遮挡。
- **方法线索：** OmniHOI 将单目视频转换为机器人灵巧手可执行轨迹，并在不同阶段加入物理一致性约束。
- **详情节点：** [OmniHOI](../../wiki/entities/paper-omnihoi.md)

#### OmniDex: Scaling Dexterous Hand Grasping to Diverse Cluttered Scenes（arXiv:2610.11194）

- **论文：** [arXiv:2610.11194](https://arxiv.org/abs/2610.11194)
- **机构（周报所述，未逐条核实）：** 香港中文大学相关研究团队等。
- **问题：** 灵巧手在简单场景中能够完成抓取，但在多个物体相互遮挡、接触关系复杂的杂乱场景中，抓取策略的泛化能力仍然有限。
- **方法线索：** OmniDex 研究面向多样化杂乱场景的灵巧手抓取学习，将复杂场景中的目标识别、抓取姿态选择和多指协调作为主要问题。
- **详情节点：** [OmniDex](../../wiki/entities/paper-omnidex.md)

### 柔性/可变形物体操作（2篇）

#### Point It, Strike It: Direction-Conditioned Dynamic Manipulation of Deformable Linear Objects（arXiv:2610.09573）

- **论文：** [arXiv:2610.09573](https://arxiv.org/abs/2610.09573)
- **机构（周报所述，未逐条核实）：** Howie Choset 等研究团队，具体机构待核实。
- **问题：** 绳索、线缆等可变形线状物体具有复杂的动态行为。
- **方法线索：** 提出方向条件驱动的动态操作方法，将期望操作方向引入动作生成与控制过程，研究机器人如何通过动态接触改变线状柔性物体的运动状态。
- **详情节点：** [Point It, Strike It](../../wiki/entities/paper-point-it-strike-it.md)

#### Same Action, Different Outcome: Variability in Dynamic Cloth Manipulation（arXiv:2610.10801）

- **论文：** [arXiv:2610.10801](https://arxiv.org/abs/2610.10801)
- **机构（周报所述，未逐条核实）：** 西班牙机器人与工业信息学研究所相关团队。
- **问题：** 即使机器人重复执行完全相同的动作，布料也可能产生明显不同的最终形态。
- **方法线索：** 作者在四类动态布料任务中，使用三种物理性质差异明显的布料，重复执行相同动作，共采集 269 条真实轨迹。
- **详情节点：** [Same Action, Different Outcome](../../wiki/entities/paper-same-action-different-outcome-cloth.md)

### 人形机器人上半身操作（4篇）

#### InterMimicGen: Scaling Humanoid Loco-Manipulation through Self-Evolving Motion Imitation（arXiv:2610.06850）

- **论文：** [arXiv:2610.06850](https://arxiv.org/abs/2610.06850)
- **机构（周报所述，未逐条核实）：** 亚利桑那州立大学相关研究团队等。
- **问题：** 人形机器人移动操作需要同时协调行走、全身姿态、手臂运动和物体交互。
- **方法线索：** InterMimicGen 研究自演化运动模仿，通过扩展已有运动示范，提高人形移动操作训练数据的多样性和可扩展性。
- **详情节点：** [InterMimicGen：用自演进动作模仿扩展人形移动操作](../../wiki/entities/paper-intermimicgen.md)

#### HULK: Learning Whole-Body Forceful Loco-Manipulation for Humanoids（arXiv:2610.08970）

- **论文：** [arXiv:2610.08970](https://arxiv.org/abs/2610.08970)
- **机构（周报所述，未逐条核实）：** 加州理工学院、密歇根大学相关研究团队等。
- **问题：** 人形机器人在推、拉、搬运等任务中，需要持续向环境施加较大的接触力。
- **方法线索：** HULK 研究通过学习式全身控制实现强力移动操作，使机器人能够协调身体运动、平衡与外部接触力。
- **详情节点：** [HULK：重载人形机器人的全身移动操作](../../wiki/entities/paper-hulk-forceful-loco-manipulation.md)

#### Workhorse: Learning Robust Whole-Body Humanoid Loco-Manipulation from Human Data（arXiv:2610.09117）

- **论文：** [arXiv:2610.09117](https://arxiv.org/abs/2610.09117)
- **机构（周报所述，未逐条核实）：** 加州大学伯克利分校相关研究团队。
- **问题：** 人类操作数据包含自然的全身协调行为，但人类与机器人在动力学、关节结构和接触约束上存在差异，直接模仿容易产生不稳定或不可执行动作。
- **方法线索：** Workhorse 将人类运动数据用于鲁棒人形全身移动操作学习，重点研究从人类示范中提取可执行的全身协调能力，并提升策略对不同操作条件的适应能力。
- **详情节点：** [Workhorse：从人类数据学习稳健全身移动操作](../../wiki/entities/workhorse-humanoid-loco-manipulation.md)

#### I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning（arXiv:2610.06129）

- **论文：** [arXiv:2610.06129](https://arxiv.org/abs/2610.06129)
- **机构（周报所述，未逐条核实）：** Hao Zhao 等研究团队，具体机构待核实。
- **问题：** 人形机器人需要适应多种外部接触和交互任务。
- **方法线索：** I-BFM 将无监督强化学习与奖励条件化策略结合，研究如何预先学习可复用的交互能力，再根据不同任务奖励选择相应行为。
- **详情节点：** [I-BFM: Reward-Conditioned Robust Humanoid Interaction via Unsupervised Reinforcement Learning](../../wiki/entities/paper-i-bfm.md)

### 长程任务、推理与规划（3篇）

#### PACE: Stage-Consistent Long-Horizon Robot Manipulation via Progress-Aligned Context for Execution（arXiv:2610.07917）

- **论文：** [arXiv:2610.07917](https://arxiv.org/abs/2610.07917)
- **机构（周报所述，未逐条核实）：** 浙江大学相关研究团队。
- **问题：** 长程操作通常包含多个连续阶段。
- **方法线索：** PACE 将任务进度与执行上下文对齐，使策略能够根据当前操作阶段选择对应的执行信息。
- **详情节点：** [PACE](../../wiki/entities/paper-pace-progress-aligned-context.md)

#### Diagnosing and Recovering from Observation-Space Shift at Long-Horizon Skill Seams（arXiv:2610.10810）

- **论文：** [arXiv:2610.10810](https://arxiv.org/abs/2610.10810)
- **机构（周报所述，未逐条核实）：** Mingyu Ding 等研究团队，具体机构待核实。
- **问题：** 长程任务经常通过串联多个独立训练的技能完成。
- **方法线索：** 论文将这种问题定义为 Observation-Space Shift（观测空间偏移）。
- **详情节点：** [Detect-Restore-Resume（OSS 技能衔接恢复）](../../wiki/entities/paper-oss-detect-restore-resume.md)

#### SkillWeave: Weaving Heterogeneous Demonstrations into Long-Horizon Manipulation Skills（arXiv:2610.12046）

- **论文：** [arXiv:2610.12046](https://arxiv.org/abs/2610.12046)
- **机构（周报所述，未逐条核实）：** Jeffrey Ichnowski 等研究团队，具体机构待核实。
- **问题：** 长程机器人操作往往需要多种技能，而这些技能可能来自不同来源、不同数据采集方式或不同操作任务。
- **方法线索：** SkillWeave 研究将异构操作示范组织为可组合的长程操作技能，提高不同来源示范数据的复用能力。
- **详情节点：** [SkillWeave](../../wiki/entities/paper-skillweave.md)

### 数据集、预训练、跨本体与硬件平台（4篇）

#### BiGym 2.0: Benchmarking Learned and Agent-Developed Policies for Humanoid Household Manipulation（arXiv:2610.07594）

- **论文：** [arXiv:2610.07594](https://arxiv.org/abs/2610.07594)
- **机构（周报所述，未逐条核实）：** Stephen James 等研究团队，具体机构待核实。
- **问题：** 人形家庭操作任务涉及不同物体、复杂环境和较长执行过程。
- **方法线索：** BiGym 2.0 扩展人形家庭操作评测体系，将学习得到的机器人策略与智能体开发的策略纳入统一评测范围，重点研究不同控制方法在家庭操作任务中的能力。
- **详情节点：** [BiGym 2.0：人形家庭操作基准](../../wiki/entities/paper-bigym2-humanoid-household-benchmark.md)

#### OpenViTac: Learning and Benchmarking Visuo-Tactile Policies in a Unified Sim-and-Real Framework（arXiv:2610.10384）

- **论文：** [arXiv:2610.10384](https://arxiv.org/abs/2610.10384)
- **机构（周报所述，未逐条核实）：** 复旦大学相关研究团队等。
- **问题：** 视觉触觉机器人策略需要同时处理图像、接触状态和动作信息，但不同研究使用的触觉传感器、仿真环境与真实机器人接口不统一，增加了算法复现和 Sim-to-Real 评测难度。
- **方法线索：** OpenViTac 建立统一的仿真与真实机器人视觉触觉学习框架，使不同视觉触觉策略能够在一致的训练和评测流程下进行比较。
- **详情节点：** [OpenViTac](../../wiki/entities/paper-openvitac.md)

#### TACROSS: An Efficient and Low-Cost Scalable Human Touch System Across Heterogeneous Tactile Sensors for Dexterous Robot Learning（arXiv:2610.11945）

- **论文：** [arXiv:2610.11945](https://arxiv.org/abs/2610.11945)
- **机构（周报所述，未逐条核实）：** Jiayu Chen 等联合研究团队，具体机构待核实。
- **问题：** 灵巧操作学习需要大量真实接触数据，但不同触觉传感器在结构、分辨率和输出形式上存在差异，导致触觉数据难以统一采集与复用。
- **方法线索：** TACROSS 研究跨异构触觉传感器的低成本人类触摸数据采集系统，重点是提高触觉数据采集的可扩展性，为灵巧机器人学习提供更容易获取的接触数据。
- **详情节点：** [TACROSS](../../wiki/entities/paper-tacross.md)

#### ManiUnit: A Manipulation Skill Dataset and Benchmark for Long-Horizon Tasks（arXiv:2610.12089）

- **论文：** [arXiv:2610.12089](https://arxiv.org/abs/2610.12089)
- **机构（周报所述，未逐条核实）：** Guoting Wei 等研究团队，具体机构待核实。
- **问题：** 长程操作任务需要将多个基础技能组合起来执行，但现有数据集往往更关注单个操作任务，缺少针对技能组合与长程执行能力的统一评测。
- **方法线索：** ManiUnit 建立面向长程任务的操作技能数据集和 Benchmark，将技能级数据与长程任务评测结合，为研究技能复用、任务组合和执行可靠性提供统一测试基础。
- **详情节点：** [ManiUnit](../../wiki/entities/paper-maniunit.md)

## 说明

- ★ 为周报标注的“与表示与强化学习、视觉运动控制、Teacher–Student、复杂地形运动及项目相关方向高度相关”条目。
- 周报机构信息仅作线索；各详情节点的机构 tag 以 arXiv 页、项目页或仓库为准，注册表缺失的机构已略过。
