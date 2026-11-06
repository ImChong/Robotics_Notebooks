# World Action Models 综述论文通读（架构 / 挑战视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/world-action-models.md`）

- **标题：** World Action Models: The Next Frontier in Embodied AI
- **作者：** Fudan University / Shanghai Innovation Institute / NUS 等（OpenMOSS）
- **类型：** paper / survey / world-action-models
- **原文：** <https://arxiv.org/abs/2605.12090>（2026-11 通过 arXiv PDF 获取全文，共 69 页）
- **一句话说明：** 形式化定义 WAM，并按 Cascaded / Joint 两大架构族整理文献，配套数据生态、评测协议与开放挑战。
- **核对范围：** 2026-11 重新获取 arXiv PDF 全文（69 页，v1，2026-05-12），用 pdftotext 抽出正文后逐节通读：摘要、§1 引言（含 Fig. 1 / Fig. 2 分类树）、§2 定义与形式化（含 §2.2 与 VAM / Video Policy 的区分）、§3 全部（§3.1 VLA、§3.2.1–3.2.3 三类世界模型、§3.3.1–3.3.2 世界模型用于学习 / 评测）、§4 全部（§4 总述；§4.1.1 / §4.1.2 逐篇方法与 Table 1；§4.2.1 / §4.2.2 逐篇方法与 Table 2 / 3）、§5 全部（§5.1–5.4 正文及 Table 4–7 的数据集清单）、§6 全部（§6.1.1–6.1.3 指标与基准、Table 8；§6.2.1–6.2.5 与 Table 9）、§7 开放挑战、§8 结论。参考文献表（第 45–69 页）未逐条阅读，仅抽查 [6]、[9]–[17] 条目以核对正文引用编号与论文名；被引原论文本身未回读，下文各被引方法的描述均以本综述叙述为准。正文中的图（Fig. 3–7 的示意图）只读到图注与可抽取文字，未读像素内容。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定义（摘要、§2）：** WAM 是统一「预测性状态建模」与「动作生成」的具身基础模型，目标是未来状态与动作的联合分布，而非仅动作；VLA 对应 p(a|o,l)，视为反应式观测到动作映射。
- **两大架构族（§4）：** Cascaded WAM 先预测下一状态（像素 / 隐空间 / 光流）再导出动作，世界模型与动作解码器作为分离模块各自训练；Joint WAM 在单一模型里同时预测未来状态与动作，统一目标联合训练，迫使模型内化动力学与控制信号的因果依赖。
- **Cascaded 的两类载体（§4.1）：** 显式规划（像素视频，动作由学习式逆动力学或闭式几何提取；UniPi 为奠基蓝图，单次长程生成有语义漂移与误差累积）；隐式规划（隐特征序列，免去像素解码，VPP 首次使规划推理速度可与实时控制兼容）。
- **Joint 的两条生成路线（§4.2）：** 自回归（世界与动作序列化为 token，三种表示：显式解耦、统一离散、预测式隐表示；主要张力是早期视觉幻觉向动作传播与串行解码延迟）；扩散 / 流匹配（统一流：单一 DiT 主干；多流：分支或模态专家经交叉注意力、隐状态条件或共享编码器耦合；摆脱串行瓶颈，利于高频闭环）。
- **数据（§5 导语）：** VLA 需要成对 (o_t, a_t) 轨迹；纯世界模型吃无动作的 (o_t, o_t+1) 视频；WAM 既受益于 (o_t, a_t, o_t+1) 三元组，又可通过联合训练消化大量无动作视频。四类数据：机器人遥操作、便携人类示教（UMI 式）、仿真、人类 / 第一视角视频。
- **评测（§6 导语）：** 尚无联合评估预测与动作因果一致性的协议；现状是世界建模（视觉保真、物理常识、动作可推断性）与动作策略（各类操作基准）分开评。
- **开放挑战（§7）：** 架构耦合缺乏匹配条件下的对照研究，且有证据表明部分 WAM 的收益来自训练期辅助梯度，测试时去掉未来预测头未必降低控制性能；预测几乎全在 RGB，触觉 / 力等不可见物理信息是盲区；数据混合配比原则不清；长程规划存在预测漂移与误差累积；推理延迟（文中称 DreamZero 约 7Hz，对比非生成式 VLA 的 50Hz 标准）；评测需要联合指标（如 Counterfactual Consistency、Foresight-Conditioned Success）；安全上，错误想象的未来可能触发难以中断的长动作序列，但预测也可用于执行前验证。

## 补充核对笔记（原通读中此前仅浏览的部分，已逐节读过）

### §3 VLA 与世界模型背景
- **§3.1 VLA：** 早期三种视觉-语言融合：FiLM 特征调制、交叉注意力、直接拼接，并引入 action chunking / temporal ensembling；第二波继承 LVLM 权重，动作头分自回归 token 化与扩散式动作专家两路；定义继续扩展到 3D 几何、深度、力 / 触觉输入。文中结论：VLA 仍是反应式映射，不捕获世界动力学，操作泛化受限。
- **§3.2 世界模型：** 共识定义为建模环境动力学与动作效应的内部表示。动作条件 P(o'|o,a) 分显式（像素：ACVP、CDNA、Deep Visual Foresight、SV2P、MCnet、ContextVP；自回归视频 iVideoGPT、Genie；扩散视频 Diffusion World Model，长程一致但算力重）与隐式（RSSM：PlaNet、Plan2Explore、Dreamer 系、LEXA；TSSM：TransDreamer；预测式表示：JEPA、I-JEPA、MC-JEPA、V-JEPA 2、LeWorldModel 用 SIGReg 防坍缩、A-JEPA）。语言条件 P(o'|o,l)：GAN 视频模型 → U-Net 扩散（VDM、Text2Video-Zero、AnimateDiff）→ ViT 时空 patch（Sora、Latte）→ 潜空间（LDM、VideoGPT）→ Wan（DiT + Flow Matching），闭源有 Sora 2、Kling 3、Veo 3、Gen-4、Pika 2.2。
- **§3.2.3 具身世界模型：** 从无动作视频学动力学（Genie 潜动作、SWIM、DreamDojo 连续潜动作、DexWM 手部关键点 + 手一致性损失）；提升生成质量（RoboDreamer 组合式、IRASim 帧级动作条件、MiLA 粗到精、Ctrl-World 多视角 token 拼接、RoboScape 关键点动力学 + RGB / 深度双分支、WoW 的 SOPHIA 与 Flow-Mask IDM）；扩展模态（VT-WM 加触觉 token、PointWorld 3D 点流）。
- **§3.3 世界模型服务 VLA（Fig. 4）：** 学习侧三类：模仿学习数据生成 / 过滤（DREMA 高斯泼溅 + 物理仿真；Ctrl-World 想象中合成成功轨迹使 π0.5-DROID 成功率提升 44.7%；RoboScape 合成数据训练的 π0 / Diffusion Policy 与真机数据相当）、RL 的代理环境（Dreamer 系；视频世界模型当环境）、奖励建模（VIPER、Diffusion Reward、GenReward、SRPO 用 V-JEPA 2 隐空间做进度奖励、RoboScape-R 联合建模观测与奖励）；评测侧：世界模型作数据驱动模拟器做闭环、可复现、安全的策略评测（Ctrl-World、Veo Robotics 含 OOD 与红队、Interactive World Simulator）。

### §4 架构逐篇方法细节
- **Cascaded / 显式（§4.1.1，Table 1）：** 学习式动作提取 —— UniPi（视频 U-Net + 卷积 IDM）、VLP（VLM 分层子动作 + 树搜索价值打分）、RoboEnvision（关键帧 + 插值的非自回归）、This&That（指示词 + 手势坐标条件）、Say, Dream & Act（对抗蒸馏减去噪步 + 帧率无关预测）、TesserAct（加深度 / 法线通道）、MVISTA-4D（轨迹级隐优化 + 残差 IDM）、Vidar（人视频转机器人执行视频，掩码 IDM）、Gen2Act（零样本 VideoPoet + 点追踪辅助损失）、Veo-Act（接触前用 IDM，接触检测后交给反应式 VLA）、VAG（视频 U-Net 中间特征条件动作 U-Net）、π0.7（BAGEL 世界模型生成多视角子目标图，注入 VLA 上下文，由流匹配动作专家出动作块）。几何式提取 —— AVDC（稠密光流 + 解析 SE(3)，训练无需动作标注）、Im2Flow2Act（隐空间光流）、3DFlowAction（3D 流）、NovaFlow / Dream2Flow（零训练、深度 + 点追踪得到物体流）、Dreamitate（立体视频扩散 + MegaPose 追踪工具 6DoF 位姿 + 逆运动学）、4DGen、RIGVid（FoundationPose 追物体，轨迹重定向，VLM 过滤生成视频）、LVP（重建 3D 手位姿再运动学映射，Diffusion Forcing）。Table 1 另列各方法是否需要动作标注、是否零样本、仿真 / 真机评测平台。
- **Cascaded / 隐式（§4.1.2）：** VPP（VAE 编码、扩散单步预测未来隐序列、轻量策略出动作）、VILP（双视角隐规划）、S-VAM（冻结多步 SVD 教师自蒸馏到单步，QFormer 式感知器聚合后条件 DiT 扩散策略）、Video Policy（冻结视频 U-Net，动作 U-Net 以解码器中间特征为条件）、ARDuP（Co-Tracker 点追踪 + SAM 掩码得到活动区域条件）、mimic-video（流匹配 + 在 ODE 中间检查点取特征的部分去噪）、MWM（预测未来语义掩码隐变量，缓存隐状态条件动作扩散头）、OmniVTA（双流 DiT 联合生成视觉 + 触觉隐变量，差分编码器）、LAPA（VQ-VAE 潜动作预训练，仅少量真实动作微调）、villa-X（proprio-FDM 将潜动作落地到物理动力学 + 潜动作专家 / 机器人动作专家联合扩散）。
- **Joint / 自回归（§4.2.1，Table 2）：** 显式解耦 —— GR-1（195M，双分支头同时解码未来图块与连续动作，视频预测作正则）、GR-MG（[PROG] token + 扩散生成视觉目标）、GR-2（VQGAN 离散视觉 token + CVAE 动作块）；统一离散 —— CoT-VLA（7B，因果注意力生成离散视觉 CoT，再切换全注意力一次预测动作 token）、WorldVLA（7B，Chameleon 系，修改因果掩码禁止当前动作 token 看同块已生成动作）、RynnVLA-002（5B，附连续 Action Transformer 头并行解码）、F1（4.2B，MoT：生成专家用 next-scale 预测 VQ token，动作专家把动作视为前瞻引导的逆动力学）；预测式隐表示 —— VLA-JEPA（2B，Qwen3-VL-2B，对冻结目标网络编码的未来表示做自回归预测，未来帧仅作监督目标，无泄漏，流匹配头出动作）。文中指出 pixel-matching 捷径（模型耗容量重建背景）是显式 token 路线的风险。
- **Joint / 扩散（§4.2.2，Table 3，Fig. 6）：** 统一流 / 显式未来预测 —— PAD（RGB + 位姿 + 深度联合去噪，动作填充 + 注意力掩码吃无动作视频，消融证实去掉未来图预测或视频共训都会降性能）、VideoVLA（CogVideoX-5B，联合去噪未来视频隐变量与 7-DoF 动作块）、UWM（世界 / 动作独立噪声调度，测试时可切换策略、前向 / 逆动力学、纯视频生成；无动作视频把动作侧全加噪）、Cosmos Policy（Cosmos-Predict2-2B，潜帧注入把本体、动作、未来状态、价值编码成潜帧，一个检查点兼任策略 / 世界模型 / 价值函数，best-of-N 规划）、DreamZero（14B，Wan2.1-I2V，KV-cache 观测替换以免视觉漂移，异步执行 / DiT 缓存 / 量化 / CUDA graph 加速）、GigaWorld-Policy（动作只看历史与当前观测，推理时免生成未来视频）、X-WAM（复制末段 DiT 块为深度分支做 RGB-D，异步噪声采样）、UD-VLA（离散扩散在统一 token 空间联合去噪）。统一流 / 隐式 —— FLARE（可学习未来 token 过同一 DiT，内部层经 MLP 与冻结教师编码的未来观测嵌入对齐，可只用视频不用动作）、FRAPPE（冻结 RDT 之上多个对齐专家，Mixture-of-Prefix-and-LoRA + 路由器，先单教师全参数中期对齐再并行）。多流 / 交叉注意力 —— CoVAR（Bridge Attention 拼接联合注意力再拆回）、LDA-1B（共享 MM-DiT，DINO 隐空间，按数据质量分层监督，任务嵌入 + 模态寄存器 token）、DUST（独立噪声时步、分离流匹配损失、异步采样）、MoT 路线：LingBot-VA（交错视频 / 动作自回归 + 累积 KV-cache + 异步预测执行）、DexWorldModel（TTT 双状态记忆替换 KV-cache，DINOv3 语义特征）、AIM（intent-causal 掩码使动作只经预测的价值图获得未来信息）、Being-H0.7（训练时世界分支蒸馏到隐查询，推理时丢弃世界分支）、Motus（再加 VLM 语义专家，三模态联合注意力，UniDiffuser 式调度，可作 VLA / 世界模型 / 逆动力学）、MotuBrain（H-bridge：仅中间层全联合注意力）、AdaWorldPolicy（第三专家为力预测器，部署时想象与观测 / 力读数的差异驱动 LoRA 在线适配）。多流 / 隐状态耦合 —— DiT4DiT（钩子截取视频 DiT 中间隐状态，tri-timestep，推理用单个确定提取步）、Fast-WAM（训练保留未来视频流匹配目标，推理移除未来视频分支，单次前向取当前上下文特征）、WAV（轨迹价值分支，推理时用高分样本迭代更新噪声分布，免外部规划器）、Act2Goal（目标图条件，Multi-Scale Temporal Hashing 近密远疏，HER 式目标重标注 + LoRA 在线改进）。多流 / 共享表示 —— UVA（0.5B，共享 Transformer 隐 token + 视频 / 动作两个轻量扩散头，推理可绕过视频生成，掩码训练可充当策略 / 视频 / 正逆动力学）、PhysGen（0.73B，NOVA 自回归，共享物理 token + Action-DiT，Lookahead Multi-Token Prediction）。

### §5 训练数据清单（Table 4–7）
- **机器人遥操作（§5.1，Table 4）：** 早期 QT-Opt、MT-Opt、RoboNet、MIME；语言条件 BridgeData / v2、BC-Z、RT-1、Language-Table；长程 TACO-RL、RH20T-P；聚合 OXE（100 万+轨迹、22 台机器人）、ARIO（3M+、35 台机器人）、RoboMIND、DROID（76k 轨迹、564 个场景）、UnifoLM-WBT（Unitree G1，1,892,118 条）；玩耍式 Jaco Play；生成 / 自动增广 Grasp-Anything、Interleave-VLA、UniVoxGen；多模态与接触：Berkeley UR5 与 OmniAction 含音频，TOTO（倒液体 / 透明物）、Cable Routing、REASSEMBLE，RH20T / Robo360 / RoboData 多视角与点云，RoboSet、DexCap、FuSe（触觉）、AgiBot World（1M+ 轨迹，含触觉）。
- **便携人类示教 UMI 式（§5.2，Table 5）：** UMI 为手持 3D 打印夹爪 + 可穿戴相机；FastUMI（10K+）、FastUMI-100K（100K+）、RealOmin（1M 轨迹、3000+ 家庭环境、含 IMU / 触觉）、Hoi!（3,048 条，力感知）、RDT2（约 10,000 小时）；UMI 生态扩展含主动感知、触觉、多视角、灵巧手、全身移动。
- **仿真（§5.3，Table 6）：** MimicGen / DexMimicGen 将少量示教程序化扩展，ManiSkill2、RoboCasa（120 个厨房场景、100k+ 轨迹）、InternData-A1（630k）、InternData-M1（244k，含 2D / 3D 框与分割）、SynGrasp-1B（10M 抓取轨迹）、RoboCerebra（长程）、QUARD-Auto（四足）、TesserAct（285k 带深度 / 法线视频片段，4D）、RoboTwin 2.0、TLA（24k 触觉-语言-动作对，FleX）。文中强调仿真提供特权信息，域随机化缓解 sim-to-real。
- **人类 / 第一视角（§5.4，Table 7）：** 被动视频：SSv2（108,499 段）、EPIC-KITCHENS（55 h）、EGTEA Gaze+、Ego4D（3,670 h）、HowTo100M（136M 段）、Kinetics-700、COM Kitchens、Egocentric-10K、DreamDojo-HV（43,827 h）；位姿 / 本体落地：Assembly101、H2O、EgoPAT3D、Ego-Exo4D（1,286 h）、ARCTIC、HOT3D、TACO、Aria Everyday、OAKINK2、Nymeria（300 h）；混合预训练：Ego-Centric Human Manipulation、UniHand（1,155 h）、EgoDex（829 h、194 任务）、EgoScale（20,854 h）、EgoVerse（1,362 h）、HumanNet（1M h）、Humanoid Everyday、PH2D。

### §6 评测细节（Table 8–9）
- **世界建模评测（§6.1）：** 视觉保真 —— PSNR、SSIM、LPIPS、DreamSim、DINO 余弦相似度、FVD（式 1–6）；物理常识 —— VideoPhy、PhyGenBench（PhyGenEval）、VBench-2.0、WorldModelBench（五条二值物理定律检查）、Physics-IQ（Spatial / Spatiotemporal / Weighted Spatial IoU、MSE）；运动轨迹 —— WorldScore、EWMBench（HSD / nDTW / DYN）；动作可行性 —— WorldSimBench（Implicit Manipulative Evaluation）、Wow, wo, val!（IDM Turing Test，许多视觉上逼真的模型在真机执行成功率接近 0）。
- **动作策略评测（§6.2，Table 9，40+ 基准）：** 通用操作 MetaWorld、RLBench、Robomimic、Franka Kitchen、CALVIN、ManiSkill 1/2/3、LIBERO 及 LIBERO-plus / pro / X、COLOSSEUM、AGNOSTOS、GemBench、SimplerEnv、PolaRiS、RoboMME、RoboCasa、RoboVerse 等；双臂 / 人形 RoboTwin、BiGym、HumanoidBench（448 个触觉点、27 任务）、HumanoidGen；移动操作 ManipulaTHOR、HomeRobot、BEHAVIOR-1K；接触与形变 SoftGym、PlasticineLab、DaXBench、TacSL、ManiFeel；真机 RoboArena、RoboChallenge（30 任务、4 平台）、Maniparena（10,812 条真机轨迹）。

### 核对中发现的小出入（仅记录，不影响上方摘录）
- 综述 Table 3 把 DexWorldModel 的引用号写作 [326]，而正文与图 2 写作 [103]；[326] 在 §3.2.3 对应 DexWM。

## 对 wiki 的映射

- [`wiki/concepts/world-action-models.md`](../../wiki/concepts/world-action-models.md)
- 既有摘要型归档：[`world_action_models_survey_2605.md`](./world_action_models_survey_2605.md)
