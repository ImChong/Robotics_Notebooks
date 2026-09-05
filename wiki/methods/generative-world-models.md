---
type: method
tags: [world-models, generative-ai, simulation, video-generation, driving]
status: complete
updated: 2026-09-05
related:
  - ../entities/paper-lejepa.md
  - ../entities/paper-lewm.md
  - ../entities/paper-lpwm.md
  - ../entities/paper-phi-wm-acteffect.md
  - ../entities/paper-vgi-white-paper.md
  - ../entities/current-robotics-currentworld.md
  - ../entities/paper-odeworld.md
  - ../queries/embodied-fm-taxonomy-loop.md
  - ../entities/paper-sc3-eval.md
  - ../entities/paper-worldecho-worldsync.md
  - ../entities/paper-motionwam-humanoid-loco-manipulation-wam.md
  - ../entities/paper-navwam-goal-conditioned-visual-navigation-wam.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../overview/e2e-autonomous-driving-top10-algorithms.md
  - ../entities/paper-gaia1.md
  - ../concepts/humanoid-policy-network-architecture.md
  - ../concepts/latent-imagination.md
  - ../concepts/world-action-models.md
  - ../concepts/functional-taxonomy-world-models.md
  - ../entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md
  - ../entities/paper-unified-robot-learning-survey.md
  - ../methods/model-based-rl.md
  - ../methods/being-h07.md
  - ../entities/nvidia-omniverse.md
  - ../entities/ewmbench.md
  - ../entities/paper-worldscore.md
  - ../entities/paper-harnesseval-w.md
  - ../entities/paper-wem-world-ego-modeling.md
  - ../entities/paper-gamma-world-multi-agent.md
  - ../entities/paper-homeworld-whole-home-scene-generation.md
  - ../entities/paper-infinite-diffusion-terrain-diffusion.md
  - ../entities/tau0-world-model.md
  - ../entities/xiaomi-robotics-u0.md
  - ../entities/cosmos-3.md
  - ../entities/nvidia-cosmos.md
  - ../entities/cosmos-transfer.md
  - ../entities/cosmos-cookbook.md
  - ../entities/paper-instant-nurec.md
  - ../entities/nvidia-nurec.md
  - ../entities/newton-physics.md
  - ../entities/paper-kairos-native-world-model-stack.md
  - ../entities/paper-physmani-dynamic-manipulation-world-model.md
  - ../entities/paper-embodiedvae.md
  - ../entities/paper-panoworld-real-world-panoramic-generation.md
  - ../entities/paper-physisforcing.md
  - ../entities/paper-flex-pi.md
  - ../entities/paper-oscar.md
  - ../entities/paper-gaussiandream-plusplus.md
  - ../entities/paper-confal-wm.md
  - ../overview/wam-vla-cross-embodiment-9-papers-technology-map.md
  - ../entities/paper-driftworld.md
  - ../entities/paper-masked-visual-actions.md
  - ../entities/paper-ctrl-world.md
  - ../entities/paper-clap-cross-embodiment.md
  - ../entities/paper-wall-ss.md
  - ../entities/paper-rofacto.md
  - ../entities/paper-vitacworld.md
  - ../entities/paper-wan-move.md
  - ../entities/paper-wan-dancer.md
  - ../entities/paper-wan-video.md
  - ../entities/paper-robointer-1-5.md
  - ../entities/molmo-motion.md
  - ../entities/robotic-world-model-eth-rsl.md
  - ../entities/world-labs.md
  - ../entities/roboscience-vloa.md
  - ../entities/paper-x-world.md
  - ../entities/paper-x-cache.md
  - ../entities/paper-x-foresight.md
  - ../entities/paper-rise-adaptive-imagination-wam.md
  - ../entities/paper-x-mind.md
  - ../entities/paper-m4world.md
  - ../entities/paper-abot-world-0.md
  - ../entities/open-dreamer.md
  - ./dwm.md
  - ./mimic-video.md
  - ../entities/paper-levjepa.md
sources:
  - ../../sources/papers/wm_robot_survey_arxiv_2605_00080.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/exoactor.md
  - ../../sources/papers/being_h07.md
  - ../../sources/papers/world_action_models_survey_2605.md
  - ../../sources/papers/flex_pi_arxiv_2608_10860.md
  - ../../sources/papers/ewmbench.md
  - ../../sources/papers/worldscore_arxiv_2504_00983.md
  - ../../sources/papers/harnesseval_w_arxiv_2608_16859.md
  - ../../sources/papers/dwm_arxiv_2512_17907.md
  - ../../sources/papers/mimic_video_arxiv_2512_15692.md
  - ../../sources/papers/infinite_diffusion_terrain_diffusion_siggraph_2026.md
  - ../../sources/papers/panoworld_arxiv_2607_09661.md
  - ../../sources/papers/wem_arxiv_2605_19957.md
  - ../../sources/papers/gamma_world_arxiv_2605_28816.md
  - ../../sources/papers/x_world_arxiv_2603_19979.md
  - ../../sources/papers/x_cache_arxiv_2604_20289.md
  - ../../sources/papers/x_foresight_arxiv_2605_24892.md
  - ../../sources/papers/x_mind_arxiv_2606_28758.md
  - ../../sources/papers/driftworld_arxiv_2607_15065.md
  - ../../sources/papers/masked_visual_actions_arxiv_2607_19343.md
  - ../../sources/papers/ctrl_world_arxiv_2510_10125.md
  - ../../sources/papers/clap_arxiv_2608_27406.md
  - ../../sources/papers/wall_ss_x_square_2026.md
  - ../../sources/blogs/current_robotics_currentworld.md
  - ../../sources/papers/odeworld_arxiv_2607_27924.md
  - ../../sources/papers/wan_move_arxiv_2512_08765.md
  - ../../sources/papers/wan_dancer_arxiv_2607_09581.md
  - ../../sources/papers/wan_video_arxiv_2503_20314.md
  - ../../sources/papers/m4world_arxiv_2607_14005.md
  - ../../sources/papers/abot_world_0_arxiv_2607_19191.md
  - ../../sources/sites/worldlabs-ai.md
  - ../../sources/blogs/worldlabs_functional_taxonomy_world_models.md
  - ../../sources/papers/world_model_definition_roadmap_arxiv_2607_06401.md
  - ../../sources/repos/abot-world.md
  - ../../sources/sites/abot-world.md
  - ../../sources/blogs/allenai_molmo_motion.md
  - ../../sources/repos/open-dreamer.md
summary: "生成式世界模型（Generative World Models）利用扩散模型或视频生成技术来模拟物理世界的动态，为机器人提供高保真的视频级仿真和无限的反事实推演能力。"
---

# Generative World Models (生成式世界模型)

**生成式世界模型** 是具身智能（Embodied AI）领域的下一代物理引擎替代者。不同于 Drake 或 MuJoCo 等基于严谨几何和力学方程的解析引擎，生成式世界模型直接利用**生成式 AI (Generative AI)** 的能力，通过海量视频数据学习世界的运动规律。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 预测环境动态，供规划/RL/评估使用 |
| GWM | Generative World Model | 用生成式 AI 从视频学习世界规律 |
| RL | Reinforcement Learning | 可在想象 rollout 中试错的训练范式 |
| MBRL | Model-Based Reinforcement Learning | 显式或学习式环境模型的 RL |
| VLA | Vision-Language-Action | 可与世界模型级联或联合训练 |

## 核心理念：以生成代替计算

在传统仿真中，我们需要手动编写复杂的接触力方程；而在生成式世界模型中，模型学会了“如果机器人向左打方向盘，画面应该如何平滑变化”。

选型时先标功能格：本页大多数系统在 [Fei-Fei 功能分类](../concepts/functional-taxonomy-world-models.md) 里是 **Renderer**（吐像素），只有动作条件、可查询几何或闭环规划时才跨到 Simulator / Planner。[上海人工智能实验室定义文](../entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md) 再加一列架构（observation / latent / 3D）：好看视频不等于有可干预的压缩物理状态。

### 主要架构
1. **视频生成器 (Video Diffusion/Autoregressive)**：如 GAIA-1 或 UniSim。给定当前画面和动作序列，生成一段长达数秒甚至数分钟的未来预测视频。
2. **反事实推演 (Counterfactual Reasoning)**：允许用户输入“如果没有躲避障碍物会怎样？”，模型会生成相应的碰撞视频，作为强化学习的负样本。

## 典型代表作

### 1. GAIA-1 (Wayve)
针对自动驾驶设计的世界模型。它不仅能生成真实的驾驶场景，还能根据文本描述（如“突然下起大雨”）动态改变天气和光影。独立实体页见 [GAIA-1](../entities/paper-gaia1.md)；端到端驾驶脉络对照 [十大算法地图](../overview/e2e-autonomous-driving-top10-algorithms.md)。

### 2. UniSim (Google DeepMind)
一个通用的具身智能世界模型。它将现实世界的视频数据和仿真数据结合，允许机器人在“视频”中练习开橱柜、拿杯子等精细操作，并将学到的技能无缝迁移到真实物理世界。

## 优势与挑战

### 优势
- **视觉真实度极致**：彻底解决了 Sim2Real 在感知层面的 Gap。
- **无需手动建模**：对于复杂的流体、软体（如折衣服、揉面团），生成式模型比物理引擎更容易捕捉其动态特性。

### 挑战
- **物理一致性缺失**：模型有时会产生违反物理常识的幻觉（如物体凭空消失）。
- **推理开销大**：目前生成一帧高质量视频的速度远低于物理引擎的 1000Hz 要求。
- **交互精度低**：很难通过生成的视频反推精确到毫米级的接触力。
- **评测口径漂移**：通用「文生视频」基准往往强调美学与粗粒度语义；面向操纵的 **场景守恒、末端时序、步骤逻辑** 需要单独量纲，参见 [EWMBench](../entities/ewmbench.md)。若目标是开放域 **多场景 + 相机布局** 的世界生成（含 3D/4D），用 [WorldScore](../entities/paper-worldscore.md) 的 Ctrl/Quality/Dynamics。若目标是 **交互干预是否被执行、长程是否持久**，用 [HarnessEval-W](../entities/paper-harnesseval-w.md) 的案例路由技能与证据树。三者轴线不同，勿混读。

### 条件分解：已知静态场景 + 灵巧手轨迹（DWM）

[Dexterous World Models（DWM）](./dwm.md) 面向「已从重建得到**静态 3D 场景**」的设定：沿第一人称相机轨迹渲染**静态场景视频**，再并上同视角**手部网格视频**，用视频扩散预测交互引起的视觉变化；借助**全掩码视频修复**初始化，把「导航一致的外观」当基线、把操纵动力学学成**残差**。与 UniSim 类「从数据中学整套交互模拟器」相比，DWM 更强调**显式冻结 \(\mathbf{S}_0\)** 以减轻背景幻觉，代价是对**上游几何与标定**依赖更强。官方代码已开源（CogVideoX-5B LoRA + WAN 两套实现），工程栈细节见 [DWM](./dwm.md)。

### 工程折中：潜空间世界–动作（示例：Being-H0.7）

若目标是**在线操作控制**而非高保真视频预览，可把「未来结构」压进**紧凑潜变量工作空间**，训练时用未来观测分支对齐、测试时只跑先验动作头，从而保留世界建模的部分收益、避免每步显式像素 rollout。详见 [Being-H0.7](./being-h07.md)。

[mimic-video（Video-Action Model）](./mimic-video.md) 走另一条「**冻结大规模视频扩散骨干**、只训 **流匹配动作解码器**」路线：用骨干在 **潜空间** 里形成与语言一致的 **视觉动力学计划**，动作头充当 **逆动力学**；推理上可用 **部分去噪** 降低完整像素合成的必要性。它与 DWM / Being-H0.7 共享「**别每步滚满分辨率视频也能控**」的工程动机，但 **条件信号来自互联网视频预训练** 而非显式静态场景渲染或 egocentric 潜世界分支。

当讨论把「预测未来」与「输出动作」在**同一策略对象**里联合建模（综述中的 **World Action Models**）时，重点会从**像素逼真度**转向**耦合结构、动作可推断性与闭环延迟**；仓库内总览见 [World Action Models（WAM）](../concepts/world-action-models.md)。

### 全模态 Physical AI 平台（示例：Cosmos 3）

**人形 loco-manip 实时 WAM 实例**：[MotionWAM](../entities/paper-motionwam-humanoid-loco-manipulation-wam.md) 以 **Cosmos-Predict2.5-2B** 系 **Video DiT** 为动力学骨干，在 **固定 flow 步单次前向隐状态** 条件下驱动 Motion DiT，相对完整未来帧去噪实现 **~7×** 推理加速（arXiv:2606.09215）。

**image-goal 导航 WAM 实例**：[NavWAM](../entities/paper-navwam-goal-conditioned-visual-navigation-wam.md) 在 **Cosmos Predict 2（2B）** 上构建 **九帧 latent canvas**，联合去噪未来 egocentric 观测、goal-progress value 与 action chunk；**policy 模式** 单次扩散即可闭环导航，**无需 CEM**（arXiv:2606.13494）。

[NVIDIA Cosmos](../entities/nvidia-cosmos.md) 是该路线的 **厂商平台**：[1.0](../entities/paper-sa-2501-03575-cosmos-world-foundation-model-platform-for-physi.md) 定义 WFM 与五类用法，[Predict2.5](../entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md) 用 flow matching 统一 T2W/I2W/V2W（PAI-Bench I2W Overall **0.810**），[Cosmos Transfer](../entities/cosmos-transfer.md) 用多 ControlNet 做仿真/真机 **world-to-world** 翻译（Transfer1 自适应时空加权，Transfer2.5-2B 更小；配方见 [Cookbook](../entities/cosmos-cookbook.md)），[Cosmos 3](../entities/cosmos-3.md)（arXiv:2606.02800）再把 **语言、图像、视频、音频与动作** 收进单一 **Mixture-of-Transformers**。与 [mimic-video](./mimic-video.md) 依赖 **Cosmos-Predict2 冻结骨干** 或 [Cosmos Policy](../entities/paper-shenlan-wm-11-cosmos-policy.md) 微调 Predict2 的 **单论文实例** 不同，Cosmos 3 是 **开源平台级母栈**（4B Edge / 16B Nano / 64B Super、Diffusers / vLLM-Omni / SGLang / NIM、cosmos-framework SFT）。它与 [Newton](../entities/newton-physics.md) 互补：后者做解析接触，前者做像素世界与合成数据。在 [Sim2Real](../concepts/sim2real.md) 课程语境中，亦常作为 **演示视频增广** 的世界基础模型（见 [NVIDIA SO-101 Sim2Real](../entities/nvidia-so101-sim2real-lab-workflow.md) Strategy 3）。

### Action flow 跨具身 WM + RoboLab 开环评估（示例：Hydra-0）

[Hydra-0](../entities/paper-hydra-0.md)（arXiv:2608.18077，NVIDIA 等）用 **action flow**（相机平面稀疏轨迹+可见性）替代 native 关节/6D 命令，在 **Cosmos 2.5 / Wan2.2** 上 mid-train **2,202 h** 多具身视频；相对 Cosmos 2.5 baseline **robot EPE −90.4%**、**object EPE −60.2%**；RoboLab 五策略开环 replay 与参考成功率 **Pearson r=0.96**；并演示 **object-flow→robot motion→action readout** 逆向控制 POC。截至入库日 **确认未开源**。

### 骨架条件跨具身 WM + 虚拟策略评估（示例：OSCAR）

[OSCAR](../entities/paper-oscar.md)（arXiv:2606.04463）在 **Cosmos-Predict2.5-2B** 上采用 **2D 运动学骨架** 作像素对齐动作条件：经 **四阶段数据管线**（策展→过滤→SigLIP+轨迹去重→字幕）从 216 万源集筛得 18 万训练集，覆盖 **四机器人具身 + 人类 MANO 手**；**单 GH200** 微调即可在开环指标上超越 **14B Kinema4D**。论文进一步在 [RoboArena](../methods/roboarena.md) **七策略池** 上验证：虚拟 rollout 成功率与真机排名 **Pearson ρ +0.750**、MMRV **0.571**——把生成式 WM 从「画面逼真」推进到 **策略评估代理**（对齐 [world-models-route-03-virtual-sandbox](../overview/world-models-route-03-virtual-sandbox.md)）。

### 1-step drifting 动作条件 WM（示例：DriftWorld）

[DriftWorld](../entities/paper-driftworld.md)（arXiv:2607.15065，MIT×Harvard）针对扩散 WM「多步去噪拖垮推理时动作搜索」的瓶颈：训练期学 **action-conditioned drifting field**（可叠 DINOv2/v3 特征空间与运动加权），推理 **单次前向** 从当前观测 + 候选动作生成未来帧，H100 上 **30+ fps**（平均约 **17×** 快于扩散基线）。同骨干 MSE baseline 也是 1-step，但 drifting 在 Push-T 视觉与 **GPC-RANK** IoU 上更优；离线策略评估与 GT 相关性最高约 **0.99**。与 OSCAR 同属「动作条件视频 WM + 虚拟评估」，但卖点是 **搜索/评估时延** 而非跨具身骨架条件。

### 掩码视觉动作统一前向/逆向（示例：Masked Visual Actions）

[Masked Visual Actions](../entities/paper-masked-visual-actions.md)（arXiv:2607.19343，Stanford×UMD×Harvard）把动作写成 **像素空间部分揭示轨迹**：同一 Wan-Fun-Control 14B LoRA 检查点，揭示机器人掩码即 **前向动力学**，揭示物体目标运动即 **逆向行为合成**（训练仅见机器人掩码，物体条件零样本涌现）。约 **15 h** 掩码数据微调后，RoboCasa 策略评估成功率相关 **r=0.982**，并支撑 Best-of-N 规划与 IDM 抽动作；相对 Skeleton / EEF 条件，在未见夹爪与双臂具身上更稳。与 DriftWorld / OSCAR 同属「动作条件视频 WM + 虚拟评估」，卖点是 **条件与视觉先验对齐 + 前向/逆向统一**，而非 1-step 速度或 2D 骨架。

### 多视角可控 VLA 闭环 WM（示例：Ctrl-World）

[Ctrl-World](../entities/paper-ctrl-world.md)（arXiv:2510.10125，ICLR 2026，Stanford×Tsinghua）从 **SVD** 初始化，用 **帧级动作条件 + 位姿记忆检索 + 第三人称/腕部联合预测**，把被动视频生成器改成可与 π₀ / π₀.₅ 等现代 VLA **policy-in-the-loop** 交互的想象环境；DROID 训练后可零样本到新机位，想象指令跟随排名对齐真机，并用合成成功轨迹 SFT 把新指令成功率 **38.7%→83.4%**（约 **+44.7 pt**）。与 MVA 同属「动作条件视频 WM + 虚拟评估」，但条件是 **低维动作/位姿**、强调 **多视角 VLA 接口**，而非像素掩码前向/逆向统一。

### 下一尺度自回归长程 WM + 虚实校准（示例：WALL-SS）

[WALL-SS](../entities/paper-wall-ss.md)（自变量机器人，2026-08-26）把 clip 级扩散换成 **InfinityStar next-scale AR**：观察–动作写成因果序列，粗尺度钉状态转移、细尺度补接触；有界时间–尺度记忆支撑约 **60 s** 流式；on-policy 视觉对齐只优化动作跟随与长程一致性。WorldArena 风格动作跟随 **0.290**（Cosmos3-Nano **0.044**）；**600** 组虚实配对成功率 MAE **0.062**、\(r=0.93\)。相对 Ctrl-World，卖点是 **自回归长程 + 校准协议**，不是合成 SFT；**训练推理代码待发布**。

### 跨本体课程视频 WM + 开源 G1 权重（示例：CLAP）

[CLAP](../entities/paper-clap-cross-embodiment.md)（arXiv:2608.27406，Princeton）同样从 **SVD** 做动作条件视频，但先用 **32-D 潜动作** 吃 OXE + EgoDex，再换成 **7-D 绝对末端** 做零样本规划。相对 Ctrl-World 的 DROID 单本体 + 合成 SFT，CLAP 强调 **跨本体先验可迁移**：同容量追上 DROID SOTA，后训练超过从零单本体，并发布 **`adapt-g1`（26-D）** 与双臂 YAM 权重。推理时对 \(\pi_{0.5}\) / MolmoAct-2 做交叉策略规划，也可用 DSRL 在想象里微调扩散策略。**已开源**（MIT + HF）。

### 跨本体交互模拟器 + Human-in-the-World-Model（示例：CurrentWorld-0）

[CurrentWorld-0](../entities/current-robotics-currentworld.md)（Current Robotics，2026-08 博客）把 WM 定义为 **interactive world simulator**：不统一低层动作空间，按本体保留动作子空间，联合训练人形（BrainCo / Wuji / 夹爪）、移动双臂与桌面双臂；同步预测头戴/腕部/第三人称，并联合力/触觉。产品用法是 **评测层**（自称与真机成功率强相关、保持排名）加上 **失败态保存/回滚/分支的人类接管后训练**（π0 / π0.5 / DP）。相对 Ctrl-World，它把多视角接口扩到跨本体 + 力触觉，并把纠正做成环境内遥操作，而不是只筛成功轨迹做 SFT。**确认未开源**；定量图为官方自报，独立复现前不作硬基准。

### 物理时间 latent ODE（示例：ODEWorld）

[ODEWorld](../entities/paper-odeworld.md)（arXiv:2607.27924，清华 AIR × Berkeley BAIR）把离散 next-step 换成 **Physical-Time Flow**：在冻结 DINOv2 特征上解耦出单 token 动力学 latent，用 JVP 直接监督物理时间速度场，推理靠 RK4 积分。相对 Ctrl-World / DriftWorld 的动作条件像素环，它 **当前版本无动作条件**，主用途是任意时刻/反向视频与 **latent 子目标** 引导策略（LIBERO-LONG 序列子目标 **83.6%**；AgileX+X-VLA **55%→80%**）。推理与 HF 权重已开源，训练脚本未随仓。

### 自一致视频策略评估器（示例：SC3-Eval）

[SC3-Eval](../entities/paper-sc3-eval.md)（arXiv:2606.18610，UToronto×Vector×NVIDIA×π）把 **Cosmos3-Nano + 统一动力学** 改造成真机 VLA 评估沙盒：联合训 **前向/逆向动力学** 与 **跨视角 inpainting**，推理时用逆动力学误差 \(U_{\mathrm{chunk}}\) **早停** off-manifold rollout。七个 π₀.₅ checkpoint 上闭环 Pearson **0.929**、MMRV **0.119**，并复现 language/lifting/placing 失败类别；相对 Ctrl-World / IRASim 等纯前向基线，主卖点是 **自一致防漂移** 而非合成 SFT。截至入库日 **确认未开源**。

### off-expert 动作跟随评测 + 对齐配方（示例：WorldEcho / WorldSync）

[WorldEcho / WorldSync](../entities/paper-worldecho-worldsync.md)（arXiv:2608.24885，北大等）指出：把 AC-WM 当策略模拟器，默认假设「任意合法动作都会被忠实生成」，但现有榜多停在专家演示。**WorldEcho** 用五类查询（专家回放 / 跨状态重放 / 局部扰动 / 策略 rollout / 可行空间采样）联合测 **视觉完整性门控** 与 **\(\mathrm{SE}(3)\) 末端 NDTW**；六套专家训模型在 off-expert 上出现 **视觉崩** 或 **画面好看但不跟命令**。**WorldSync** 用仿真+少量真机覆盖扩展、Action-Forcing Expert 与 Intervention-Effect 配对监督；RoboTwin 50 任务门控误差 **0.0661**，匹配预算两轮改进把仿真倾倒 **~52%→65%**、真机叠杯 **48%→68%**。截至入库日 **确认未开源**。相对 SC3-Eval「自一致评估器」、Ctrl-World「闭环+合成 SFT」，本页主轴是 **动作跟随本身是否成立**。

### 开源视频先验与轨迹可控 I2V（示例：Wan / Wan-Move / Wan-Dancer）

[Wan](../entities/paper-wan-video.md)（arXiv:2503.20314）提供开源 **DiT + Wan-VAE** 视频基础模型族（Wan2.1/2.2）；[Wan-Move](../entities/paper-wan-move.md)（arXiv:2512.08765，NeurIPS 2025）在 **不改 I2V 架构** 的前提下，把点轨迹映射到 latent 并复制首帧特征作运动引导，微调 **Wan-I2V-14B** 达到商用 Motion Brush 级可控性，并发布 **MoveBench**；[Wan-Dancer](../entities/paper-wan-dancer.md)（arXiv:2607.09581）同在 Wan-I2V 上做分层 **music-to-dance**，把连贯生成推到 **分钟级 720p**。三者本身不是操纵 WM，但是 MVA（Wan-Fun-Control）与大量机器人视频 WM / 参考视频先验的 **上游对照**。[DreamX-Phi 1.0](../entities/paper-dreamx-phi.md)（arXiv:2608.13489，阿里 AMAP）把 **Wan2.2-TI2V-5B** 做成动作条件操纵 WM：每臂 **SE(3)** 经 PRoPE-style 编码注入 attention，并加 depth / SAM3 / 冻结 V-JEPA；自报 WorldArena 2.0 Track 1 第一。**权重与推理待赛后**，入库日仓为占位 README。

[Flex-π](../entities/paper-flex-pi.md)（arXiv:2608.10860，UW / AI2）把同一冻结 **Wan-2.2 VAE** 直接用于 **3D pointmap**：论文报 RGB 训练的 VAE 对点图近无损重建（PSNR **31.1 dB**），再与 DINOv3 语义流在 MoT 里联合去噪动作。相对「另训几何编码器」，这是 **借用视频先验吃 3D**；相对 DreamWAM 的训练多视图、部署关分支，Flex-π 把流组合留到 **推理掩码**（action-only ~60 ms ↔ full joint）。**代码待发布**。

### 中间表示条件可控推演（示例：RoboInter-World）

[RoboInter1.5](../entities/paper-robointer-1-5.md)（arXiv:2607.18709）把 **子任务 / 轨迹 / 分割点** 等稠密中间表示渲染为控制视频，条件化未来观测生成（**RoboInter-World**），并与 plan-then-execute VLA 共用同一 IR 脚手架。相对 Masked Visual Actions 的「掩码实体轨迹」，它强调 **多类型 IR 作为双向接口**（既正则化动作，也约束 WM latent）；公开仓目前以 Data/VLM 为主，World 代码待齐。

### 原生 CEDC + 混合线性时序记忆（示例：Kairos）

[Kairos（kairos-agi）](../entities/paper-kairos-native-world-model-stack.md)（arXiv:2606.16533 **v3**）走 **regret-aware「学–维持–跑」** 路线：目标是 **control-sufficient state**（而非全像素仿真），以 **CEDC** 按干预强度从开放视频渐进到人类行为与机器人接地；以 **理解/生成/预测统一 MoT** + **SWA / DSWA / GLA** 维持多时间尺度状态；并以 **DMD+CM 少步蒸馏** 与硬件协同设计追求可嵌入闭环。**Kairos-4B / 3.1** 在 WorldModelBench / DreamGen / PAI-Bench 与 **LIBERO-Plus / RoboTwin 2.0** 报告强竞争力结果，官方仓已迁至 [`kairos-agi/kairos`](https://github.com/kairos-agi/kairos)。与 [Cosmos 3](../entities/cosmos-3.md) 对照，Kairos 更强调 **4B 边缘部署 + 原生 CEDC**；与 [HomeWorld](../entities/paper-homeworld-whole-home-scene-generation.md) **品牌名易混**（后者为静态全屋 3D）。

### 训练期分层物理对齐（示例：PhysisForcing）

[PhysisForcing](../entities/paper-physisforcing.md)（arXiv:2606.28128，PKU × NVIDIA）针对「**重建损失对接触区与背景一视同仁**」的痛点，在 **DiT 微调** 时用 **深度感知运动掩码** 聚焦操纵/接触区域，并联合 **像素级 CoTracker3 轨迹对齐** 与 **语义级 token 关系对齐**（冻结视频理解编码器）。相对 **preference 后训练** 与 **纯几何单点约束**，它把物理合理拆成 **可局部化、可分层、训练期可微** 的两项损失，且 **推理零额外开销**。**PF-Cosmos** 在 **R-Bench** 报告整体最佳 **63.8**；**WorldArena IDM** 闭环 **16.0%→24.0%**；作 **Fast-WAM** 骨干时 **RoboTwin 2.0** 平均 **+4.6%**——说明物理对齐不只服务开环视频榜，也强化下游 WAM 表征。

### Joint 视频–动作 + 测试时想象（示例：τ₀-WM）

[τ₀-World Model（τ0-WM）](../entities/tau0-world-model.md) 在 **5B** 规模上把 **多视角视频扩散** 与 **连续 action chunk** 绑在同一 VAM 表征：动作支路 **逐层 cross-attention** 读视频中间层，使「预测未来」成为控制相关目标；异构 **遥操作 / UMI / 自我中心人视频** 用 **模态掩码** 分监督。推理侧除策略采样外，还提供 **动作条件多视角 rollout + 任务进度轨迹**，并以 **Re-denoising Consistency Score** 与 **propose–evaluate–revise** 把算力花在执行前——与 [mimic-video](./mimic-video.md) 的「冻结骨干 + 潜计划动作头」及 [GE-Sim 2.0](../entities/ge-sim-2.md) 的「独立 World Judge 闭环模拟器」形成同生态对照。

### Dreamer 4 开源交互式游戏 WM（示例：Open Dreamer）

[Open Dreamer](../entities/open-dreamer.md)（next-state，2026-07）复现 [Dreamer 4](https://arxiv.org/abs/2509.24527)：因果 **MAE tokenizer** + **diffusion forcing / flow matching / shortcut** 动作条件潜动力学，面向 Minecraft/VPT；训练仓、Reactor 推理仓、HF Orbax 权重与浏览器 **Game⟷Dream** demo 已开放，完整 BC/RL agent 环仍在 roadmap。相对机器人视频 WM，它是 **游戏域可交互沙盒** 的可复现基线，挂接 [虚拟沙盒路线](../overview/world-models-route-03-virtual-sandbox.md) 与 [Latent Imagination](../concepts/latent-imagination.md) 的 Dreamer 谱系。

### 统一具身合成世界基础模型（示例：Xiaomi-Robotics-U0）

[Xiaomi-Robotics-U0](../entities/xiaomi-robotics-u0.md)（arXiv:2607.11643，小米）在 **38B** **自回归离散 token** 框架内 **继续共训** foundation **T2I/X2I** 与 **多视角具身场景生成、五维结构化迁移、多 FPS 操纵视频**，避免「只后训机器人轨迹」导致的 foundation 遗忘。初始化 **EMU3.5（Qwen3-32B + IBQ）**；**FlashAR+** + **vLLM** 将 1024² 单图延迟压到秒级。**WorldArena #1** 与对 **GPT-Image-2** 的多视角人类评测领先；真机侧用 **零样本 transfer 增广** 将 **π₀.₅** OOD 完成度 **36.9%→63.2%**。与 [Xiaomi-Robotics-0](../entities/xiaomi-robotics-0.md) **VLA** 形成 **「WM 合成数据 → 策略后训练」** 闭环；与 τ₀-WM 的 **Joint 5B WAM** 对照，U0 **不内置动作头**，侧重 **可扩展观测合成与 agentic 轨迹引擎**。

### 多智能体共享世界（示例：Gamma-World）

当环境中有 **多个同时可控主体**（多人游戏、多机编队）时，世界模型除「动作–像素对齐」外，还需 **跨体一致的世界演化** 与 **可扩展的身份编码**。[Gamma-World](../entities/paper-gamma-world-multi-agent.md)（arXiv:2605.28816）用 **Simplex Rotary Agent Encoding**（置换对称、无 slot ID）与 **Sparse Hub Attention**（跨体通信线性于智能体数）扩展交互式视频 WM，并经教师–学生蒸馏实现约 **24 FPS** 流式 rollout；**2 人训练可零样本泛化 4 人**。与单流 [WEM](../entities/paper-wem-world-ego-modeling.md) 的 world/ego 长程分解正交：γ-World 强调 **主体数与实时交互**，而非单机器人导航–操作交错。

### 静态 sim-ready 全屋 3D（示例：HomeWorld）

与 **video rollout** 不同，[HomeWorld](../entities/paper-homeworld-whole-home-scene-generation.md)（arXiv:2606.06390）走 **文本 → 四阶段分层流水线 → sim-ready furnished 全屋 3D** 路线：K-D tree LLM 平面图 + 图像 roaming 软装 + VLM 递归修正 + surface-centric 可操纵小物；强调 **300K 中国住宅矢量平面图** 与 **>15 manipulable objects/scene**。它回答的是 **仿真环境资产从哪来**，而非 **给定动作后下一帧像素长什么样**——与 [Video-as-Simulation](../concepts/video-as-simulation.md) 中 GE-Sim / UniSim 等 **动态** 模拟器互补。

### 学习式无限户外地形（示例：InfiniteDiffusion / Terrain Diffusion）

[InfiniteDiffusion / Terrain Diffusion](../entities/paper-infinite-diffusion-terrain-diffusion.md)（SIGGRAPH 2026，arXiv:2512.08309）走 **扩散模型 + 惰性无界采样** 的 **程序化噪声式接口**：按 **seed + 坐标 O(1)** 查询高程/气候，**training-free** 推广 MultiDiffusion 到无限域；**Terrain Diffusion** 用 **分层扩散 + Laplacian 编码** 覆盖地球尺度垂直动态范围，并开源 **[Minecraft Fabric mod](https://modrinth.com/mod/terrain-diffusion)**。与 HomeWorld 的 **室内 furnished 3D**、上文 **像素视频 WM** 正交：它服务 **开放世界户外几何/气候场**，可作为腿式仿真 [程序化地形](../concepts/procedural-terrain-generation.md) 的高保真资产源，但 **不含接触动力学**，接入 RL 仍需 DR 与碰撞对齐。

### 语言统一动作的具身世界模型（示例：Qwen-RobotWorld）

[Qwen-RobotWorld](../entities/qwen-robot-world.md)（通义 [Qwen-Robot Suite](../entities/qwen-robot-suite.md) 第三件）把 **关节角、方向盘、航向** 等异构控制 **投影到自然语言**，在 **Embodied World Knowledge（8.6M video-text）** 上训练 **60 层双流 MMDiT**（**Qwen2.5-VL** 动作编码 + 视频 latent 生成），联合 **操作 / 驾驶 / 室内导航 / Scene2Robot 人→机** 并输出 **2–4 视角几何一致** 未来视频。与 [WorldVLA / RynnVLA-002](../entities/paper-shenlan-wm-07-worldvla.md) 的 **VLA+WM 单框架** 不同，RobotWorld 侧重 **跨场景语言条件视频物理**；与 Suite 内 [Qwen-RobotManip](../entities/qwen-robot-manip.md) **动作输出** 互补。

### 语言条件 3D 点轨迹预测（示例：MolmoMotion）

[MolmoMotion](../entities/molmo-motion.md)（Ai2，arXiv:2606.18558）走 **「预测 compact 3D 运动结构，而非整段像素视频」** 路线：以 **Molmo 2** 融合 RGB、**2D query 点特征** 与 **动作文本**，预测物体上各点在 **metric 世界坐标** 的未来轨迹（**MolmoMotion-AR** 坐标文本自回归 / **MolmoMotion-FM** 连续 flow matching）。配套 **MolmoMotion-1M**（116 万视频自动 3D 轨迹标注）与 **PointMotionBench**（2.7K 人工校验、ADE 米级误差）。下游上，DROID 微调后的 **MolmoBot** 在 pick-and-place **闭环成功率与样本效率** 显著优于 Molmo 2 初始化；预测轨迹亦可作 **DaS + I2V** 的 motion guidance，使 CogVideoX-5B 等小模型在 motion 指标上逼近更大 Wan2.2。与上文 **像素 rollout** 世界模型互补：轨迹 **更轻、更几何稳定**，但 **不直接给出力/接触**；与 [mimic-video](./mimic-video.md) 共享「**先学动力学结构再控**」动机，但中间表示是 **显式 3D 点** 而非 **视频潜计划**。

### 在线 3D Gaussian 物理速度场（示例：PhysMani）

[PhysMani](../entities/paper-physmani-dynamic-manipulation-world-model.md)（ECCV 2026，arXiv:2607.01938）把 **3D Gaussian Splatting** 从 **内容/渲染管线** 拉回到 **动态操作控制回路**：流式 RGB-D 上 **在线优化无散度 per-Gaussian 速度场**（~**200 ms/帧**），预报 **六维基本速度分量** 再经 **KNN + 可学习 token cross-attention** 注入 **3DFA** 策略。相对 **2D 视频扩散 WM**，强调 **显式 3D 几何 + 物理有意义轨迹**；相对 FreeGave 等离线 3DGS 物理学习，强调 **实时在线** 与 **操纵 SR** 评测（**PhysMani-Bench** 16 任务）。与 [GS-Playground](../entities/gs-playground.md)（仿真训练观测）互补：PhysMani 面向 **真机/仿真闭环动态目标** 而非批量 RL 渲染吞吐。

### 轨迹可控全景 ERP 世界模型（示例：PanoWorld）

[PanoWorld](../entities/paper-panoworld-real-world-panoramic-generation.md)（arXiv:2607.09661，Insta360 Research 等）针对 **360° 等距圆柱（ERP）** 视频世界合成：利用 **rotation-equivariance** 将 **旋转视为几何变换**、**仅显式建模平移**，以 **Dense Panoramic Ray-Conditioning（DPRC）** 把相机平移编码为 **per-ray SE(3) 射线场**，并以 **Geometry-aware Memory Augmentation（GMA）** 在 **同一 PRoPE 流形** 检索历史特征、**confidence 门控** 抑制未观测区幻觉。骨干 **Wan2.2-5B + LoRA** 经 **三阶段** 训练（全景几何 → 纯平移动作 → 记忆锚定）；配套 **World360**（**120k** clip：7 万真实 UAV + 5 万 AirSim360）强调 **multi-altitude 户外物理变化**。相对 **Matrix-3D / OmniRoam**，在 FID 与轨迹 PSNR 上全面领先；**Causal Forcing** 蒸馏可实现 **161 帧 / 8 s** 交互式生成。与上文 **窄 FOV pinhole rollout** 正交：服务 **UAV / 自动驾驶环视** 等需 **全视场一致预测** 的场景，但 **地面人形 egocentric** 覆盖仍有限。

### 产业驾驶栈：多摄仿真 → 加速 → VLA 内嵌（示例：小鹏 X-World 系列）

[X-World](../entities/paper-x-world.md)（arXiv:2603.19979）给出 **动作条件 7 摄** 自车中心视频世界模型，服务端到端智驾可扩展评测；[X-Cache](../entities/paper-x-cache.md)（arXiv:2604.20289）在少步蒸馏后改沿 **跨 chunk** 缓存 DiT block，约 **2.7×** 加速近无损。策略侧 [X-Foresight](../entities/paper-x-foresight.md) 把预测式世界模型嵌进 VLA（chunk-wise 因果 + Renderer），[X-Mind](../entities/paper-x-mind.md) 则把 PWM 压成 **Visual CoT 抽象 sketch** 以上车。规划调度侧 [RISE（酷哇）](../entities/paper-rise-adaptive-imagination-wam.md) 在 Encoder–Predictor–Planner 上按 Future Planning Gain 逐步停想象（NAVSIM；代码+CounterDrive 已开、权重未发）。整条链偏 **级联仿真底座 + 联合策略变体**；小鹏系列截至入库日项目页均 **未开源**，适合读设计对照而非复现。

### 多视角多模态驾驶仿真（示例：M⁴World）

[M⁴World](../entities/paper-m4world.md)（arXiv:2607.14005，美团 × CASIA × BIT）在共享 **DiT** 潜空间上联合生成 **环视视频 + 同步 LiDAR range map**，并把物体条件从几何 box 扩展为 **布局 + SigLIP/文本外观**；经 Teacher Forcing → 4-step ODE → Self-Forcing/DMD → 长视频微调，支撑 **分钟级** 因果流式，并用 VLM judge 评可控性。相对 X-World 的 **动作条件评测底座**，M⁴World 更强调 **物体级交互操纵与相机–LiDAR 多模态**；截至入库日 **未开源**。

### 单卡桌面实时交互世界（示例：ABot-World-0）

[ABot-World-0](../entities/paper-abot-world-0.md)（arXiv:2607.19191，高德 AMAP CV Lab）把交互式视频世界模型写成 **数据闭环 + 双向→因果蒸馏 + 全栈流式共设计**：WorldExplorer 多源采集（游戏 / 仿真 / 互联网）、原始键盘统一漫游与第三人称角色、**LongForcing** 对齐长程学生自 rollout，并在 **单卡 RTX 5090** 上把 **720P** 推到最高约 **16 FPS**（首帧 **1.2 s**，峰值约 **19 GiB**）。相对 M⁴World 的驾驶多传感器仿真与 Open Dreamer 的游戏潜动力学，它更强调 **消费级实时像素世界可玩性**；**推理与 5B 因果学生已开源**，教师权重与约 500 h 数据集仍待发布。

### 相邻方向：三维世界生成与流式 3DGS（产业样本）

部分团队将「世界模型」叙事延伸到 **持久 3D 世界** 的生成与编辑，并以 **3D Gaussian Splatting** 在 Web 或工具链中交付可漫游场景；这与上文以 **像素视频 rollout** 为中心的讨论共享「生成式环境」动机，但 **评测对象与训练目标** 往往更接近内容管线而非机器人控制回路。产业侧公开样本见 [World Labs](../entities/world-labs.md)：**[Marble](../entities/marble-world-model.md)**（2025-11 GA：文/图/视频/Chisel → 可导出 splat/mesh，模型闭源 + World API；[文档](https://docs.worldlabs.ai/)）+ **[Atlas](../entities/atlas-world-model.md)**（2026-09 omni 底座：相机可控长视频、稀疏重建、Real-to-Sim 传感器 rollout；早期访问未开源）+ [Spark](../entities/spark-3dgs-renderer.md)；同类 Web 渲染可对照 [Aholo Viewer](../entities/aholo-viewer.md)（见 [Spark vs Aholo](../comparisons/spark-vs-aholo-web-3dgs-renderers.md)）。驾驶侧的对照是 [Instant NuRec](../entities/paper-instant-nurec.md)：它 **不发明像素**，而是一次前向给出可重姿态的分层 3DGS，再交给 [NuRec](../entities/nvidia-nurec.md) / AlpaSim——论文把它放在「重建骨干」，与生成修复 / 联合世界模型互补。

### 术语对照：状态动力学「世界模型」（RWM）

足式控制与 MBRL 文献里也会出现 *Robotic World Model* 指 **学习的前向动力学 + 想象 rollout**（例如 ETH RSL 的 **RWM / RWM-U**：集成 RNN 预测 **状态与特权量**，而非扩散视频）。这与本页以 **像素 / Token 视频** 为中心的生成式世界模型 **共享「预测未来」动机**，但 **观测空间、训练目标与评测口径** 不同；工程入口与双仓分工见 [Robotic World Model（ETH RSL）](../entities/robotic-world-model-eth-rsl.md)。

## 主要技术路线
- **视频即仿真 (Video-as-Simulation)**：利用交互式视频预测器代替解析引擎，详见 [Video-as-Simulation](../concepts/video-as-simulation.md)。
- **扩散模型 (Diffusion-based)**：利用 DDPM 逐步去噪生成未来帧，代表：UniSim。
- **离散 Token 流 (Discrete Token flow)**：将图像量化为 Token，利用 Transformer 预测序列，代表：π₀ 的动作建模部分。
- **生成视频作为人形控制 demo 源**：把第三人称视频生成当成"想象出来的示教"，再用动作估计 + 通用动作跟踪把视频翻译为机器人动作，代表：[ExoActor](./exoactor.md)。

## 关联页面
- [Visual General Intelligence 白皮书](../entities/paper-vgi-white-paper.md) — 生成视频可当 VFM，但观感保真 ≠ 可干预的物理结构；VGI 议程把本页方法族放进视觉通向 AGI 的坐标系
- [Query：具身大模型分类学选型闭环知识链](../queries/embodied-fm-taxonomy-loop.md) — 生成式世界模型是五层选型闭环 **⑤ 世界模型推演层** 的 **级联预演** 范式（VLA 出候选 → WM 逐帧推演择优 → 真机执行），与 WAM 的「联合建模」范式并列，注意推演步长↑累积误差↑
- [Latent Imagination (潜空间想象)](../concepts/latent-imagination.md)
- [WCM](../entities/paper-wcm-world-critic-model.md) — JEPA 隐空间预测路线：不生成像素，只用预测目标监督 critic 表征（靠 SIGReg 防坍塌）
- [LeJEPA](../entities/paper-lejepa.md) — SIGReg 图像配方；后续规划/视频 JEPA 的防坍塌起点
- [LeWM](../entities/paper-lewm.md) — 像素端到端动作条件 JEPA，规划至 48× 快于 DINO-WM
- [LpWM](../entities/paper-lpwm.md) — 稀疏非负码；PushT 中等预测器相对稠密 LeWM 最高 +57 pp
- [LeVJEPA](../entities/paper-levjepa.md) — 把 LeJEPA+SIGReg 接到视频编码器：不要 EMA teacher / predictor / 像素重建；因果表征免费，但本文不做规划 WM
- [ActEffect](../entities/paper-phi-wm-acteffect.md) — 训练时后果反馈，部署卸 WM（光象技术报告；未开源）
- [Model-Based RL](../methods/model-based-rl.md)
- [Being-H0.7](./being-h07.md) — 潜空间世界–动作模型，测试时不滚未来像素。
- [World Action Models（WAM）](../concepts/world-action-models.md) — 世界预测与动作生成的联合范式与文献taxonomy
- [NVIDIA Cosmos](../entities/nvidia-cosmos.md) — 1.0 / Predict2.5 / Cosmos 3 平台与 Newton 分工
- [Cosmos Transfer](../entities/cosmos-transfer.md) — 多控 world-to-world；Sim2Real 合成数据
- [Cosmos Cookbook](../entities/cosmos-cookbook.md) — 2.x 可运行配方
- [Instant NuRec](../entities/paper-instant-nurec.md) — 驾驶日志前向 3DGS（显式世界，不是像素 WM）
- [NVIDIA Omniverse NuRec](../entities/nvidia-nurec.md) — 神经体积 USDZ 与 Instant 初始化
- [统一机器人学习综述](../entities/paper-unified-robot-learning-survey.md) — 把 WM 写成三轴之一，用耦合类型诊断长程/不确定失败（TMLR 2026）
- [GaussianDream++](../entities/paper-gaussiandream-plusplus.md) — 训练期高斯世界、部署 20 令牌
- [ConfAL-WM](../entities/paper-confal-wm.md) — 稠密置信度主动后训练
- [EmbodiedVAE](../entities/paper-embodiedvae.md) — 为操作视频世界模型解耦臂/背景的 video VAE
- [NVIDIA Omniverse](../entities/nvidia-omniverse.md)
- [ExoActor](./exoactor.md) — 视频生成驱动的交互式人形控制。
- [EWMBench](../entities/ewmbench.md) — 具身视频世界模型生成质量的多维基准与开源工具链。
- [WorldScore](../entities/paper-worldscore.md) — 3D/4D/I2V/T2V **多场景世界生成** 统一评测（ICCV 2025；HF 活榜）。
- [HarnessEval-W](../entities/paper-harnesseval-w.md) — 交互式世界 **agentic** 评测：干预/持久证据树，330 例 × 18 模型（arXiv:2608.16859；评测代码已开源）。
- [GE-Sim 2.0](../entities/ge-sim-2.md) — Agibot **闭环** 操纵视频世界模拟器：本体状态专家 + World Judge + 加速 rollout（arXiv:2605.27491）。
- [Cosmos 3](../entities/cosmos-3.md) — NVIDIA **全模态 MoT 世界模型平台**：Reasoner + Generator 双路径，覆盖 VLM、视频生成、policy 与正/逆动力学（arXiv:2606.02800）。
- [Kairos（原生世界–动作模型栈）](../entities/paper-kairos-native-world-model-stack.md) — **regret-aware CEDC + SWA/DSWA/GLA + 4B/3.1 部署导向 WAM**（arXiv:2606.16533 v3，[kairos-agi/kairos](https://github.com/kairos-agi/kairos)）。
- [PhysMani](../entities/paper-physmani-dynamic-manipulation-world-model.md) — **在线 3D Gaussian 无散度速度场 WM + 3DFA 动态操作**；PhysMani-Bench 16 任务（arXiv:2607.01938，ECCV 2026）。
- [PanoWorld](../entities/paper-panoworld-real-world-panoramic-generation.md) — **ERP 轨迹可控全景 WM**：DPRC 射线动作 + GMA 几何记忆 + World360 数据集（arXiv:2607.09661）。
- [PhysisForcing](../entities/paper-physisforcing.md) — **训练期区域聚焦分层物理对齐**（像素轨迹 + 语义关系）；Wan/Cosmos 跨骨干，R-Bench SOTA 与 WorldArena / Fast-WAM 下游增益（arXiv:2606.28128）。
- [Flex-π](../entities/paper-flex-pi.md) — **冻结 Wan VAE 共享编码 RGB+pointmap** 的多流 Joint WAM；部署算力柔性（arXiv:2608.10860；代码待发布）。
- [Hydra-0](../entities/paper-hydra-0.md) — **action flow** 跨具身 WM 条件 + RoboLab 开环 **r=0.96**；逆向 object-flow 控制 POC（arXiv:2608.18077；未开源）。
- [GigaBrain-WBC-0.5](../entities/paper-gigabrain-wbc-0-5.md) — 人形 **行为世界模型（BWM）** 低层全身控制 + 地形/跌倒 OOD filter（arXiv:2608.18234；代码 coming soon）。
- [LT-Mem](../entities/paper-lt-mem.md) — **波动性感知** Live/Delta/Meta 长期场景记忆 + LT-VQA（arXiv:2608.19059；数据集可下）。
- [OSCAR](../entities/paper-oscar.md) — **2D 骨架跨具身动作条件** + 大规模数据管线；**2B Cosmos-Predict2.5** 微调，RoboArena 虚拟策略评测与真机强相关（arXiv:2606.04463）。
- [DriftWorld](../entities/paper-driftworld.md) — **1-step drifting** 动作条件 WM：推理时搜索 + 离线评估（arXiv:2607.15065）。
- [Masked Visual Actions](../entities/paper-masked-visual-actions.md) — **像素掩码轨迹** 统一前向/逆向；RoboCasa 策略评估 **r=0.982**（arXiv:2607.19343）。
- [Ctrl-World](../entities/paper-ctrl-world.md) — **多视角** 可控 WM：VLA 闭环评估 + 合成 SFT（ICLR 2026）。
- [CLAP](../entities/paper-clap-cross-embodiment.md) — **跨本体** LAM→EE 课程 + 开源 G1/YAM 适配权重（arXiv:2608.27406）。
- [WALL-SS](../entities/paper-wall-ss.md) — **下一尺度自回归** 长程 WM：60 s 流式 + 虚实成功率校准 \(r=0.93\)（训练代码待发布）。
- [CurrentWorld-0](../entities/current-robotics-currentworld.md) — 跨本体 / 多视角 / 力触觉 **交互模拟器** + Human-in-the-World-Model 后训练（2026-08 博客；确认未开源）。
- [ODEWorld](../entities/paper-odeworld.md) — **物理时间 latent ODE**：任意时刻/反向预测 + 子目标策略（arXiv:2607.27924）。
- [SC3-Eval](../entities/paper-sc3-eval.md) — **自一致** 视频策略评估器：前向–逆向 + 跨视角 + 早停；闭环 \(r=0.929\)（arXiv:2606.18610；确认未开源）。
- [WorldEcho / WorldSync](../entities/paper-worldecho-worldsync.md) — **off-expert 动作跟随** 评测 + AFE/IE 对齐配方（arXiv:2608.24885；确认未开源）。
- [World Action Planner](../entities/paper-world-action-planner.md) — **pose-image** 条件多视角 WM + VLM 想象规划（arXiv:2607.27599；代码/权重已开源）。
- [Rofacto](../entities/paper-rofacto.md) — **名义轨迹 + URDF 渲染** 动作接口；相对向量条件提升场景响应（arXiv:2607.22535）。
- [ViTacWorld](../entities/paper-vitacworld.md) — **视触觉** 动作条件 WM：dream 数据增强 + 策略评估（arXiv:2607.22530）。
- [Wan](../entities/paper-wan-video.md) / [Wan-Move](../entities/paper-wan-move.md) / [Wan-Dancer](../entities/paper-wan-dancer.md) — 开源视频基础模型、轨迹运动控制与分钟级 music-to-dance。
- [RoboInter1.5 / RoboInter-World](../entities/paper-robointer-1-5.md) — **IR 控制视频** 条件世界模型 + VLA 套件（arXiv:2607.18709）。
- [τ₀-World Model（τ0-WM）](../entities/tau0-world-model.md) — Agibot **5B 统一视频–动作世界模型**：异构掩码预训练 + 测试时 propose–evaluate–revise（技术报告 2026-05-31）。
- [Xiaomi-Robotics-U0](../entities/xiaomi-robotics-u0.md) — 小米 **38B 统一具身合成世界基础模型**：foundation T2I/X2I 与多视角场景/迁移/视频共训 + FlashAR+ 加速（arXiv:2607.11643）。
- [WEM（World-Ego Model）](../entities/paper-wem-world-ego-modeling.md) — **world/ego 显式解耦** 的长程混合导航–操作视频 rollout 与 **HTEWorld** 基准（arXiv:2605.19957）。
- [Gamma-World](../entities/paper-gamma-world-multi-agent.md) — **多智能体** 置换对称编码 + hub 注意力 + 24 FPS 交互 rollout（arXiv:2605.28816）。
- [HomeWorld](../entities/paper-homeworld-whole-home-scene-generation.md) — **静态 sim-ready 全屋 3D** 场景生成与中文住宅平面图数据（arXiv:2606.06390）。
- [InfiniteDiffusion / Terrain Diffusion](../entities/paper-infinite-diffusion-terrain-diffusion.md) — **学习式无限户外地形**（惰性扩散 + 分层高程/气候场；Minecraft mod 集成，SIGGRAPH 2026）。
- [Robotic World Model（ETH RSL）](../entities/robotic-world-model-eth-rsl.md) — 状态空间神经动力学 + 想象 rollout（与像素生成式 WBM 对照）。
- [世界模型功能分类（Renderer / Simulator / Planner）](../concepts/functional-taxonomy-world-models.md) — 先问输出是观测、状态还是动作
- [世界模型定义与路线图](../entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md) — 压缩定义 + 功能×架构二维表（arXiv:2607.06401）
- [World Labs](../entities/world-labs.md) — 空间智能与 3D 世界生成产品侧样本（Atlas / Marble / Spark）。
- [Marble（World Labs 多模态世界模型）](../entities/marble-world-model.md) — 可注册产品 + 文档/API；生成闭源，Spark 开源。
- [Atlas（World Labs omni 世界模型）](../entities/atlas-world-model.md) — 相机可控生成、稀疏 3D 重建、Real-to-Sim；早期访问。
- [Spark（Web 3DGS）](../entities/spark-3dgs-renderer.md) — LoD splat 树、.RAD 流式与 splat 分页（Spark 2.0）。
- [Aholo Viewer](../entities/aholo-viewer.md) — Chunked Streaming LoD + 3DGS/Mesh 混渲。
- [DWM（Dexterous World Models）](./dwm.md) — 已知静态 3D 场景上的场景–手条件视频扩散与残差动力学学习。
- [INTACT](../entities/paper-intact.md) — 同构意图→动作无搜索 JEPA（相对 LeWM+CEM；文档仓 Coming Soon）。
- [mimic-video（Video-Action Model）](./mimic-video.md) — 互联网视频骨干潜计划 + 流匹配动作解码器的操作策略。
- [MolmoMotion](../entities/molmo-motion.md) — 语言条件 **3D 点轨迹** 预测 + MolmoMotion-1M / PointMotionBench（arXiv:2606.18558）。
- [X-World](../entities/paper-x-world.md) — 小鹏 **7 摄动作条件** 驾驶世界模型（arXiv:2603.19979）。
- [X-Cache](../entities/paper-x-cache.md) — 少步 AR 世界模型 **跨 chunk** 推理加速（arXiv:2604.20289）。
- [X-Foresight](../entities/paper-x-foresight.md) — 驾驶 VLA **内嵌** 长视界预测式世界建模（arXiv:2605.24892）。
- [RISE（酷哇 · 驾驶 WAM 自适应想象）](../entities/paper-rise-adaptive-imagination-wam.md) — 测试时按规划增益停 latent rollout（arXiv:2608.20430；代码+CounterDrive 已开，权重未发）。
- [X-Mind](../entities/paper-x-mind.md) — Visual CoT + 压缩 sketch / RBD 的车载高效变体（arXiv:2606.28758）。
- [M⁴World](../entities/paper-m4world.md) — 美团等 **多视角多模态** 驾驶 WM：物体外观控制 + 分钟级流式（arXiv:2607.14005）。
- [ABot-World-0](../entities/paper-abot-world-0.md) — 高德 **单卡桌面** 键盘交互视频 WM：LongForcing + 720P 实时流式（arXiv:2607.19191；部分开源）。

## 参考来源
- [机器人论文阅读笔记：Generative World Modelling for Humanoids](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Generative_World_Modelling_for_Humanoids__1X_World_Model_Challenge_Technical_Report/Generative_World_Modelling_for_Humanoids__1X_World_Model_Challenge_Technical_Report.html)
- Hu, A., et al. (2023). *GAIA-1: A Generative AI for Embodied AI*.
- Yang, S., et al. (2023). *Learning Interactive Real-World Simulators (UniSim)*.
- Zhou Y., et al. (2026). *ExoActor: Exocentric Video Generation as Generalizable Interactive Humanoid Control* — 见 [sources/papers/exoactor.md](../../sources/papers/exoactor.md)。
- Luo, H., et al. (2026). *Being-H0.7: A Latent World-Action Model from Egocentric Videos* — 见 [sources/papers/being_h07.md](../../sources/papers/being_h07.md)。
- Wang, S., et al. (2026). *World Action Models: The Next Frontier in Embodied AI* — 见 [sources/papers/world_action_models_survey_2605.md](../../sources/papers/world_action_models_survey_2605.md)。
- Hu, Y., et al. (2025). *EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models* — 见 [sources/papers/ewmbench.md](../../sources/papers/ewmbench.md)。
- Duan, H., et al. (2025). *WorldScore: A Unified Evaluation Benchmark for World Generation* — 见 [sources/papers/worldscore_arxiv_2504_00983.md](../../sources/papers/worldscore_arxiv_2504_00983.md)。
- Fei-Fei Li / World Labs (2026). *A Functional Taxonomy of World Models* — 见 [worldlabs_functional_taxonomy_world_models.md](../../sources/blogs/worldlabs_functional_taxonomy_world_models.md)。
- Physical Intelligence Team, Shanghai AI Lab (2026). *A Definition and Roadmap for World Models* — 见 [world_model_definition_roadmap_arxiv_2607_06401.md](../../sources/papers/world_model_definition_roadmap_arxiv_2607_06401.md)。
- World Labs 官方站点与 Spark/Marble 关联归档 — 见 [sources/sites/worldlabs-ai.md](../../sources/sites/worldlabs-ai.md)。
- Marble 文档与 GA 博客 — 见 [worldlabs-docs.md](../../sources/sites/worldlabs-docs.md)、[worldlabs_marble_world_model.md](../../sources/blogs/worldlabs_marble_world_model.md)。
- Atlas 技术博客归档 — 见 [sources/blogs/worldlabs_atlas_omni_world_model.md](../../sources/blogs/worldlabs_atlas_omni_world_model.md)。
- Spark 2.0 技术博客归档 — 见 [sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md](../../sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md)。
- Kim, B., et al. (2026). *Dexterous World Models* — 见 [sources/papers/dwm_arxiv_2512_17907.md](../../sources/papers/dwm_arxiv_2512_17907.md)。
- Pai, J., et al. (2025). *mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs* — 见 [sources/papers/mimic_video_arxiv_2512_15692.md](../../sources/papers/mimic_video_arxiv_2512_15692.md)。
- Lin, Z., et al. (2026). *World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks* — 见 [sources/papers/wem_arxiv_2605_19957.md](../../sources/papers/wem_arxiv_2605_19957.md)。
- Liu, F., et al. (2026). *Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players* — 见 [sources/papers/gamma_world_arxiv_2605_28816.md](../../sources/papers/gamma_world_arxiv_2605_28816.md)。
- Zhang, J., et al. (2026). *MolmoMotion: Forecasting Point Trajectories in 3D with Language Instruction* — 见 [sources/blogs/allenai_molmo_motion.md](../../sources/blogs/allenai_molmo_motion.md)。
- Qiu, B., et al. (2026). *GE-Sim 2.0: A Roadmap Towards Comprehensive Closed-loop Video World Simulators for Robotic Manipulation* — 见 [sources/papers/ge_sim_2_arxiv_2605_27491.md](../../sources/papers/ge_sim_2_arxiv_2605_27491.md)。
- Zheng, C., et al. (2026). *X-World: Controllable Ego-Centric Multi-Camera World Models for Scalable End-to-End Driving* — 见 [sources/papers/x_world_arxiv_2603_19979.md](../../sources/papers/x_world_arxiv_2603_19979.md)。
- Zeng, Y., et al. (2026). *X-Cache: Cross-Chunk Block Caching for Few-Step Autoregressive World Models Inference* — 见 [sources/papers/x_cache_arxiv_2604_20289.md](../../sources/papers/x_cache_arxiv_2604_20289.md)。
- Li, B., et al. (2026). *X-Foresight: A Joint Vision-Action Causal Forecasting Network via Predictive World Modeling* — 见 [sources/papers/x_foresight_arxiv_2605_24892.md](../../sources/papers/x_foresight_arxiv_2605_24892.md)。
- Zhao, B., et al. (2026). *X-Mind: Efficient Visual Chain-of-Thought via Predictive World Model for End-to-End Driving* — 见 [sources/papers/x_mind_arxiv_2606_28758.md](../../sources/papers/x_mind_arxiv_2606_28758.md)。
- Zayer, H., et al. (2026). *Masked Visual Actions* — 见 [sources/papers/masked_visual_actions_arxiv_2607_19343.md](../../sources/papers/masked_visual_actions_arxiv_2607_19343.md)。
- Guo, Y., et al. (2026). *Ctrl-World* — 见 [sources/papers/ctrl_world_arxiv_2510_10125.md](../../sources/papers/ctrl_world_arxiv_2510_10125.md)。
- X Square Robot Team (2026). *WALL-SS* — 见 [sources/papers/wall_ss_x_square_2026.md](../../sources/papers/wall_ss_x_square_2026.md)。
- Current Robotics Team (2026). *CurrentWorld-0* — 见 [sources/blogs/current_robotics_currentworld.md](../../sources/blogs/current_robotics_currentworld.md)。
- Liu, D., Niu, H., et al. (2026). *ODEWorld* — 见 [sources/papers/odeworld_arxiv_2607_27924.md](../../sources/papers/odeworld_arxiv_2607_27924.md)。
- Chen, S., et al. (2026). *Do Robotic World Models Really Follow Actions?* — 见 [sources/papers/worldecho_worldsync_arxiv_2608_24885.md](../../sources/papers/worldecho_worldsync_arxiv_2608_24885.md)。
- Zhang, X., & Du, Y. (2026). *World Action Planner* — 见 [sources/papers/world_action_planner_arxiv_2607_27599.md](../../sources/papers/world_action_planner_arxiv_2607_27599.md)。
- Chu, R., et al. (2025). *Wan-Move* — 见 [sources/papers/wan_move_arxiv_2512_08765.md](../../sources/papers/wan_move_arxiv_2512_08765.md)。
- Huang, M., et al. (2026). *Wan-Dancer* — 见 [sources/papers/wan_dancer_arxiv_2607_09581.md](../../sources/papers/wan_dancer_arxiv_2607_09581.md)。
- Wan Team (2025). *Wan* — 见 [sources/papers/wan_video_arxiv_2503_20314.md](../../sources/papers/wan_video_arxiv_2503_20314.md)。
- Cheng, K., et al. (2026). *M⁴World: A Multi-view Multimodal Driving World Model for Interactive Object Manipulation and Minute-long Streaming* — 见 [sources/papers/m4world_arxiv_2607_14005.md](../../sources/papers/m4world_arxiv_2607_14005.md)。
- Jiang, F., et al. (2026). *ABot-World-0: Infinite Interactive World Rollout on a Single Desktop GPU* — 见 [sources/papers/abot_world_0_arxiv_2607_19191.md](../../sources/papers/abot_world_0_arxiv_2607_19191.md)。
