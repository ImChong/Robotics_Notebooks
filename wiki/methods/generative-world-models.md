---
type: method
tags: [world-models, generative-ai, simulation, video-generation, driving]
status: complete
updated: 2026-06-14
related:
  - ../entities/paper-motionwam-humanoid-loco-manipulation-wam.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../concepts/humanoid-policy-network-architecture.md
  - ../concepts/latent-imagination.md
  - ../concepts/world-action-models.md
  - ../methods/model-based-rl.md
  - ../methods/being-h07.md
  - ../entities/nvidia-omniverse.md
  - ../entities/ewmbench.md
  - ../entities/paper-wem-world-ego-modeling.md
  - ../entities/paper-gamma-world-multi-agent.md
  - ../entities/paper-homeworld-whole-home-scene-generation.md
  - ../entities/tau0-world-model.md
  - ../entities/cosmos-3.md
  - ../entities/robotic-world-model-eth-rsl.md
  - ../entities/world-labs.md
  - ./dwm.md
  - ./mimic-video.md
sources:
  - ../../sources/papers/wm_robot_survey_arxiv_2605_00080.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/exoactor.md
  - ../../sources/papers/being_h07.md
  - ../../sources/papers/world_action_models_survey_2605.md
  - ../../sources/papers/ewmbench.md
  - ../../sources/papers/dwm_arxiv_2512_17907.md
  - ../../sources/papers/mimic_video_arxiv_2512_15692.md
  - ../../sources/papers/wem_arxiv_2605_19957.md
  - ../../sources/papers/gamma_world_arxiv_2605_28816.md
  - ../../sources/sites/worldlabs-ai.md
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

### 主要架构
1. **视频生成器 (Video Diffusion/Autoregressive)**：如 GAIA-1 或 UniSim。给定当前画面和动作序列，生成一段长达数秒甚至数分钟的未来预测视频。
2. **反事实推演 (Counterfactual Reasoning)**：允许用户输入“如果没有躲避障碍物会怎样？”，模型会生成相应的碰撞视频，作为强化学习的负样本。

## 典型代表作

### 1. GAIA-1 (Wayve)
针对自动驾驶设计的世界模型。它不仅能生成真实的驾驶场景，还能根据文本描述（如“突然下起大雨”）动态改变天气和光影。

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
- **评测口径漂移**：通用「文生视频」基准往往强调美学与粗粒度语义；面向操纵的 **场景守恒、末端时序、步骤逻辑** 需要单独量纲，参见 [EWMBench](../entities/ewmbench.md)。

### 条件分解：已知静态场景 + 灵巧手轨迹（DWM）

[Dexterous World Models（DWM）](./dwm.md) 面向「已从重建得到**静态 3D 场景**」的设定：沿第一人称相机轨迹渲染**静态场景视频**，再并上同视角**手部网格视频**，用视频扩散预测交互引起的视觉变化；借助**全掩码视频修复**初始化，把「导航一致的外观」当基线、把操纵动力学学成**残差**。与 UniSim 类「从数据中学整套交互模拟器」相比，DWM 更强调**显式冻结 \(\mathbf{S}_0\)** 以减轻背景幻觉，代价是对**上游几何与标定**依赖更强。官方代码已开源（CogVideoX-5B LoRA + WAN 两套实现），工程栈细节见 [DWM](./dwm.md)。

### 工程折中：潜空间世界–动作（示例：Being-H0.7）

若目标是**在线操作控制**而非高保真视频预览，可把「未来结构」压进**紧凑潜变量工作空间**，训练时用未来观测分支对齐、测试时只跑先验动作头，从而保留世界建模的部分收益、避免每步显式像素 rollout。详见 [Being-H0.7](./being-h07.md)。

[mimic-video（Video-Action Model）](./mimic-video.md) 走另一条「**冻结大规模视频扩散骨干**、只训 **流匹配动作解码器**」路线：用骨干在 **潜空间** 里形成与语言一致的 **视觉动力学计划**，动作头充当 **逆动力学**；推理上可用 **部分去噪** 降低完整像素合成的必要性。它与 DWM / Being-H0.7 共享「**别每步滚满分辨率视频也能控**」的工程动机，但 **条件信号来自互联网视频预训练** 而非显式静态场景渲染或 egocentric 潜世界分支。

当讨论把「预测未来」与「输出动作」在**同一策略对象**里联合建模（综述中的 **World Action Models**）时，重点会从**像素逼真度**转向**耦合结构、动作可推断性与闭环延迟**；仓库内总览见 [World Action Models（WAM）](../concepts/world-action-models.md)。

### 全模态 Physical AI 平台（示例：Cosmos 3）

**人形 loco-manip 实时 WAM 实例**：[MotionWAM](../entities/paper-motionwam-humanoid-loco-manipulation-wam.md) 以 **Cosmos-Predict2.5-2B** 系 **Video DiT** 为动力学骨干，在 **固定 flow 步单次前向隐状态** 条件下驱动 Motion DiT，相对完整未来帧去噪实现 **~7×** 推理加速（arXiv:2606.09215）。

[NVIDIA Cosmos 3](../entities/cosmos-3.md)（arXiv:2606.02800）把 **语言、图像、视频、音频与动作** 收进单一 **Mixture-of-Transformers**：**Reasoner** 路径（因果 AR）承担 VLM 式物理推理与 2D 轨迹 CoT，**Generator** 路径（扩散 DM）承担 T2I/T2V/I2V、带声视频、**policy** 与 **正/逆动力学** rollout。与 [mimic-video](./mimic-video.md) 依赖 **Cosmos-Predict2 冻结骨干** 或 [Cosmos Policy](../entities/paper-shenlan-wm-11-cosmos-policy.md) 微调 Predict2 的 **单论文实例** 不同，Cosmos 3 是 **开源平台级母栈**（16B Nano / 64B Super、Diffusers / vLLM-Omni / NIM、OpenMDW-1.1 权重与合成数据）。在 [Sim2Real](../concepts/sim2real.md) 课程语境中，亦常作为 **演示视频增广** 的世界基础模型（见 [NVIDIA SO-101 Sim2Real](../entities/nvidia-so101-sim2real-lab-workflow.md) Strategy 3）。

### Joint 视频–动作 + 测试时想象（示例：τ₀-WM）

[τ₀-World Model（τ0-WM）](../entities/tau0-world-model.md) 在 **5B** 规模上把 **多视角视频扩散** 与 **连续 action chunk** 绑在同一 VAM 表征：动作支路 **逐层 cross-attention** 读视频中间层，使「预测未来」成为控制相关目标；异构 **遥操作 / UMI / 自我中心人视频** 用 **模态掩码** 分监督。推理侧除策略采样外，还提供 **动作条件多视角 rollout + 任务进度轨迹**，并以 **Re-denoising Consistency Score** 与 **propose–evaluate–revise** 把算力花在执行前——与 [mimic-video](./mimic-video.md) 的「冻结骨干 + 潜计划动作头」及 [GE-Sim 2.0](../entities/ge-sim-2.md) 的「独立 World Judge 闭环模拟器」形成同生态对照。

### 多智能体共享世界（示例：Gamma-World）

当环境中有 **多个同时可控主体**（多人游戏、多机编队）时，世界模型除「动作–像素对齐」外，还需 **跨体一致的世界演化** 与 **可扩展的身份编码**。[Gamma-World](../entities/paper-gamma-world-multi-agent.md)（arXiv:2605.28816）用 **Simplex Rotary Agent Encoding**（置换对称、无 slot ID）与 **Sparse Hub Attention**（跨体通信线性于智能体数）扩展交互式视频 WM，并经教师–学生蒸馏实现约 **24 FPS** 流式 rollout；**2 人训练可零样本泛化 4 人**。与单流 [WEM](../entities/paper-wem-world-ego-modeling.md) 的 world/ego 长程分解正交：γ-World 强调 **主体数与实时交互**，而非单机器人导航–操作交错。

### 静态 sim-ready 全屋 3D（示例：HomeWorld）

与 **video rollout** 不同，[HomeWorld](../entities/paper-homeworld-whole-home-scene-generation.md)（arXiv:2606.06390）走 **文本 → 四阶段分层流水线 → sim-ready furnished 全屋 3D** 路线：K-D tree LLM 平面图 + 图像 roaming 软装 + VLM 递归修正 + surface-centric 可操纵小物；强调 **300K 中国住宅矢量平面图** 与 **>15 manipulable objects/scene**。它回答的是 **仿真环境资产从哪来**，而非 **给定动作后下一帧像素长什么样**——与 [Video-as-Simulation](../concepts/video-as-simulation.md) 中 GE-Sim / UniSim 等 **动态** 模拟器互补。

### 语言统一动作的具身世界模型（示例：Qwen-RobotWorld）

[Qwen-RobotWorld](../entities/qwen-robot-world.md)（通义 [Qwen-Robot Suite](../entities/qwen-robot-suite.md) 第三件）把 **关节角、方向盘、航向** 等异构控制 **投影到自然语言**，在 **Embodied World Knowledge（8.6M video-text）** 上训练 **60 层双流 MMDiT**（**Qwen2.5-VL** 动作编码 + 视频 latent 生成），联合 **操作 / 驾驶 / 室内导航 / Scene2Robot 人→机** 并输出 **2–4 视角几何一致** 未来视频。与 [WorldVLA / RynnVLA-002](../entities/paper-shenlan-wm-07-worldvla.md) 的 **VLA+WM 单框架** 不同，RobotWorld 侧重 **跨场景语言条件视频物理**；与 Suite 内 [Qwen-RobotManip](../entities/qwen-robot-manip.md) **动作输出** 互补。

### 相邻方向：三维世界生成与流式 3DGS（产业样本）

部分团队将「世界模型」叙事延伸到 **持久 3D 世界** 的生成与编辑，并以 **3D Gaussian Splatting** 在 Web 或工具链中交付可漫游场景；这与上文以 **像素视频 rollout** 为中心的讨论共享「生成式环境」动机，但 **评测对象与训练目标** 往往更接近内容管线而非机器人控制回路。产业侧公开样本见 [World Labs](../entities/world-labs.md)（Marble + [Spark](../entities/spark-3dgs-renderer.md)）；同类 Web 渲染可对照 [Aholo Viewer](../entities/aholo-viewer.md)（见 [Spark vs Aholo](../comparisons/spark-vs-aholo-web-3dgs-renderers.md)）。

### 术语对照：状态动力学「世界模型」（RWM）

足式控制与 MBRL 文献里也会出现 *Robotic World Model* 指 **学习的前向动力学 + 想象 rollout**（例如 ETH RSL 的 **RWM / RWM-U**：集成 RNN 预测 **状态与特权量**，而非扩散视频）。这与本页以 **像素 / Token 视频** 为中心的生成式世界模型 **共享「预测未来」动机**，但 **观测空间、训练目标与评测口径** 不同；工程入口与双仓分工见 [Robotic World Model（ETH RSL）](../entities/robotic-world-model-eth-rsl.md)。

## 主要技术路线
- **视频即仿真 (Video-as-Simulation)**：利用交互式视频预测器代替解析引擎，详见 [Video-as-Simulation](../concepts/video-as-simulation.md)。
- **扩散模型 (Diffusion-based)**：利用 DDPM 逐步去噪生成未来帧，代表：UniSim。
- **离散 Token 流 (Discrete Token flow)**：将图像量化为 Token，利用 Transformer 预测序列，代表：π₀ 的动作建模部分。
- **生成视频作为人形控制 demo 源**：把第三人称视频生成当成"想象出来的示教"，再用动作估计 + 通用动作跟踪把视频翻译为机器人动作，代表：[ExoActor](./exoactor.md)。

## 关联页面
- [Latent Imagination (潜空间想象)](../concepts/latent-imagination.md)
- [Model-Based RL](../methods/model-based-rl.md)
- [Being-H0.7](./being-h07.md) — 潜空间世界–动作模型，测试时不滚未来像素。
- [World Action Models（WAM）](../concepts/world-action-models.md) — 世界预测与动作生成的联合范式与文献taxonomy
- [NVIDIA Omniverse](../entities/nvidia-omniverse.md)
- [ExoActor](./exoactor.md) — 视频生成驱动的交互式人形控制。
- [EWMBench](../entities/ewmbench.md) — 具身视频世界模型生成质量的多维基准与开源工具链。
- [GE-Sim 2.0](../entities/ge-sim-2.md) — Agibot **闭环** 操纵视频世界模拟器：本体状态专家 + World Judge + 加速 rollout（arXiv:2605.27491）。
- [Cosmos 3](../entities/cosmos-3.md) — NVIDIA **全模态 MoT 世界模型平台**：Reasoner + Generator 双路径，覆盖 VLM、视频生成、policy 与正/逆动力学（arXiv:2606.02800）。
- [τ₀-World Model（τ0-WM）](../entities/tau0-world-model.md) — Agibot **5B 统一视频–动作世界模型**：异构掩码预训练 + 测试时 propose–evaluate–revise（技术报告 2026-05-31）。
- [WEM（World-Ego Model）](../entities/paper-wem-world-ego-modeling.md) — **world/ego 显式解耦** 的长程混合导航–操作视频 rollout 与 **HTEWorld** 基准（arXiv:2605.19957）。
- [Gamma-World](../entities/paper-gamma-world-multi-agent.md) — **多智能体** 置换对称编码 + hub 注意力 + 24 FPS 交互 rollout（arXiv:2605.28816）。
- [HomeWorld](../entities/paper-homeworld-whole-home-scene-generation.md) — **静态 sim-ready 全屋 3D** 场景生成与中文住宅平面图数据（arXiv:2606.06390）。
- [Robotic World Model（ETH RSL）](../entities/robotic-world-model-eth-rsl.md) — 状态空间神经动力学 + 想象 rollout（与像素生成式 WBM 对照）。
- [World Labs](../entities/world-labs.md) — 空间智能与 3D 世界生成产品侧样本（Marble / Spark）。
- [Spark（Web 3DGS）](../entities/spark-3dgs-renderer.md) — LoD splat 树、.RAD 流式与 splat 分页（Spark 2.0）。
- [Aholo Viewer](../entities/aholo-viewer.md) — Chunked Streaming LoD + 3DGS/Mesh 混渲。
- [DWM（Dexterous World Models）](./dwm.md) — 已知静态 3D 场景上的场景–手条件视频扩散与残差动力学学习。
- [mimic-video（Video-Action Model）](./mimic-video.md) — 互联网视频骨干潜计划 + 流匹配动作解码器的操作策略。

## 参考来源
- [机器人论文阅读笔记：Generative World Modelling for Humanoids](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/Generative_World_Modelling_for_Humanoids__1X_World_Model_Challenge_Technical_Report/Generative_World_Modelling_for_Humanoids__1X_World_Model_Challenge_Technical_Report.html)
- Hu, A., et al. (2023). *GAIA-1: A Generative AI for Embodied AI*.
- Yang, S., et al. (2023). *Learning Interactive Real-World Simulators (UniSim)*.
- Zhou Y., et al. (2026). *ExoActor: Exocentric Video Generation as Generalizable Interactive Humanoid Control* — 见 [sources/papers/exoactor.md](../../sources/papers/exoactor.md)。
- Luo, H., et al. (2026). *Being-H0.7: A Latent World-Action Model from Egocentric Videos* — 见 [sources/papers/being_h07.md](../../sources/papers/being_h07.md)。
- Wang, S., et al. (2026). *World Action Models: The Next Frontier in Embodied AI* — 见 [sources/papers/world_action_models_survey_2605.md](../../sources/papers/world_action_models_survey_2605.md)。
- Hu, Y., et al. (2025). *EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models* — 见 [sources/papers/ewmbench.md](../../sources/papers/ewmbench.md)。
- World Labs 官方站点与 Spark/Marble 关联归档 — 见 [sources/sites/worldlabs-ai.md](../../sources/sites/worldlabs-ai.md)。
- Spark 2.0 技术博客归档 — 见 [sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md](../../sources/blogs/worldlabs_spark_2_0_streaming_3dgs.md)。
- Kim, B., et al. (2026). *Dexterous World Models* — 见 [sources/papers/dwm_arxiv_2512_17907.md](../../sources/papers/dwm_arxiv_2512_17907.md)。
- Pai, J., et al. (2025). *mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs* — 见 [sources/papers/mimic_video_arxiv_2512_15692.md](../../sources/papers/mimic_video_arxiv_2512_15692.md)。
- Lin, Z., et al. (2026). *World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks* — 见 [sources/papers/wem_arxiv_2605_19957.md](../../sources/papers/wem_arxiv_2605_19957.md)。
- Liu, F., et al. (2026). *Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players* — 见 [sources/papers/gamma_world_arxiv_2605_28816.md](../../sources/papers/gamma_world_arxiv_2605_28816.md)。
- Qiu, B., et al. (2026). *GE-Sim 2.0: A Roadmap Towards Comprehensive Closed-loop Video World Simulators for Robotic Manipulation* — 见 [sources/papers/ge_sim_2_arxiv_2605_27491.md](../../sources/papers/ge_sim_2_arxiv_2605_27491.md)。
