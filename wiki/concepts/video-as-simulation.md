---
type: concept
tags: [simulation, video-generation, deepmind, computer-vision, robotics]
status: complete
updated: 2026-09-05
related:
  - ../entities/paper-vgi-white-paper.md
  - ../methods/generative-world-models.md
  - ./functional-taxonomy-world-models.md
  - ../entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md
  - ../entities/ewmbench.md
  - ../entities/paper-worldscore.md
  - ../entities/paper-harnesseval-w.md
  - ../entities/paper-wem-world-ego-modeling.md
  - ../entities/rekacs2-10k-dataset.md
  - ../methods/dwm.md
  - ../methods/mimic-video.md
  - ../concepts/sim2real.md
  - ../methods/model-based-rl.md
  - ../entities/paper-homeworld-whole-home-scene-generation.md
  - ../entities/molmo-motion.md
  - ../entities/paper-x-world.md
  - ../entities/paper-x-cache.md
  - ../entities/paper-driftworld.md
  - ../entities/paper-masked-visual-actions.md
  - ../entities/paper-ctrl-world.md
  - ../entities/paper-wall-ss.md
  - ../entities/paper-odeworld.md
  - ../entities/paper-levjepa.md
  - ../entities/paper-wan-move.md
  - ../entities/paper-wan-dancer.md
  - ../entities/paper-wan-video.md
  - ../entities/paper-robointer-1-5.md
  - ../entities/paper-m4world.md
  - ../entities/paper-abot-world-0.md
  - ../entities/paper-roboreact.md
  - ../entities/cosmos-transfer.md
sources:
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/blogs/worldlabs_functional_taxonomy_world_models.md
  - ../../sources/papers/world_model_definition_roadmap_arxiv_2607_06401.md
  - ../../sources/papers/exoactor.md
  - ../../sources/papers/roboreact_arxiv_2608_03387.md
  - ../../sources/papers/ewmbench.md
  - ../../sources/papers/worldscore_arxiv_2504_00983.md
  - ../../sources/papers/harnesseval_w_arxiv_2608_16859.md
  - ../../sources/papers/dwm_arxiv_2512_17907.md
  - ../../sources/papers/mimic_video_arxiv_2512_15692.md
  - ../../sources/papers/wem_arxiv_2605_19957.md
  - ../../sources/papers/x_world_arxiv_2603_19979.md
  - ../../sources/papers/driftworld_arxiv_2607_15065.md
  - ../../sources/papers/m4world_arxiv_2607_14005.md
  - ../../sources/papers/abot_world_0_arxiv_2607_19191.md
  - ../../sources/papers/masked_visual_actions_arxiv_2607_19343.md
  - ../../sources/papers/ctrl_world_arxiv_2510_10125.md
  - ../../sources/papers/wall_ss_x_square_2026.md
  - ../../sources/papers/odeworld_arxiv_2607_27924.md
  - ../../sources/papers/wan_move_arxiv_2512_08765.md
  - ../../sources/papers/wan_video_arxiv_2503_20314.md
  - ../../sources/blogs/allenai_molmo_motion.md
  - ../../sources/sites/rekacs2-10k.md
summary: "视频即仿真（Video-as-Simulation）代表了仿真技术的新范式：通过交互式视频预测器代替传统的刚体动力学引擎，实现了在像素级别进行无限逼真的反事实物理演练。"
---

# Video-as-Simulation (视频即仿真)

**视频即仿真 (Video-as-Simulation)** 是具身智能领域最激进也最前沿的技术范式。它的核心假设是：如果一个生成模型能够完美预测“给定当前动作后，下一帧图像应该长什么样”，那么这个模型本身就可以充当一个端到端的、像素级的物理引擎。在 [功能分类](./functional-taxonomy-world-models.md) 里，这类系统默认是 **Renderer**；加上可靠动作条件与可查询状态后才靠近 Simulator。[定义文](../entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md) 提醒：像素逼真既非必要也非充分。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Vid2Sim | Video-as-Simulation | 用视频/生成模型替代或补充解析仿真 |
| WM | World Model | 从视频学环境动态的相邻概念 |
| Sim2Real | Simulation to Real | 生成资产/动力学仍须工程验收 |
| RL | Reinforcement Learning | 可在生成场景中训练策略 |
| NeRF / 3DGS | Neural Radiance Fields / 3D Gaussian Splatting | 常见场景重建与资产表示 |

## 核心差异：从“算力矩”到“算像素”

| 维度 | 传统物理仿真 (如 MuJoCo) | 视频即仿真 (如 UniSim) |
|------|------------------------|-----------------------|
| **建模对象** | 几何体、力学常数、接触点 | 原始像素、时空关联、视觉语义 |
| **真实度** | 受限于手工调参的数学模型 | 无限趋近于真实视频录像 |
| **交互方式** | 求解动力学 ODE | 预测视频下一帧 (Next-frame Prediction) |
| **泛化性** | 仅限于建立好模型的物体 | 见过视频的所有场景均可模拟 |

## 关键技术：交互式视频预测器

实现“视频即仿真”的核心是一个**条件生成模型**。给定当前的 RGB 观测 $O_t$ 和机器人预期的控制动作 $A_t$，模型输出预测的下一帧 $\hat{O}_{t+1}$：

$$ \hat{O}_{t+1} = \mathcal{G}(O_t, A_t, \text{Instruction}) $$

### 代表性工作：UniSim (Google DeepMind)
UniSim 证明了通过将“大量无人干预的人类视频”与“少量机器人交互视频”混合训练，可以构建出一个可交互的模拟器。用户可以像玩电子游戏一样输入指令（如“打开微波炉”），并在生成的视频中看到机器人的执行反馈。

## 为什么这一范式能够成功？

1. **Sim2Real 的终极解**：在“视频”中训练的策略，其观测空间与真实摄像头完全一致，不存在视觉层面的鸿沟。
2. **解决“建模难题”**：对于折衣服、液体倾倒等柔体/流体任务，传统仿真极难精准建模，而视频模型可以从海量互联网视频中自发习得这些复杂的物理常识。
3. **数据 Scaling**：它允许机器人利用 YouTube 等非结构化视频进行“离线学习”，极大地提升了数据的利用率。

## 当前局限

- **因果偏差**：模型可能会产生“动作未到，物体先动”的逻辑错误。
- **长程发散**：随着预测步数增加，生成的视频细节会逐渐模糊，导致策略在长时域任务中失效。
- **无法闭环力控**：由于缺乏接触力反馈，它更适合训练视觉策略，而非底层高频关节控制。
- **评测口径**：像素 rollout 的「任务是否真被完成」难以用单一感知分数刻画；可对照公开 **具身视频世界模型** 基准（如 [EWMBench](../entities/ewmbench.md) 的场景守恒 / 末端轨迹 / 语义逻辑三轴）做系统性体检，而不是只看通用文生视频榜单。若关心的是开放域 **多场景运镜与跨场景一致性**（含 3D/4D 方法），改用 [WorldScore](../entities/paper-worldscore.md)。若关心 **指定干预是否发生、离开再回/离屏过程是否还在**，改用 [HarnessEval-W](../entities/paper-harnesseval-w.md)。

## 在人形控制上的延伸：[ExoActor](../methods/exoactor.md)

UniSim 把视频生成模型当作可交互的物理引擎来训练视觉策略，而 [ExoActor (BAAI, 2026)](../methods/exoactor.md) 把同一思想推到了**真实物理人形机器人控制**层：用第三人称视频生成模型生成"想象的示教"，再通过 [GENMO](../methods/genmo.md)/[WiLoR](../methods/wilor.md) 估计 SMPL 全身 + 双手动作，并直接喂给 [SONIC](../methods/sonic-motion-tracking.md) 这种通用 motion tracking 控制器在 Unitree G1 上执行。这给"视频即仿真"提供了一个无需真实数据采集的端到端落地实例。[RoboReact](../entities/paper-roboreact.md)（arXiv:2608.03387）走另一条落地：生成视频只当任务顺序与手–物结构先验，编译成物体中心关键帧后在真机标定，测试时不再跑视频模型。

在**已知静态 3D 场景**、以**第一人称手–物交互**为主线的设定下，[DWM（Dexterous World Models）](../methods/dwm.md) 把视频扩散当作「手条件驱动的场景动力学」模拟器：显式渲染静态场景作基线，再预测操纵残差；论文还演示用 rollout 视频对候选动作做**视觉层面的粗评估**，与闭环力控仿真仍是不同层级。

在**开放词汇操作**设定下，[mimic-video（Video-Action Model）](../methods/mimic-video.md) 把同一类「视频模型懂动力学」的直觉接到 **通用操作策略**：用大规模视频骨干的 **潜空间计划** 条件化 **流匹配动作解码器**，默认推理强调 **部分去噪** 而非完整像素 rollout——与把视频当作可点击仿真器的 UniSim 式用法相比，**交互闭环发生在真机控制回路**而非纯像素沙盒。

在**双臂操纵闭环仿真**设定下，[GE-Sim 2.0](../entities/ge-sim-2.md) 把动作条件多视角视频与 **从 latent 解码的关节/夹爪状态**、**任务指令对齐的 World Judge 奖励** 捆在同一平台：策略在模拟器内 chunk 级 rollout 并获得机器可验证成功信号，论文报告 WorldArena 榜首与真机策略增益；与 [EWMBench](../entities/ewmbench.md) 的 **开环生成质量** 评测互补（同属 Genie Envisioner / Agibot 生态）。

在**长程、导航与操作交错**设定下，[WEM（World-Ego Modeling）](../entities/paper-wem-world-ego-modeling.md) 把单流像素 rollout 进一步结构化：将未来演化拆为 **指令无关的场景 world** 与 **指令条件的机体 ego**，由 **RCA 规划器 + CP-MoE 级联并行扩散** 实例化，并发布基于 BEHAVIOR-1K 的 **HTEWorld** 基准（125K 训练片段、300 条多轮评测轨迹）以补齐 [EWMBench](../entities/ewmbench.md) 偏单任务操纵的评测缺口。这给「视频即仿真」提供了一个面向 **多轮混合具身指令** 的结构化预测与评测样板，而不只是单段任务的像素生成质量。

在**静态 3D 仿真资产**设定下，[HomeWorld（Kairos · Whole-Home Scene Generation）](../entities/paper-homeworld-whole-home-scene-generation.md) 走 **互补路线**：不预测像素未来，而是从文本 prompt 经 **四阶段分层流水线**（K-D tree LLM 平面图 → 图像 roaming 软装 → VLM 递归修正 → surface-centric 可操纵小物）直接产出 **sim-ready 全屋 furnished 3D**（300K 中国住宅矢量平面图 + 5K 全屋场景待开源）。它与 UniSim / GE-Sim 等 **video WM** 解决的是 **「环境从哪来」** 的上游问题——尤其面向 **跨房间导航与家务** 需要 **全局连贯多房间** 而非单 room 拼接的场景库。

在**中间层 motion guidance** 设定下，[MolmoMotion](../entities/molmo-motion.md)（Ai2，arXiv:2606.18558）不直接生成整段像素 rollout，而是先预测 **语言条件下的 metric 3D 点轨迹**，再注入 **DaS + I2V** 等视频生成器以约束 **小幅度精确运动**；与「纯文本 prompt 猜 motion」相比，把 **物理运动结构** 从像素生成中 **显式解耦**，亦与 [mimic-video](../methods/mimic-video.md) 的 **潜视频计划** 形成 **3D 几何 vs 潜空间** 两种中间表示对照。

当瓶颈是 **推理时延**（多提案想象搜索）而非跨具身条件时，[DriftWorld](../entities/paper-driftworld.md)（arXiv:2607.15065）用 **drifting 1-step** 替代多步扩散采样：单次前向即可从当前帧 + 动作生成未来帧（H100 上 30+ fps），并把同一快仿真器接到 **GPC-RANK** 与离线策略评估——说明「视频即仿真」的工程上限常受 **采样步数** 约束，而不只是视觉逼真度。

当瓶颈是 **动作如何注入视频先验**（且希望同一模型兼顾前向仿真与逆向行为合成）时，[Masked Visual Actions](../entities/paper-masked-visual-actions.md)（arXiv:2607.19343）用 **像素掩码轨迹** 作控制接口：揭示机器人即前向动力学，揭示物体目标运动即逆向合成；约 15 h 数据微调后即可做 Best-of-N 规划与策略评估（RoboCasa **r=0.982**）。

当瓶颈是 **与现代多视角 VLA 闭环兼容**（腕部相机 + 高维动作块）时，[Ctrl-World](../entities/paper-ctrl-world.md)（arXiv:2510.10125，ICLR 2026）用帧级动作条件与位姿记忆，在想象中做策略排序与合成轨迹 SFT（新指令 **38.7%→83.4%**）。

当瓶颈是 **clip 级扩散的动作捷径 + 长程误差累积** 时，[WALL-SS](../entities/paper-wall-ss.md)（自变量，2026-08）用 **next-scale 自回归** 把动作写进 coarse→fine 链条，有界记忆滚约 **60 s**，并以 **600** 对虚实成功率校准（\(r=0.93\)）；训练代码待发布。

当瓶颈是 **离散帧率 / 只能正向 next-step** 时，[ODEWorld](../entities/paper-odeworld.md)（arXiv:2607.27924）把预测改成物理时间 latent ODE：可任意 \(\tau\)、反向积分、缺帧插值，再可选 RAE 解码像素。规划发生在单 token 动力学 latent，**不是**动作条件像素沙盒（作者写明当前无动作条件）。

通用开源视频先验侧，[Wan](../entities/paper-wan-video.md) / [Wan-Move](../entities/paper-wan-move.md) / [Wan-Dancer](../entities/paper-wan-dancer.md) 提供可微调的 I2V、点轨迹运动控制与分钟级 music-to-dance；它们不是操纵仿真器，但是像素条件 WM / 参考视频先验的常见上游（MVA 对照基线）。

当瓶颈是 **多类型中间表示如何同时服务 VLA 与长程像素推演** 时，[RoboInter1.5](../entities/paper-robointer-1-5.md)（arXiv:2607.18709）用稠密 IR 渲染控制视频条件化 **RoboInter-World**，并把同一 IR 接到 plan-then-execute VLA——与「纯语言/原始动作条件」形成对照。

在 **车载多传感器仿真** 设定下，[M⁴World](../entities/paper-m4world.md)（arXiv:2607.14005）把「视频即仿真」扩展到 **环视相机 + 同步 LiDAR range map**，并以物体级外观条件支持长尾场景编辑；4-step 因果蒸馏面向 **分钟级** 流式，与操纵域 DriftWorld 的「少步」动机相近但评测与传感栈不同。

在 **开放域交互式内容 / 桌面实时** 设定下，[ABot-World-0](../entities/paper-abot-world-0.md)（arXiv:2607.19191）用 **原始键盘动作** 统一场景漫游与第三人称角色，经 **LongForcing** 压长程自回归漂移，并在 **单卡 RTX 5090** 上做到 **720P ≤16 FPS**；它更接近 UniSim「可玩的像素世界」叙事，而不是关节级机器人策略评估代理——推理学生已开源，完整训练数据/教师仍待发。

## 关联页面
- [Visual General Intelligence 白皮书](../entities/paper-vgi-white-paper.md) — Geirhos「视频模型即 VFM」与 Wu & Wu「像素生成 ≠ 物理理解」的对张力，用来读本页「视频即仿真」的适用边界
- [仿真物理保真度链路选型指南](../queries/simulation-physics-fidelity.md) — 本页所述物理/仿真要素在保真度链路（建模 ① → 数值 ② → 接触 ③ → 随机化 ④）中的定位
- [RekaCS2-10k](../entities/rekacs2-10k-dataset.md) — 职业 CS2 ego 视频 + 逐帧控制，动作条件交互世界模型语料
- [世界模型功能分类](./functional-taxonomy-world-models.md) — 视频即仿真默认是 Renderer；动作条件后才靠近 Simulator
- [世界模型定义与路线图](../entities/paper-sa-2607-06401-a-definition-and-roadmap-for-world-models.md) — 像素逼真既非必要也非充分
- [Generative World Models](../methods/generative-world-models.md)
- [EWMBench](../entities/ewmbench.md) — 操纵场景下视频世界模型生成的多维评测坐标
- [WorldScore](../entities/paper-worldscore.md) — 开放域多场景 + 相机可控世界生成统一评测（ICCV 2025）
- [HarnessEval-W](../entities/paper-harnesseval-w.md) — 交互式世界 agentic 评测：干预是否发生、drift/revisit/offscreen
- [Cosmos Transfer](../entities/cosmos-transfer.md) — 控制条件 world-to-world，补仿真视觉域而不是当像素物理引擎
- [Sim2Real (仿真到现实迁移)](../concepts/sim2real.md)
- [Model-Based RL](../methods/model-based-rl.md)
- [ExoActor](../methods/exoactor.md) — 把视频即仿真思想用到人形机器人交互行为生成上。
- [RoboReact](../entities/paper-roboreact.md) — 生成 egocentric 视频当任务先验，编译成物体中心技能再上真机（不是像素物理引擎）。
- [DWM（Dexterous World Models）](../methods/dwm.md) — 静态场景已知时的手条件交互视频 rollout 与评估型用法。
- [mimic-video（Video-Action Model）](../methods/mimic-video.md) — 潜空间视频计划条件化流匹配动作解码器的操作策略路线。
- [WEM（World-Ego Modeling）](../entities/paper-wem-world-ego-modeling.md) — world/ego 解耦的长程混合导航–操作视频世界模型与 HTEWorld 评测基准。
- [Gamma-World](../entities/paper-gamma-world-multi-agent.md) — 多智能体共享世界的实时动作条件视频 rollout（arXiv:2605.28816）。
- [GE-Sim 2.0](../entities/ge-sim-2.md) — 闭环操纵视频模拟器：视觉 + 本体双专家与世界裁判（arXiv:2605.27491）。
- [HomeWorld](../entities/paper-homeworld-whole-home-scene-generation.md) — 文本到 sim-ready 全屋 3D 场景（arXiv:2606.06390）；与 video WM 互补的静态仿真资产路线。
- [MolmoMotion](../entities/molmo-motion.md) — 3D 点轨迹预测作 I2V motion guidance 与机器人规划先验（arXiv:2606.18558）。
- [DriftWorld](../entities/paper-driftworld.md) — 1-step drifting 动作条件视频 WM：推理时搜索 + 离线评估（arXiv:2607.15065）。
- [Masked Visual Actions](../entities/paper-masked-visual-actions.md) — 掩码视觉动作统一前向/逆向 + 策略评估（arXiv:2607.19343）。
- [Ctrl-World](../entities/paper-ctrl-world.md) — 多视角可控 WM：VLA 闭环评估 + 合成 SFT（ICLR 2026）。
- [WALL-SS](../entities/paper-wall-ss.md) — next-scale AR 长程 WM：60 s 流式 + 虚实校准（训练代码待发布）。
- [ODEWorld](../entities/paper-odeworld.md) — 物理时间 latent ODE：任意时刻/反向视频，规划不在像素环（arXiv:2607.27924）。
- [LeVJEPA](../entities/paper-levjepa.md) — 视频当**表征底物**而非仿真器：无像素重建，只要因果 JEPA 编码器。
- [Wan](../entities/paper-wan-video.md) / [Wan-Move](../entities/paper-wan-move.md) / [Wan-Dancer](../entities/paper-wan-dancer.md) — 开源视频基础模型、轨迹运动控制与分钟级 music-to-dance。
- [RoboInter1.5](../entities/paper-robointer-1-5.md) — IR 控制视频条件世界模型 + 操作 VLA 套件（arXiv:2607.18709）。
- [X-World](../entities/paper-x-world.md) — 小鹏 **7 摄动作条件** 驾驶视频世界模型（arXiv:2603.19979；未开源）。
- [X-Cache](../entities/paper-x-cache.md) — 少步 AR 世界模型跨 chunk 加速（arXiv:2604.20289）。
- [M⁴World](../entities/paper-m4world.md) — 多视角多模态驾驶 WM：物体外观控制 + 分钟级流式（arXiv:2607.14005；未开源）。
- [ABot-World-0](../entities/paper-abot-world-0.md) — 单卡桌面键盘交互视频世界模型（arXiv:2607.19191；部分开源）。

## 参考来源
- Fei-Fei Li / World Labs (2026). *A Functional Taxonomy of World Models* — 见 [worldlabs_functional_taxonomy_world_models.md](../../sources/blogs/worldlabs_functional_taxonomy_world_models.md)。
- Physical Intelligence Team (2026). *A Definition and Roadmap for World Models* — 见 [world_model_definition_roadmap_arxiv_2607_06401.md](../../sources/papers/world_model_definition_roadmap_arxiv_2607_06401.md)。
- Yang, S., et al. (2023). *UniSim: Learning Interactive Real-World Simulators*.
- [Google DeepMind Blog on UniSim](https://deepmind.google/discover/blog/unisim/).
- Zhou Y., et al. (2026). *ExoActor: Exocentric Video Generation as Generalizable Interactive Humanoid Control* — 见 [sources/papers/exoactor.md](../../sources/papers/exoactor.md)。
- He S., et al. (2026). *RoboReact* (arXiv:2608.03387) — 见 [sources/papers/roboreact_arxiv_2608_03387.md](../../sources/papers/roboreact_arxiv_2608_03387.md)。
- Hu, Y., et al. (2025). *EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models* — 见 [sources/papers/ewmbench.md](../../sources/papers/ewmbench.md)。
- Duan, H., et al. (2025). *WorldScore: A Unified Evaluation Benchmark for World Generation* — 见 [sources/papers/worldscore_arxiv_2504_00983.md](../../sources/papers/worldscore_arxiv_2504_00983.md)。
- Kim, B., et al. (2026). *Dexterous World Models* — 见 [sources/papers/dwm_arxiv_2512_17907.md](../../sources/papers/dwm_arxiv_2512_17907.md)。
- Pai, J., et al. (2025). *mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs* — 见 [sources/papers/mimic_video_arxiv_2512_15692.md](../../sources/papers/mimic_video_arxiv_2512_15692.md)。
- Lin, Z., et al. (2026). *World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks* (arXiv:2605.19957) — 见 [sources/papers/wem_arxiv_2605_19957.md](../../sources/papers/wem_arxiv_2605_19957.md)。
- Qiu, B., et al. (2026). *GE-Sim 2.0* (arXiv:2605.27491) — 见 [sources/papers/ge_sim_2_arxiv_2605_27491.md](../../sources/papers/ge_sim_2_arxiv_2605_27491.md)。
- Li, W., et al. (2026). *HomeWorld* (arXiv:2606.06390) — 见 [sources/papers/homeworld_arxiv_2606_06390.md](../../sources/papers/homeworld_arxiv_2606_06390.md)。
- Zhang, J., et al. (2026). *MolmoMotion* (arXiv:2606.18558) — 见 [sources/blogs/allenai_molmo_motion.md](../../sources/blogs/allenai_molmo_motion.md)。
- Lu, S., et al. (2026). *DriftWorld* (arXiv:2607.15065) — 见 [sources/papers/driftworld_arxiv_2607_15065.md](../../sources/papers/driftworld_arxiv_2607_15065.md)。
- Zheng, C., et al. (2026). *X-World* (arXiv:2603.19979) — 见 [sources/papers/x_world_arxiv_2603_19979.md](../../sources/papers/x_world_arxiv_2603_19979.md)。
- Zayer, H., et al. (2026). *Masked Visual Actions* (arXiv:2607.19343) — 见 [sources/papers/masked_visual_actions_arxiv_2607_19343.md](../../sources/papers/masked_visual_actions_arxiv_2607_19343.md)。
- Guo, Y., et al. (2026). *Ctrl-World* (arXiv:2510.10125) — 见 [sources/papers/ctrl_world_arxiv_2510_10125.md](../../sources/papers/ctrl_world_arxiv_2510_10125.md)。
- Chu, R., et al. (2025). *Wan-Move* (arXiv:2512.08765) — 见 [sources/papers/wan_move_arxiv_2512_08765.md](../../sources/papers/wan_move_arxiv_2512_08765.md)。
- Wan Team (2025). *Wan* (arXiv:2503.20314) — 见 [sources/papers/wan_video_arxiv_2503_20314.md](../../sources/papers/wan_video_arxiv_2503_20314.md)。
- Team of RoboInter1.5 (2026). *RoboInter1.5* (arXiv:2607.18709) — 见 [sources/papers/robointer_1_5_arxiv_2607_18709.md](../../sources/papers/robointer_1_5_arxiv_2607_18709.md)。
- Cheng, K., et al. (2026). *M⁴World* (arXiv:2607.14005) — 见 [sources/papers/m4world_arxiv_2607_14005.md](../../sources/papers/m4world_arxiv_2607_14005.md)。
- Jiang, F., et al. (2026). *ABot-World-0* (arXiv:2607.19191) — 见 [sources/papers/abot_world_0_arxiv_2607_19191.md](../../sources/papers/abot_world_0_arxiv_2607_19191.md)。
