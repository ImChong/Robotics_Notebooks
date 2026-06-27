---
type: concept
tags: [simulation, video-generation, deepmind, computer-vision, robotics]
status: complete
updated: 2026-06-20
related:
  - ../methods/generative-world-models.md
  - ../entities/ewmbench.md
  - ../entities/paper-wem-world-ego-modeling.md
  - ../methods/dwm.md
  - ../methods/mimic-video.md
  - ../concepts/sim2real.md
  - ../methods/model-based-rl.md
  - ../entities/paper-homeworld-whole-home-scene-generation.md
  - ../entities/molmo-motion.md
sources:
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/exoactor.md
  - ../../sources/papers/ewmbench.md
  - ../../sources/papers/dwm_arxiv_2512_17907.md
  - ../../sources/papers/mimic_video_arxiv_2512_15692.md
  - ../../sources/papers/wem_arxiv_2605_19957.md
  - ../../sources/blogs/allenai_molmo_motion.md
summary: "视频即仿真（Video-as-Simulation）代表了仿真技术的新范式：通过交互式视频预测器代替传统的刚体动力学引擎，实现了在像素级别进行无限逼真的反事实物理演练。"
---

# Video-as-Simulation (视频即仿真)

**视频即仿真 (Video-as-Simulation)** 是具身智能领域最激进也最前沿的技术范式。它的核心假设是：如果一个生成模型能够完美预测“给定当前动作后，下一帧图像应该长什么样”，那么这个模型本身就可以充当一个端到端的、像素级的物理引擎。

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
- **评测口径**：像素 rollout 的「任务是否真被完成」难以用单一感知分数刻画；可对照公开 **具身视频世界模型** 基准（如 [EWMBench](../entities/ewmbench.md) 的场景守恒 / 末端轨迹 / 语义逻辑三轴）做系统性体检，而不是只看通用文生视频榜单。

## 在人形控制上的延伸：[ExoActor](../methods/exoactor.md)

UniSim 把视频生成模型当作可交互的物理引擎来训练视觉策略，而 [ExoActor (BAAI, 2026)](../methods/exoactor.md) 把同一思想推到了**真实物理人形机器人控制**层：用第三人称视频生成模型生成"想象的示教"，再通过 [GENMO](../methods/genmo.md)/[WiLoR](../methods/wilor.md) 估计 SMPL 全身 + 双手动作，并直接喂给 [SONIC](../methods/sonic-motion-tracking.md) 这种通用 motion tracking 控制器在 Unitree G1 上执行。这给"视频即仿真"提供了一个无需真实数据采集的端到端落地实例。

在**已知静态 3D 场景**、以**第一人称手–物交互**为主线的设定下，[DWM（Dexterous World Models）](../methods/dwm.md) 把视频扩散当作「手条件驱动的场景动力学」模拟器：显式渲染静态场景作基线，再预测操纵残差；论文还演示用 rollout 视频对候选动作做**视觉层面的粗评估**，与闭环力控仿真仍是不同层级。

在**开放词汇操作**设定下，[mimic-video（Video-Action Model）](../methods/mimic-video.md) 把同一类「视频模型懂动力学」的直觉接到 **通用操作策略**：用大规模视频骨干的 **潜空间计划** 条件化 **流匹配动作解码器**，默认推理强调 **部分去噪** 而非完整像素 rollout——与把视频当作可点击仿真器的 UniSim 式用法相比，**交互闭环发生在真机控制回路**而非纯像素沙盒。

在**双臂操纵闭环仿真**设定下，[GE-Sim 2.0](../entities/ge-sim-2.md) 把动作条件多视角视频与 **从 latent 解码的关节/夹爪状态**、**任务指令对齐的 World Judge 奖励** 捆在同一平台：策略在模拟器内 chunk 级 rollout 并获得机器可验证成功信号，论文报告 WorldArena 榜首与真机策略增益；与 [EWMBench](../entities/ewmbench.md) 的 **开环生成质量** 评测互补（同属 Genie Envisioner / Agibot 生态）。

在**长程、导航与操作交错**设定下，[WEM（World-Ego Modeling）](../entities/paper-wem-world-ego-modeling.md) 把单流像素 rollout 进一步结构化：将未来演化拆为 **指令无关的场景 world** 与 **指令条件的机体 ego**，由 **RCA 规划器 + CP-MoE 级联并行扩散** 实例化，并发布基于 BEHAVIOR-1K 的 **HTEWorld** 基准（125K 训练片段、300 条多轮评测轨迹）以补齐 [EWMBench](../entities/ewmbench.md) 偏单任务操纵的评测缺口。这给「视频即仿真」提供了一个面向 **多轮混合具身指令** 的结构化预测与评测样板，而不只是单段任务的像素生成质量。

在**静态 3D 仿真资产**设定下，[HomeWorld（Kairos · Whole-Home Scene Generation）](../entities/paper-homeworld-whole-home-scene-generation.md) 走 **互补路线**：不预测像素未来，而是从文本 prompt 经 **四阶段分层流水线**（K-D tree LLM 平面图 → 图像 roaming 软装 → VLM 递归修正 → surface-centric 可操纵小物）直接产出 **sim-ready 全屋 furnished 3D**（300K 中国住宅矢量平面图 + 5K 全屋场景待开源）。它与 UniSim / GE-Sim 等 **video WM** 解决的是 **「环境从哪来」** 的上游问题——尤其面向 **跨房间导航与家务** 需要 **全局连贯多房间** 而非单 room 拼接的场景库。

在**中间层 motion guidance** 设定下，[MolmoMotion](../entities/molmo-motion.md)（Ai2，arXiv:2606.18558）不直接生成整段像素 rollout，而是先预测 **语言条件下的 metric 3D 点轨迹**，再注入 **DaS + I2V** 等视频生成器以约束 **小幅度精确运动**；与「纯文本 prompt 猜 motion」相比，把 **物理运动结构** 从像素生成中 **显式解耦**，亦与 [mimic-video](../methods/mimic-video.md) 的 **潜视频计划** 形成 **3D 几何 vs 潜空间** 两种中间表示对照。

## 关联页面
- [仿真物理保真度链路选型指南](../queries/simulation-physics-fidelity.md) — 本页所述物理/仿真要素在保真度链路（建模 ① → 数值 ② → 接触 ③ → 随机化 ④）中的定位
- [Generative World Models](../methods/generative-world-models.md)
- [EWMBench](../entities/ewmbench.md) — 操纵场景下视频世界模型生成的多维评测坐标
- [Sim2Real (仿真到现实迁移)](../concepts/sim2real.md)
- [Model-Based RL](../methods/model-based-rl.md)
- [ExoActor](../methods/exoactor.md) — 把视频即仿真思想用到人形机器人交互行为生成上。
- [DWM（Dexterous World Models）](../methods/dwm.md) — 静态场景已知时的手条件交互视频 rollout 与评估型用法。
- [mimic-video（Video-Action Model）](../methods/mimic-video.md) — 潜空间视频计划条件化流匹配动作解码器的操作策略路线。
- [WEM（World-Ego Modeling）](../entities/paper-wem-world-ego-modeling.md) — world/ego 解耦的长程混合导航–操作视频世界模型与 HTEWorld 评测基准。
- [Gamma-World](../entities/paper-gamma-world-multi-agent.md) — 多智能体共享世界的实时动作条件视频 rollout（arXiv:2605.28816）。
- [GE-Sim 2.0](../entities/ge-sim-2.md) — 闭环操纵视频模拟器：视觉 + 本体双专家与世界裁判（arXiv:2605.27491）。
- [HomeWorld](../entities/paper-homeworld-whole-home-scene-generation.md) — 文本到 sim-ready 全屋 3D 场景（arXiv:2606.06390）；与 video WM 互补的静态仿真资产路线。
- [MolmoMotion](../entities/molmo-motion.md) — 3D 点轨迹预测作 I2V motion guidance 与机器人规划先验（arXiv:2606.18558）。

## 参考来源
- Yang, S., et al. (2023). *UniSim: Learning Interactive Real-World Simulators*.
- [Google DeepMind Blog on UniSim](https://deepmind.google/discover/blog/unisim/).
- Zhou Y., et al. (2026). *ExoActor: Exocentric Video Generation as Generalizable Interactive Humanoid Control* — 见 [sources/papers/exoactor.md](../../sources/papers/exoactor.md)。
- Hu, Y., et al. (2025). *EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models* — 见 [sources/papers/ewmbench.md](../../sources/papers/ewmbench.md)。
- Kim, B., et al. (2026). *Dexterous World Models* — 见 [sources/papers/dwm_arxiv_2512_17907.md](../../sources/papers/dwm_arxiv_2512_17907.md)。
- Pai, J., et al. (2025). *mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs* — 见 [sources/papers/mimic_video_arxiv_2512_15692.md](../../sources/papers/mimic_video_arxiv_2512_15692.md)。
- Lin, Z., et al. (2026). *World-Ego Modeling for Long-Horizon Evolution in Hybrid Embodied Tasks* (arXiv:2605.19957) — 见 [sources/papers/wem_arxiv_2605_19957.md](../../sources/papers/wem_arxiv_2605_19957.md)。
- Qiu, B., et al. (2026). *GE-Sim 2.0* (arXiv:2605.27491) — 见 [sources/papers/ge_sim_2_arxiv_2605_27491.md](../../sources/papers/ge_sim_2_arxiv_2605_27491.md)。
- Li, W., et al. (2026). *HomeWorld* (arXiv:2606.06390) — 见 [sources/papers/homeworld_arxiv_2606_06390.md](../../sources/papers/homeworld_arxiv_2606_06390.md)。
- Zhang, J., et al. (2026). *MolmoMotion* (arXiv:2606.18558) — 见 [sources/blogs/allenai_molmo_motion.md](../../sources/blogs/allenai_molmo_motion.md)。
