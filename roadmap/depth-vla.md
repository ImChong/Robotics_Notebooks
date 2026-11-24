# 视觉-语言-动作模型（VLA）

**首屏导读**：

- **为谁**：想让机器人听懂指令干活的人。
- **路径**：Stage 0–5 · 具身模型全景 → 模仿学习基座 → VLA 主线（RT → OpenVLA → π0）→ 数据与 Scaling → 部署整合 → 进阶方向。
- **纵深路线**：[运动控制主路线](motion-control.md)的一条分支；与 [BFM 纵深](depth-bfm.md)构成"任务级语义 / 身体级协调"的姊妹路线。

## 路线一览

```mermaid
flowchart LR
  S0["<b>Stage 0</b><br/>全景与前置<br/><em>VLM/VLN/VLA/BFM 分类学</em>"]
  S1["<b>Stage 1</b><br/>策略基座<br/><em>BC / ACT / Diffusion Policy</em>"]
  S2["<b>Stage 2</b><br/>VLA 主线<br/><em>RT 系列 → OpenVLA → π0</em>"]
  S3["<b>Stage 3</b><br/>数据与 Scaling<br/><em>跨本体 / 人类视频 / WAM</em>"]
  S4["<b>Stage 4</b><br/>部署与整合<br/><em>推理延迟 / 编排 / 分层接口</em>"]
  S5["<b>Stage 5</b><br/>进阶方向<br/><em>RL 微调 / 世界模型 / 评测</em>"]

  S0 --> S1 --> S2 --> S3 --> S4 --> S5

  classDef stage fill:#142a3a,stroke:#f1c40f,stroke-width:2px,color:#fff
  class S0,S1,S2,S3,S4,S5 stage
```

## 这条路径怎么用

- 目标读者是有深度学习基础、想让机器人"看图、听指令、出动作"的人——主战场是操作（manipulation）任务
- VLA 解决 **任务级语义**：把 VLM 的跨模态理解接到机器人动作上；它不负责人形全身协调，那是 [BFM 纵深](depth-bfm.md) 的主题
- 每个阶段都有前置知识、核心问题、推荐做什么、推荐读什么、学完输出什么

**和主路线的关系：**
- 本路线是主路线 L5（RL 与模仿学习）之后偏"学习侧"的进阶方向，Stage 0–1 与 [模仿学习纵深](depth-imitation-learning.md) 的策略基座高度重叠
- 如果目标是"让机械臂听话干活"，走完 Stage 3 即可上手工程；Stage 4–5 面向部署与研究前沿
- 如果目标是人形全身控制或"高层 VLA + 低层 BFM"整机栈，读完 Stage 2 后切到 [BFM 纵深](depth-bfm.md)

---

## Stage 0 具身基础模型全景与前置

**先把缩写地图铺开：VLM / VLN / VLA / WAM / BFM 各管一段，混着读论文只会越读越乱。**

### 前置知识
- Python + PyTorch 熟练
- 理解 Transformer / attention（参考 [Transformer](../wiki/concepts/transformer.md)）
- 对 LLM / VLM 有使用级直觉（知道 CLIP、LLaVA 大概是什么）

### 核心问题
- VLM / VLN / VLA / 世界模型这些缩写各自指什么、边界在哪
- 什么是 foundation policy，它和"单任务策略"的本质区别是什么
- VLA 与 BFM 在机器人栈里各自解决哪一层的问题、为什么常常要叠加而不是二选一

### 推荐做什么
- 按分类学页给五类模型各找一个代表工作，写一页纸对照表
- 用 [LeRobot](../wiki/entities/lerobot.md) 跑通一个现成策略的推理 demo（只推理、不训练）

### 推荐读什么

- [Robot Weight Debugging｜让语言模型直接稀疏编辑机器人策略权重](../wiki/entities/paper-robot-weight-debugging-2611-09376.md) — 该研究以语言模型驱动的稀疏词嵌入行编辑，在不进行梯度再训练、也不补充机器人示范的前提下改善已掌握技能与不同指令措辞的匹配。（原文 arXiv:2611.09376；详见开放和实测边界）。
- [VLM / VLN / VLA / VLX / 世界模型分类学](../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)
- [VLA 全生命周期综述（2611.18372）](../wiki/entities/paper-vla-lifecycle-survey-2611-18372.md) — Stage 0 先用模型四轴、训练两轴、部署一轴统一选型语言，并核对成功率、鲁棒性与硬件时延口径。
- [Foundation Policy](../wiki/concepts/foundation-policy.md) — VLA 的母概念页
- [具身基础模型纵深](../wiki/overview/hub-embodied-foundation-model.md)
- [Query：具身大模型家族分类学闭环](../wiki/queries/embodied-fm-taxonomy-loop.md)
- [LLMs-from-scratch（Raschka）](../wiki/entities/llms-from-scratch-raschka.md) — **可选前置**：Transformer/GPT 从零实现；配套 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11)
- [Karpathy Zero to Hero（YouTube）](../wiki/entities/andrej-karpathy.md) — **可选技术轨前置**：[10 集播放列表](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)（micrograd → GPT → GPT-2）；配套 [`nn-zero-to-hero`](https://github.com/karpathy/nn-zero-to-hero)
- [Andrej Karpathy LLM 科普（YouTube）](../wiki/entities/andrej-karpathy.md) — **可选前置（偏直觉）**：[Intro to LLMs（~1 h）](https://www.youtube.com/watch?v=zjkBMFhNj_g) → [Deep Dive into LLMs（~3.5 h）](https://www.youtube.com/watch?v=7xTGNNLPyMI)；建立 pretrain/SFT/RLHF、工具调用与上下文窗口心智模型，再读 VLA 论文更省力

### 学完输出什么
- 能一句话说清 VLA 是什么、不是什么
- 拿到一篇新论文能放进 VLM / VLN / VLA / WAM / BFM 的正确格子里

---

## Stage 1 模仿学习策略基座

**VLA 的"动作头"建在这一层上。走过 [模仿学习纵深](depth-imitation-learning.md) Stage 0–3 的可以跳。**

### 前置知识
- Stage 0 内容
- 理解监督学习与 [Behavior Cloning](../wiki/methods/behavior-cloning.md) 基本概念

### 核心问题
- BC 的 compounding error 从哪来，为什么 action chunking 能显著缓解
- ACT（BC with Transformer）与 Diffusion Policy 的建模差异（显式回归 vs 生成式去噪）
- 为什么高维、多峰的动作分布需要生成式建模

### 推荐做什么
- 用 LeRobot / ACT 官方实现在仿真里训一个 pick-and-place 策略
- 同一任务上对比 ACT 与 Diffusion Policy 的成功率与推理延迟

### 推荐读什么
- [Action Chunking](../wiki/methods/action-chunking.md) 与 [BC with Transformer](../wiki/methods/bc-with-transformer.md)
- [Diffusion Policy](../wiki/methods/diffusion-policy.md) 与 [Diffusion Model](../wiki/concepts/diffusion-model.md)
- [Imitation Learning](../wiki/methods/imitation-learning.md)
- 若目标是把 **DP → Robotic DiT → Flow → GR00T** 与 PredActor 八篇代表作读串：见专题 [扩散与流匹配策略纵深](depth-robotics-diffusion-dit-flow.md)

### 学完输出什么
- 一个能在仿真里跑通的视觉-动作模仿策略
- 能解释 action chunking 与生成式动作头为什么成了 VLA 的标配组件

---

## Stage 2 VLA 主线：从 RT 系列到 π0

**VLA 概念由 RT-2（2023）确立：把 VLM 的语义能力直接接到机器人动作上。这是本路线的主干。**

### 前置知识
- Stage 1 内容
- 了解 VLM 的基本结构（视觉编码器 + LLM backbone）

### 核心问题
- RT-1 → RT-2 的关键跃迁：动作离散化为 token、与互联网 VQA 数据联合微调（co-fine-tuning）
- OpenVLA / Octo 的开源路线与跨本体（cross-embodiment）数据集 OXE 的作用
- π0 为什么用 flow matching 动作专家而不是自回归动作 token，π0.7 又改了什么
- SayCan 一系"LLM 高层规划"与端到端 VLA 的关系，指令增强（DIAL）解决什么问题

### 推荐做什么
- 用 OpenVLA 或 π0 开源权重在 LIBERO / 自建仿真任务上跑一轮评测
- 用 LoRA 把一个小 VLA 微调到自己的数据上，记录数据量–成功率曲线

### 推荐读什么

- [stereovlm](../wiki/entities/paper-stereovlm.md) — 标定双目几何与视觉语言骨干的空间推理。
- [WALL-OSS-1.0：原生 VLA 预训练与多分支动作生成](../wiki/entities/wall-oss-1-0.md) — 自变量 Qwen3.5 原生语言/视觉/动作预训练，Action Expert + RTC、Lite/Nano/Gen/AR 家族；区分 312 项零样本真机 55.1% 与 X2Real 仿真 67.55 分，API 可申请但 1.0 代码/权重未核实。
- [BiCAT：双臂协调动作分词](../wiki/entities/paper-bicat.md) — 共享 token 原语、跨臂注意力与两块动作预测，补齐离散动作接口。
- [VLA](../wiki/methods/vla.md) 与 [VLA 知识链汇总](../wiki/overview/hub-vla.md) — 主线索引页
- [VLA / 世界模型 14 篇阅读路线](../wiki/overview/vla-wm-reading-roadmap-14-papers-technology-map.md) — CLIP→RT→OpenVLA→π₀ + 四篇 WM，每篇独立 `paper-*`
- [Robotics Transformer（RT 系列）](../wiki/methods/robotics-transformer-rt-series.md)、[RT-1](../wiki/entities/paper-rt-1.md)、[RT-2](../wiki/entities/paper-rt-2.md)、[OpenVLA](../wiki/entities/paper-openvla.md)、[Octo](../wiki/entities/paper-octo.md)
- [π0](../wiki/entities/paper-pi0.md) 与 [π0.7](../wiki/methods/pi07-policy.md)
- [StarVLA-α](../wiki/entities/paper-starvla-alpha.md) — 强 VLM 加轻量动作头的受控基线，区分 specialist、generalist 与不同本体的评测协议。
- [SayCan](../wiki/entities/paper-saycan.md) 与 [DIAL 指令增强](../wiki/methods/dial-instruction-augmentation.md)
- [InternVLA-A1.5](../wiki/entities/paper-internvla-a15-unified-vla.md) — 2026 主线前沿对照：统一理解 + 潜式前瞻 + flow matching 动作的单一 MoT 框架，组合泛化与长程执行显著超 π₀.₅；`InternRobotics/InternVLA-A1` `main` 分支与 HF `InternRobotics/internvla-a15` 已开源（CC BY-NC-SA 4.0）
- [LingBot-VLA](../wiki/entities/lingbot-vla.md) — Qwen2.5-VL-3B + flow 动作头，2 万小时双臂真机预训练；开源 4B 权重与 LeRobot v3.0 后训练栈，RoboTwin 仿真领先 π₀.₅
- [KUAVO-VLA-1.0](../wiki/entities/kuavo-vla-1.md) — 乐聚基于 LingBot-VLA-2.0（6B）在 680 h / 100 任务上重训的工业人形垂域模型，25 任务评测达 48.3% SR / 74.5% PS；`LeTools-Learning` 已开源，24 页技术报告与官网已公开，HF 模型与数据 `gated: auto` 需申请审核
- [Robbyant（蚂蚁灵波）](../wiki/entities/robbyant.md) — LingBot-VLA 的发布方：把 VLA 放回 Vision / Depth / Map / Video / World / VA 全栈里看，并与 π₀ / GR00T / Skild Brain 对照
- [BridgeVLA++](../wiki/entities/paper-bridgevla-plusplus.md) — 多视图 heatmap 对齐 3D VLA 加统一时空记忆（粗阶段关键帧检索 + 细阶段初始几何），RMBench 记忆依赖任务 18.9%→96.0%，RLBench 93.7%；代码与权重已开源
- [Galaxea G0.5](../wiki/entities/paper-galaxea-g05.md) — VLM-as-Actor + 学出来的 ActionCodec 27 维去掉自回归 token 税，原生 CoT 直接 attend；真机六设定 76.7% vs π0.5 53.3%，LIBERO 98.9% / RoboTwin 93.3%；GalaxeaVLA + HF 权重（`gated: auto`）已开源（G0.5 Community License，非商用）
- [GSR / ParaVLA](../wiki/entities/paper-gsr-paravla.md) — 指出 VLA 指令改写崩溃来自联合 V-L 路由而非不懂语义，冻结 T5 重绑原生视觉并重训动作专家；LIBERO-Para 上 SmolVLA +44.6 pp；训练与 HF 权重已开源
- [Indi](../wiki/entities/paper-indi.md) — 冻结教师 VLM 把示范片段的局部行为意图蒸馏进动作解码器，部署时无需教师；GR00T-N1.7 在 SimplerEnv-Bridge 64.3%→84.7%，真机 62.0%→68.7%（官方仓暂仅占位 README）
- [GazeVLA：从人类注视学习机器人操作意图](../wiki/entities/paper-gazevla.md) — CoRL 2026 Spotlight，先预测二维注视意图再生成动作；人类视频预训练→人/机混合后训练，机器人侧不需要 gaze 标注，官方权重及评测脚本仍待发布
- [Topology-VLA](../wiki/entities/paper-topology-informed-visual-prompting-vla.md) — 用拓扑签名区分外观相似却需不同绕行的状态，借仿真规划扩增示范，再让 VLM 生成视觉路径提示（未开源）
- [RoboMP-DINOv2](../wiki/entities/paper-robomp-dinov2.md) — 在完整图像特征上加入区域提示，而不是用硬掩码过滤，避免丢掉障碍与背景线索（仅开源视觉编码器与颜色增强，无下游策略/权重）
- [X-Planner](../wiki/entities/paper-x-planner-event-structured-task-planning.md) — 用事件状态衔接高层指令与下游动作模型，把长任务计划写成可执行事件，并探索连续潜在推理接口（推理与静态基准已开源，训练依赖未开内部后端）
- [MIGU](../wiki/entities/paper-migu-multimodal-instruction-grounding.md) — 融合语言语义与指向手势的三维几何不确定性，先形成目标信念，再决定执行或澄清（`MingkeLu/MIGU` 部分开源核心 grounding、样本与 ROS 接口，不含 POMDP 澄清对话与 S2 规划器）
- [VGE：冻结 VLM 引导多目标指代接地](../wiki/entities/paper-vge-2611-22625.md) — Stage 2 感知接地前端：逐层联合注意力 + 指令条件实例评分，稳健恢复零/单/多目标框；RoboRefIt A/B 三指令平均 F1 95.02/89.67%，论文 v1 尚无官方 VGE 代码。
- [VLA-Δ：物理状态转移监督](../wiki/entities/paper-vla-delta.md) — Stage 2：用未来差异文本、结构化转移槽和光流在预训练期监督 Query，再由 flow matching 直接预测动作；LIBERO 98.4%、RoboTwin 90.06%/89.74%；已公开训练/评测代码与权重。
- [AutoVLA-D](../wiki/entities/paper-autovla-d.md) — NeurIPS 2026 驾驶 VLA：用 1D-Brohan 离散差分动作作 TrajCond 中间条件，再回归连续航点；区分离散最终输出与条件输入（官方数据引擎、SFT 配置已发布，GRPO 外置）。
- [TEMPO](../wiki/entities/paper-tempo-dynamic-manipulation.md) — CoRL 2026 Oral；为单帧 VLA 补充视觉运动摘要与动作历史，缓解双臂动态操作歧义；`tempo-robot/TEMPO` 模型代码与 `TempoBench` 257 回合数据已开源，权重与真机部署待补

### 学完输出什么
- 能画出典型 VLA 的三段式结构（视觉编码 → 语义 backbone → 动作专家）并说清各家差异
- 一份自己任务上的 VLA 微调实验记录

---

## Stage 3 数据与 Scaling

**VLA 的瓶颈不在结构在数据：真机演示太贵，人类视频、世界模型、跨本体数据成为主战场。**

### 前置知识
- Stage 2 内容

### 核心问题
- 真机演示之外还有哪些可扩数据源：人类第一视角视频（EgoScale、HumanNet）、互联网视频（mimic-video）
- WAM（World Action Model）如何把"预测未来"与"生成动作"联合建模
- 前向 / 逆动力学解耦预训练（DeFI）解决什么问题
- 具身 Scaling Laws 目前有哪些证据、哪些只是外推

### 推荐做什么
- 按开源复现全景挑一条可在消费级 GPU 上跑通的路线，完整复现一次
- 对比"有 / 无人类视频预训练"的下游微调差距（读论文实验即可）

### 推荐读什么
- [PI 人视频到机器人迁移 / π₀.₅ + ego](../wiki/entities/paper-pi-human-to-robot.md) — 50:50 人机共微调；预训练多样性决定人类示范能否迁移，四任务细分评测指标；本文实验未开源。
- [Awaken-HI：全身人类经验扩展人形 VLA](../wiki/entities/paper-awaken-hi.md) — 50K h 人体视觉与全身运动共同监督共享动作专家；在固定 T800 后训练后验证 Scaling 与移动操作，注意 HF 只公开样例、代码/权重待发布。
- [ScaleVLA：552B 视觉语言骨干的动作化](../wiki/entities/paper-scalevla-2611-21607.md) — 对照十条演示 LIBERO-Goal 87.8%、LIBERO-PRO 45.1% 与单动作块 216 ms，分别核查规模、少样本与推理硬件。
- [SALA](../wiki/entities/paper-sala-latent-actions.md) — 人类视频潜在动作的无配对语义对齐，连接机器人控制锚点与预/后训练扩展
- [VLA 开源复现全景 2025](../wiki/overview/vla-open-source-repro-landscape-2025.md)
- [具身数据金字塔综述](../wiki/entities/paper-data-pyramid-embodied-manipulation.md) — 五层数据生态 × 六维属性的类目级坐标系；「该补哪一层数据」的选型框架
- [EgoScale](../wiki/methods/egoscale.md)、[HumanNet](../wiki/entities/humannet.md)、[mimic-video](../wiki/methods/mimic-video.md)
- [World Action Models（WAM）](../wiki/concepts/world-action-models.md) 与 [Pelican-Unified 1.0](../wiki/methods/pelican-unified-1.md)
- [DeFI](../wiki/methods/defi-decoupled-dynamics-vla.md) 与 [具身 Scaling Laws](../wiki/concepts/embodied-scaling-laws.md)
- [Xiaomi-Robotics-1](../wiki/entities/xiaomi-robotics-1.md) — 10 万小时 embodiment-free UMI 预训练 + 跨本体后训练，验证数据/模型规模双向可预测 scaling；`XiaomiRobotics/Xiaomi-Robotics-1`（Apache-2.0）已开源 5B 权重与评测/后训练代码，10 万小时预训练管线未公开
- [JoyAI-RA 0.5](../wiki/entities/paper-joyai-ra-05.md) — VLWA：latent-action 隐式对齐吃无标签人视频 + 130-D 规范动作显式对齐吃可靠轨迹，双通道监督把人视频当主缩放轴；AgiBot G1 真机 seen 92.0/unseen 75.5 大幅超 π₀.₅（74.0），人视频缩放在最大测试规模仍未见饱和；确认未开源
- [ACE-Data-0](../wiki/entities/paper-ace-data-0.md) — 真实家居双尺度同步度量 HOI/HSI（ego/exo/运动/物体/音频/触觉）；150 h 中规模高保真人类演示层，与 EgoScale/RekaDaily「拼小时」互补；HF 数据集 API 已设为公开非门控（`gated: false`，约 22.18 TB），README 仍留旧 Coming Soon 标记；训练代码未见
- [RoboEdit](../wiki/entities/paper-roboedit.md) — 把人类操作 RGB 视频编辑为物理 plausible 机器人视频 + 3D hand states，自动构造 14M 帧 RoboEdit-14M（7 种本体）；编辑 SOTA + 真机 Franka 下游控制；`v4` 增补项目页 `roboedit.github.io`，仍无公开代码/数据仓
- [Ego2Robot](../wiki/entities/paper-ego2robot.md) — 第一人称人视频经重定向 + 臂合成 + 三级质检，合成 15 形态 18,561 h 机器人数据；与真机共训提升 RoboTwin 解耦 OOD；管线未开源
- [Spirit-v1.5](../wiki/entities/cn-os-spirit-v1-5.md) — [千寻智能](../wiki/entities/spirit-ai.md)开源操作基础模型（`Qwen3-VL-4B` + 18 层 `BaseDiT`，MIT 代码 / Apache-2.0 HF 权重）；主打非脚本 Open-Ended Diverse Collection 替代 Clean Data，微调未见任务所需迭代步数减少 40% 并曾登 RoboChallenge Table30 榜首（预训练全量语料未公开）

### 学完输出什么
- 能说清 VLA 数据金字塔（真机演示 / 仿真 / 人类视频 / 互联网视频）各层的作用与代价
- 一次完整的开源 VLA 复现或消融记录

---

## Stage 4 部署与系统整合

**论文里的成功率不等于产线上的可用性：推理延迟、任务编排与分层接口是 VLA 落地的三大工程问题。**

### 前置知识
- Stage 3 内容

### 核心问题
- 真机部署的工程问题：推理延迟、异步 action chunk 执行（Xiaomi-Robotics-0）
- 延迟与泛化的取舍：大模型更聪明但更慢，边缘侧怎么选（延迟–泛化权衡）
- 多技能长时程任务怎么编排：行为树 + VLA 的分工边界在哪
- 当任务需要人形全身（搬箱、开门带行走）时，VLA 输出什么接口给低层——这是 [BFM 纵深](depth-bfm.md) Stage 4 的正题

### 推荐做什么
- 测量一个开源 VLA 在目标硬件上的端到端延迟（拍照 → 动作下发），画出延迟分解表
- 用行为树把 2–3 个 VLA 技能串成一个长时程任务，观察失败恢复逻辑

### 推荐读什么

- [intrinsic-core](../wiki/entities/intrinsic-core.md) — 本地运行时、硬件抽象和工业部署接口的工程参考。
- [PGX：人形长程任务的物理接地执行](../wiki/entities/paper-pgx-2611-26596.md) — 对照「单个 VLA 承担全任务」：谓词 FSM 负责阶段与恢复，WCL 负责位置精度，同权重 GR00T 咖啡任务 C0 35%→80%；源码暂无公开入口。
- [H-VLA（关键动作→规划）](../wiki/entities/paper-h-vla.md) — 统一双臂相机坐标、关键末端目标与连续规划接口；关注相机标定和阶段间失效。
- [Co-VLA（双臂协同 SAE/LAC）](../wiki/entities/paper-co-vla.md) — 共享/残差双臂结构加执行期修正，逐项对照 Handover 与 OOD 任务的正负增益。
- [VLA 动作到关节力矩的执行链路](../wiki/concepts/vla-action-to-joint-torque.md) — 从模型动作反归一化、IK 与阻抗控制贯通到关节力矩与伺服电流环，核对模式、频率和安全约束。
- [Agentic-Teacher：失败驱动 VLA 后训练](../wiki/entities/paper-agentic-teacher-2611-17394.md)：定向生成纠错演示并通过流匹配损失过滤。
- [PEAR：仿真强化学习训练机器人原生推理模型](../wiki/entities/paper-pear-robot-use-rl.md)：记录大模型动作接口与分布式 rollout 部署前验收。
- [Towards Agentic Robots：模型、数据、环境与 Harness 综述](../wiki/entities/paper-towards-agentic-robots-survey.md) — 固定 arXiv:2610.20442v1 的四域十七类地图；对照 VLA 执行器与工具编排、状态交接、失败恢复及分层实时控制，不是可直接部署的算法。
- [HITCH：VLM 通过主动三维测量与全身 harness 操作 G1](../wiki/entities/paper-hitch.md) — 作为端到端 VLA 的分层接口对照：VLM 选图像点取深度、下发骨盆/躯干/手/行走/视线命令，规划器与 50 Hz 跟踪器执行；只用五个基础任务的仿真教师轨迹微调后，在**作者修改相机与指令的协议**下实现 SIMPLE 51.1%、HumanoidArena 四任务子集 42.5%，不可直接横比官方榜单。
- [ActionCodec2](../wiki/entities/paper-actioncodec2-streaming-action-tokenizer.md) — Stage 4 动作端低延迟：误差反馈量化 + Set-BPE 逐步解码；流式 token 输出与实际闭环执行需分别验证
- [JustRUA](../wiki/entities/paper-justrua-robot-use-agents.md) — RoboDojo 受控对照：实测状态、估深、短运动步进与 Coding Actor
- [ACE-CaP-0](../wiki/entities/paper-ace-cap-0.md) — 开发期按需示范/仿真，冻结代码策略在执行期脱离编码 Agent
- [Reins](../wiki/entities/paper-reins-robot-harness.md) — 多模态 Agent 通过三维感知、条件化动作链与并行旁白驱动真机
- [Uni-Robot-Agent / OpenRUA](../wiki/entities/paper-uni-robot-agent.md) — Stage 4 统一 Tool/环境适配器、带回执的执行与资源调度；同一 Qwen 在 RoboDojo 的 1.90→31.39 是跨 harness 比较而非新权重训练。工程侧可对照已发布 ARX ONE 与仅源码可用的 Piper-X；复现实验应区分 Score/SR、相机标定、深度输入和真机安全条件。
- [STVLA](../wiki/entities/paper-stvla-spatial-temporal-memory.md) — 用空间几何与有界历史记忆改善长任务中遮挡与状态歧义
- [Xiaomi-Robotics-0](../wiki/entities/xiaomi-robotics-0.md) — 异步 action chunk 部署
- [ARP：异步机器人策略](../wiki/entities/paper-asynchronous-robot-policies.md) — 将 VLM 低频语义缓存与自带实时视觉的高频动作专家分进程运行，训练显式建模缓存陈旧和执行延迟；动态扰动更稳，但通用 RoboLab 仍落后 π0.5（官方源码/权重待发布）
- [2026 年 VLA 开源生态三转变](../wiki/comparisons/vla-open-source-ecosystem-2026-three-shifts.md) — 按策略模型、实验框架、统一评测与后训练系统选择复现入口；从同协议基线再到部署/后训练，不把工具框架当成新的 VLA 模型
- [具身模型延迟–泛化权衡](../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md)
- [Evo-1](../wiki/entities/paper-evo1-lightweight-vla.md) — 0.77B 轻量 VLA：两阶段训练保持 VLM 语义对齐，消费级 GPU 2.3 GB / 16.4 Hz，边缘侧选型的代表样本
- [行为树 VLA 编排](../wiki/concepts/behavior-tree-vla-orchestration.md)
- [Harness VLA](../wiki/entities/paper-harness-vla.md) · [GaP](../wiki/entities/paper-gap-graph-as-policy.md) — 对照固定原语和记忆驱动的冻结 VLA 编排、计算图与仿真排练，分别检查工具接口和复现入口。
- [EventVLA](../wiki/entities/paper-eventvla-visual-evidence-memory.md) — 稀疏视觉证据记忆端到端 VLA，用基础锚点 + 前瞻式关键帧预测解决长程操作的记忆瓶颈，是行为树编排之外的模型内记忆路线
- [RoboTTT](../wiki/entities/paper-robottt-test-time-training-vla-context.md) — 在 VLA 层内嵌测试时训练，将 visuomotor 上下文压缩进固定大小 fast weights，扩到约 8K 步且支持部署后在线自纠偏
- [ActFovea](../wiki/entities/paper-actfovea.md) — 不重训、不改权重的 VLA 运行时防护层，用动作条件中央凹 + 时空视觉–动作一致性检测扰动；LIBERO 40 任务上把视觉叠加攻击下的成功率从 49.3% 拉回 90.3%，2000 次重放试验 100% 及时安全失败
- [RoboHarness](../wiki/entities/paper-robo-harness.md) — 把 VLA / RL / TAMP 等异构策略封装为 agentic skills，用理解/记忆/自进化辅助技能做能力边界路由，Memory Bridge 稳定交接；LIBERO-LoHo 成绩与协议见 v3 实体记录；官方仓暂为项目页镜像
- [RTCF](../wiki/entities/paper-rtcf.md) — 免训练测试时纠偏（**⚠️ v3 于 2026-10-28 由作者以实验证据暂不足撤稿；归档核对 v2 全文**）：Progressive Memory Alignment 按执行历史对齐成功轨迹，将低频运动残差转移给冻结 PI-FAST；LIBERO 86.4%→88.4%，LIBERO-Long 61.6%→68.6%，CPU 侧约 11 ms/chunk 额外开销；无公开代码
- [Neural Introspection Gating](../wiki/entities/paper-neural-introspection-gating.md)（IROS 2026）— 训练无关、可插拔的 VLA 推理调度层：用上一步动作 token logit margin 当免费不确定性信号，门控 VLA-Cache 静态 patch 复用，在保留约 80% 算力节省（1.54 vs 1.43 TFLOPs）的同时收回盲缓存在 LIBERO-Long 上的掉点；已开源 Apache-2.0 实现（`zjw4321/gated-vla-cache`）适配 OpenVLA/OpenVLA-OFT
- [VLAQuantBench](../wiki/entities/paper-vlaquantbench.md) — 统一量化格式与作用范围，在闭环任务上比较 VLA 的训练后量化（独立层组与联合量化），把成功率、校准条件与真实算子延迟一并检查（`jiuyixu25/VLAQuantBench` MIT 已开源）。
- [vla.simd](../wiki/entities/paper-vla-simd.md) — 用 SIMD 内核与计算复用加速 CPU 推理，并把查询延迟和动作块长度一起分析（项目页当前仅挂双盲匿名仓库，实名代码与权重录用后公开）。
- [ReflexVLA](../wiki/entities/paper-reflexvla.md) — ReflexBench 六任务延迟感知评测 + 1B VLA（冻结 DINOv3 未来预测 + 时序融合骨干 + CUDA Graph）；均值 50.4%、LIBERO 97.2%；`LxRoboticsLab/ReflexBench` 基准与数据已开源（BSD-3），策略代码录用后开放
- [FlashVLA](../wiki/entities/paper-flashvla.md)（UCSD/MIT）— 给流匹配 VLA 接上交错噪声缓冲 + chunk 级因果注意力，稳态每步吐一块动作，同时压低逐步延迟与异步观测错配；LIBERO 异步 2.43×（96.9%→97.8%）；`z-lab/flashvla` 算法/仿真评测与 HF `π₀.₅` 权重已开源（不含 Franka 实机驱动与 Gello 数据）
- [Legato](../wiki/entities/paper-legato.md)（RSS 2026 · 上交/千寻/清华等）— 训练期以动作–噪声混合与重塑速度场目标学会三段调度动作延续，异步部署时无需推理期 VJP 反向传播、真机 $\pi_{0.5}$ / GR00T N1.6 抑制块间模态切换并缩短执行时间（MIT 仅开源 `Legato-kinetix` 2D 仿真实现）
- [Query：操作 VLA 架构选型](../wiki/queries/manipulation-vla-architecture-selection.md)

### 加速方法与闭环延迟选型

- [FastRoverVLA](../wiki/entities/paper-fastrovervla.md) — 免训练动作缓存、每步速度–曲率对齐与写回；对照冻结预测重放和串行实际请求，尚无闭环驾驶验证。

- [VLA 加速 25 法：模型内部、异步执行、动作头与云边协同](../wiki/comparisons/vla-acceleration-25-methods-2026.md) — 按耗时位置选择优化层，保留硬件、动作可用延迟、任务完成时间与成功率的可比口径（Turing Post 2026-11-22 原文策展）。

### 学完输出什么
- 一份目标平台上的 VLA 部署延迟分解与优化清单
- 能为"单臂桌面任务 / 移动操作 / 人形全身"三类场景分别给出 VLA 的接入方案

---

## Stage 5 进阶方向

- [2026年11月上半月具身智能九篇技术地图](../wiki/overview/embodied-trends-2026-11-h1.md) — 按原语规划、视角鲁棒、动力学表征与遮挡推演选读；统计仅援引原作者。


### 前置知识
- Stage 4 内容

**方向 A：RL 微调与自改进**
- [RoboEval 冠军方案：离策略 RL 与在线推理调优](../wiki/entities/paper-roboeval-solution.md) — RDS 评分优势、共享 critic、Thompson 调参及失败恢复；仿真八项双臂任务的后训练实例。
- [π*₀.₆ / RECAP：VLA 从自主经验中学习](../wiki/entities/paper-rcl-2511-14759-0-6-a-vla-that-learns-from-experience.md) — 离线 RL 预训练 + 真机 rollout/人工纠正 + 分布式价值函数与优势条件；用成功率及每小时成功次数同时衡量专精效果（官方完整训练代码未开放）。
- [TidyBot](../wiki/entities/paper-tidybot-2305-05658.md)、[OLAF](../wiki/entities/paper-olaf-verbal-correction.md)、[LMPC](../wiki/entities/paper-lmpc-2402-11450.md) — 对照规则总结、动作重标注和语言模型微调三种适应对象。
- [PAFF](../wiki/entities/paper-paff-2212-07398.md) — 用基础模型反馈重标注目标域交互，再监督微调策略；对照 RL 与重标注驱动的适应。
- 用 RL / 真机数据闭环继续改进预训练策略
- [PEAR / Robot-Use RL](../wiki/entities/paper-pear-robot-use-rl.md) — 无机器人示范、在仿真中强化学习多轮视觉推理及末端位姿操作；对照传统动作块 VLA 的训练信号与执行延迟。
- 关键词：[ENPIRE](../wiki/methods/enpire.md)、[安全真机 RL 微调](../wiki/concepts/safe-real-world-rl-fine-tuning.md)、[STEAM](../wiki/entities/paper-steam-advantage-modeling.md)（自监督时序 advantage 离线提纯 π₀，已集成于官方 `RLinf`）、[WCM](../wiki/entities/paper-wcm-world-critic-model.md)（世界模型 critic 修正 VLA RL 单帧价值估计错配，4 基准 149 任务提升显著；`sylvestf/WCM` 已 MIT 开源 critic，RL 策略代码待发）、[TEMPO](../wiki/entities/paper-tempo.md)（语义 projection 低频 / action expert 高频双 TD3；CALVIN SR5 81.7%，匿名页 404 且未开源）、[AutoIntervene](../wiki/entities/paper-autointervene.md)（chunk 策略视觉–动作支持校准接管，选择性 DAgger；算法未开源，同名仓为静态页镜像）、[Prism-GRPO](../wiki/entities/paper-prism-grpo.md)（success+quality 打破 Binary GRPO 同结果组退化，RoboTwin rollout 最多 −56%；独立补丁未开源）

**方向 B：世界模型融合**
- [WEVO：视频世界模型驱动全身 VLA 后训练](../wiki/entities/paper-wevo-world-model-enabled-vla-loco-manipulation.md) — LEGS 合成演示初始化 + 每任务 20 条留存自主真机轨迹对齐 DreamDojo，SARM 密集奖励和 DICE-RL 在 WM 中迭代优化；G1 三任务平均真机成功率 23.3%→75.0%，与 VLAW 的成功想象轨迹 BC 对比。
- [HyWorldVLA：自动驾驶 VLA 的像素–潜变量混合世界建模](../wiki/entities/paper-hyworldvla.md) — 预训练时联合未来像素 token 与 latent 监督，规划时仅预测未来 latent；NAVSIM v1 PDMS 90.59、v2 EPDMS 89.71。**驾驶域**案例，非桌面操作策略。
- 把"预测未来"并入策略训练或推理时预演——完整 Stage 路径见 [WAM 纵深路线](depth-wam.md)
- [VLAW：VLA 与世界模型迭代协同后训练](../wiki/entities/paper-rcl-2602-12063-vlaw-iterative-co-improvement-of-vision-language.md) — 少量真机成功/失败 rollout 校准 Ctrl-World；Qwen3-VL 奖励筛选合成成功轨迹后微调 π₀.₅，属于训练期世界模型辅助而非 Joint WAM。
- 关键词：[Generative World Models](../wiki/methods/generative-world-models.md)、[World Action Models](../wiki/concepts/world-action-models.md)、[WAM 纵深](depth-wam.md)、[τ₀-VLA](../wiki/entities/paper-tau0-vla.md)（记忆增强高层子任务策略 + 世界模型引导 TTC beam search，长程真机分层 45.0% vs 直出 27.5%；`sii-research/tau-0-vla` 已开源低层 MoT、Proposal 与世界模型 LoRA，Value/Reflective/Router/TTC 搜索未公开）

**方向 C：全身与移动操作扩展**
- 把 VLA 从桌面机械臂扩展到全身移动操作
- 关键词：[VLA 与世界模型（loco-manip 161 分类）](../wiki/overview/loco-manip-161-category-09-vla-world-models.md)、[Loco-Manipulation 纵深路线](depth-loco-manipulation.md)、[BFM 纵深路线](depth-bfm.md)

**方向 D：导航 VLA**
- 把语言接地从"怎么动手"扩展到"往哪里走"
- 关键词：[视觉–语言导航（VLN）](../wiki/tasks/vision-language-navigation.md)、[导航纵深路线](depth-navigation.md)

---

## 快速入口汇总

| 阶段 | 核心问题 | 知识页入口 |
|------|---------|-----------|
| Stage 0 | 具身基础模型分类学 | [VLM/VLN/VLA/VLX/世界模型分类学](../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) |
| Stage 1 | 模仿学习策略基座 | [Diffusion Policy](../wiki/methods/diffusion-policy.md) |
| Stage 2 | VLA 主线 | [VLA](../wiki/methods/vla.md) |
| Stage 3 | 数据与 Scaling | [VLA 开源复现全景 2025](../wiki/overview/vla-open-source-repro-landscape-2025.md) |
| Stage 4 | 部署与整合 | [Xiaomi-Robotics-0](../wiki/entities/xiaomi-robotics-0.md) |
| Stage 5 | 进阶方向 | [ENPIRE](../wiki/methods/enpire.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制 → 物理智能全栈成长路线](motion-control.md)
- 相关纵深路线（按主题邻近，完整目录见 [路线总览](motion-control.md#depth-optional-index)）：
  - [模仿学习](depth-imitation-learning.md) — VLA Stage 1 重叠的 BC/ACT/DP 底座
  - [WAM](depth-wam.md) — 出动作前显式预知世界
  - [导航](depth-navigation.md) — 导航 VLA / VLN
  - [具身数据](depth-embodied-data.md) — 跨具身语料与配比
  - [ICL](depth-icl.md) — 不动权重的上下文适应
  - [模型架构](depth-model-architecture.md) — 横切视角：token 化方案、动作头形式与时延预算怎么选

## 参考来源

- [VLA](../wiki/methods/vla.md) 与 [VLA 知识链汇总](../wiki/overview/hub-vla.md)
- [VLA 开源复现全景 2025](../wiki/overview/vla-open-source-repro-landscape-2025.md)
- "RT-2: Vision-Language-Action Models" (Brohan et al., 2023) — VLA 概念确立
- "π0: A Vision-Language-Action Flow Model" (Black et al., 2024) — flow matching 动作专家代表
- "OpenVLA: An Open-Source Vision-Language-Action Model" (Kim et al., 2024) — 开源 VLA 与 OXE 跨本体路线
- [GlanceWAM / VLA Crew 10 篇技术地图（2026-08-30 策展）](../wiki/overview/glancewam-vla-crew-10-papers-technology-map.md)
