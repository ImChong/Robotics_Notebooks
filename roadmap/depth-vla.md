# 路线（纵深）：如果目标是 VLA（视觉-语言-动作模型）

**摘要**：面向"想让机器人听懂指令干活"的纵深路线，从具身模型分类学、模仿学习策略基座，到 VLA 语义策略主线（RT 系列 → OpenVLA → π0）、数据与 Scaling，再到部署整合与进阶方向，按 Stage 0–5 串通核心方法；本路线是 [运动控制主路线](motion-control.md) 的一条分支，与 [BFM 纵深](depth-bfm.md) 构成"任务级语义 / 身体级协调"的姊妹路线。

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
- [VLM / VLN / VLA / VLX / 世界模型分类学](../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md)（本仓库）
- [Foundation Policy](../wiki/concepts/foundation-policy.md)（本仓库）— VLA 的母概念页
- [具身基础模型纵深](../wiki/overview/hub-embodied-foundation-model.md)（本仓库）
- [Query：具身大模型家族分类学闭环](../wiki/queries/embodied-fm-taxonomy-loop.md)（本仓库）
- [LLMs-from-scratch（Raschka）](../wiki/entities/llms-from-scratch-raschka.md)（本仓库）— **可选前置**：Transformer/GPT 从零实现；配套 [YouTube 播放列表](https://www.youtube.com/playlist?list=PLTKMiZHVd_2IIEsoJrWACkIxLRdfMlw11)
- [Karpathy Zero to Hero（YouTube）](../wiki/entities/andrej-karpathy.md)（本仓库）— **可选技术轨前置**：[10 集播放列表](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)（micrograd → GPT → GPT-2）；配套 [`nn-zero-to-hero`](https://github.com/karpathy/nn-zero-to-hero)
- [Andrej Karpathy LLM 科普（YouTube）](../wiki/entities/andrej-karpathy.md)（本仓库）— **可选前置（偏直觉）**：[Intro to LLMs（~1 h）](https://www.youtube.com/watch?v=zjkBMFhNj_g) → [Deep Dive into LLMs（~3.5 h）](https://www.youtube.com/watch?v=7xTGNNLPyMI)；建立 pretrain/SFT/RLHF、工具调用与上下文窗口心智模型，再读 VLA 论文更省力

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
- [Action Chunking](../wiki/methods/action-chunking.md) 与 [BC with Transformer](../wiki/methods/bc-with-transformer.md)（本仓库）
- [Diffusion Policy](../wiki/methods/diffusion-policy.md) 与 [Diffusion Model](../wiki/concepts/diffusion-model.md)（本仓库）
- [Imitation Learning](../wiki/methods/imitation-learning.md)（本仓库）

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
- [VLA](../wiki/methods/vla.md) 与 [VLA 知识链汇总](../wiki/overview/hub-vla.md)（本仓库）— 主线索引页
- [VLA / 世界模型 14 篇阅读路线](../wiki/overview/vla-wm-reading-roadmap-14-papers-technology-map.md)（本仓库）— CLIP→RT→OpenVLA→π₀ + 四篇 WM，每篇独立 `paper-*`
- [Robotics Transformer（RT 系列）](../wiki/methods/robotics-transformer-rt-series.md)、[RT-1](../wiki/entities/paper-rt-1.md)、[RT-2](../wiki/entities/paper-rt-2.md)、[OpenVLA](../wiki/entities/paper-openvla.md)、[Octo](../wiki/entities/paper-octo.md)（本仓库）
- [π0](../wiki/entities/paper-pi0.md) 与 [π0.7](../wiki/methods/pi07-policy.md)（本仓库）
- [SayCan](../wiki/methods/saycan.md) 与 [DIAL 指令增强](../wiki/methods/dial-instruction-augmentation.md)（本仓库）
- [InternVLA-A1.5](../wiki/entities/paper-internvla-a15-unified-vla.md)（本仓库）— 2026 主线前沿对照：统一理解 + 潜式前瞻 + flow matching 动作的单一 MoT 框架，组合泛化与长程执行显著超 π₀.₅
- [LingBot-VLA](../wiki/entities/lingbot-vla.md)（本仓库）— Qwen2.5-VL-3B + flow 动作头，2 万小时双臂真机预训练；开源 4B 权重与 LeRobot v3.0 后训练栈，RoboTwin 仿真领先 π₀.₅
- [BridgeVLA++](../wiki/entities/paper-bridgevla-plusplus.md)（本仓库）— 多视图 heatmap 对齐 3D VLA 加统一时空记忆（粗阶段关键帧检索 + 细阶段初始几何），RMBench 记忆依赖任务 18.9%→96.0%，RLBench 93.7%；代码与权重已开源
- [Galaxea G0.5](../wiki/entities/paper-galaxea-g05.md)（本仓库）— VLM-as-Actor + 学出来的 ActionCodec 27 维去掉自回归 token 税，原生 CoT 直接 attend；真机六设定 76.7% vs π0.5 53.3%，LIBERO 98.9% / RoboTwin 93.3%；GalaxeaVLA + HF 权重已开源（G0.5 Community License，非商用）
- [GSR / ParaVLA](../wiki/entities/paper-gsr-paravla.md)（本仓库）— 指出 VLA 指令改写崩溃来自联合 V-L 路由而非不懂语义，冻结 T5 重绑原生视觉并重训动作专家；LIBERO-Para 上 SmolVLA +44.6 pp；训练与 HF 权重已开源
- [Indi](../wiki/entities/paper-indi.md)（本仓库）— 冻结教师 VLM 把示范片段的局部行为意图蒸馏进动作解码器，部署时无需教师；GR00T-N1.7 在 SimplerEnv-Bridge 64.3%→84.7%，真机 62.0%→68.7%

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
- [VLA 开源复现全景 2025](../wiki/overview/vla-open-source-repro-landscape-2025.md)（本仓库）
- [具身数据金字塔综述](../wiki/entities/paper-data-pyramid-embodied-manipulation.md)（本仓库）— 五层数据生态 × 六维属性的类目级坐标系；「该补哪一层数据」的选型框架
- [EgoScale](../wiki/methods/egoscale.md)、[HumanNet](../wiki/entities/humannet.md)、[mimic-video](../wiki/methods/mimic-video.md)（本仓库）
- [World Action Models（WAM）](../wiki/concepts/world-action-models.md) 与 [Pelican-Unified 1.0](../wiki/methods/pelican-unified-1.md)（本仓库）
- [DeFI](../wiki/methods/defi-decoupled-dynamics-vla.md) 与 [具身 Scaling Laws](../wiki/concepts/embodied-scaling-laws.md)（本仓库）
- [Xiaomi-Robotics-1](../wiki/entities/xiaomi-robotics-1.md)（本仓库）— 10 万小时 embodiment-free UMI 预训练 + 跨本体后训练，验证数据/模型规模双向可预测 scaling，预训练收益直接迁移到未见环境开箱成功率
- [JoyAI-RA 0.5](../wiki/entities/paper-joyai-ra-05.md)（本仓库）— VLWA：latent-action 隐式对齐吃无标签人视频 + 130-D 规范动作显式对齐吃可靠轨迹，双通道监督把人视频当主缩放轴；AgiBot G1 真机 seen 92.0/unseen 75.5 大幅超 π₀.₅（74.0），人视频缩放在最大测试规模仍未见饱和；确认未开源
- [ACE-Data-0](../wiki/entities/paper-ace-data-0.md)（本仓库）— 真实家居双尺度同步度量 HOI/HSI（ego/exo/运动/物体/音频/触觉）；150 h 中规模高保真人类演示层，与 EgoScale/RekaDaily「拼小时」互补；HF gated 研究许可，训练代码未见
- [RoboEdit](../wiki/entities/paper-roboedit.md)（本仓库）— 把人类操作 RGB 视频编辑为物理 plausible 机器人视频 + 3D hand states，自动构造 14M 帧 RoboEdit-14M（7 种本体）；编辑 SOTA + 真机 Franka 下游控制；无官方代码
- [Ego2Robot](../wiki/entities/paper-ego2robot.md)（本仓库）— 第一人称人视频经重定向 + 臂合成 + 三级质检，合成 15 形态 18,561 h 机器人数据；与真机共训提升 RoboTwin 解耦 OOD；管线未开源

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
- [Xiaomi-Robotics-0](../wiki/entities/xiaomi-robotics-0.md)（本仓库）— 异步 action chunk 部署
- [具身模型延迟–泛化权衡](../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md)（本仓库）
- [Evo-1](../wiki/entities/paper-evo1-lightweight-vla.md)（本仓库）— 0.77B 轻量 VLA：两阶段训练保持 VLM 语义对齐，消费级 GPU 2.3 GB / 16.4 Hz，边缘侧选型的代表样本
- [行为树 VLA 编排](../wiki/concepts/behavior-tree-vla-orchestration.md)（本仓库）
- [EventVLA](../wiki/entities/paper-eventvla-visual-evidence-memory.md)（本仓库）— 稀疏视觉证据记忆端到端 VLA，用基础锚点 + 前瞻式关键帧预测解决长程操作的记忆瓶颈，是行为树编排之外的模型内记忆路线
- [RoboTTT](../wiki/entities/paper-robottt-test-time-training-vla-context.md)（本仓库）— 在 VLA 层内嵌测试时训练，将 visuomotor 上下文压缩进固定大小 fast weights，扩到约 8K 步且支持部署后在线自纠偏
- [ActFovea](../wiki/entities/paper-actfovea.md)（本仓库）— 不重训、不改权重的 VLA 运行时防护层，用动作条件中央凹 + 时空视觉–动作一致性检测扰动；LIBERO 40 任务上把视觉叠加攻击下的成功率从 49.3% 拉回 90.3%，2000 次重放试验 100% 及时安全失败
- [RoboHarness](../wiki/entities/paper-robo-harness.md)（本仓库）— 把 VLA / RL / TAMP 等异构策略封装为 agentic skills，用理解/记忆/自进化辅助技能做能力边界路由，Memory Bridge 稳定交接；LIBERO-LoHo 上 95.2% 成功，远超 π₀.₅ 的 6.4%；官方仓暂为项目页镜像
- [RTCF](../wiki/entities/paper-rtcf.md)（本仓库）— 免训练测试时纠偏：Progressive Memory Alignment 按执行历史对齐成功轨迹，只把低频运动残差转移给冻结的 PI-FAST；LIBERO 86.4%→88.4%，LIBERO-Long 61.6%→68.6%，CPU 侧约 11 ms/chunk 额外开销；截至入库日无公开代码
- [Neural Introspection Gating](../wiki/entities/paper-neural-introspection-gating.md)（本仓库，IROS 2026）— 训练无关、可插拔的 VLA 推理调度层：用上一步动作 token logit margin 当免费不确定性信号，门控 VLA-Cache 静态 patch 复用，在保留约 80% 算力节省（1.54 vs 1.43 TFLOPs）的同时收回盲缓存在 LIBERO-Long 上的掉点；适配已部署的 OpenVLA/OpenVLA-OFT；确认未开源
- [ReflexVLA](../wiki/entities/paper-reflexvla.md)（本仓库）— ReflexBench 六任务延迟感知评测 + 1B VLA（冻结 DINOv3 未来预测 + 时序融合骨干 + CUDA Graph）；均值 50.4%、LIBERO 97.2%；代码录用后开放
- [Query：操作 VLA 架构选型](../wiki/queries/manipulation-vla-architecture-selection.md)（本仓库）

### 学完输出什么
- 一份目标平台上的 VLA 部署延迟分解与优化清单
- 能为"单臂桌面任务 / 移动操作 / 人形全身"三类场景分别给出 VLA 的接入方案

---

## Stage 5 进阶方向

### 前置知识
- Stage 4 内容

**方向 A：RL 微调与自改进**
- 用 RL / 真机数据闭环继续改进预训练策略
- 关键词：[ENPIRE](../wiki/methods/enpire.md)、[安全真机 RL 微调](../wiki/concepts/safe-real-world-rl-fine-tuning.md)、[STEAM](../wiki/entities/paper-steam-advantage-modeling.md)（自监督时序 advantage 离线提纯 π₀，无需在线 rollout 与人工标注）、[WCM](../wiki/entities/paper-wcm-world-critic-model.md)（世界模型 critic 修正 VLA RL 单帧价值估计错配，4 基准 149 任务上大幅提升 π₀/π₀.₅/OpenVLA-OFT，OOD 增益尤明显）、[TEMPO](../wiki/entities/paper-tempo.md)（语义 projection 低频 / action expert 高频双 TD3；CALVIN SR5 81.7%，未开源）、[AutoIntervene](../wiki/entities/paper-autointervene.md)（chunk 策略视觉–动作支持校准接管，选择性 DAgger）、[Prism-GRPO](../wiki/entities/paper-prism-grpo.md)（success+quality 打破 Binary GRPO 同结果组退化，RoboTwin rollout 最多 −56%）

**方向 B：世界模型融合**
- 把"预测未来"并入策略训练或推理时预演——完整 Stage 路径见 [WAM 纵深路线](depth-wam.md)
- 关键词：[Generative World Models](../wiki/methods/generative-world-models.md)、[World Action Models](../wiki/concepts/world-action-models.md)、[WAM 纵深](depth-wam.md)、[τ₀-VLA](../wiki/entities/paper-tau0-vla.md)（记忆增强高层子任务策略 + 世界模型引导 TTC beam search，长程真机分层 45.0% vs 直出 27.5%）

**方向 C：全身与移动操作扩展**
- 把 VLA 从桌面机械臂扩展到全身移动操作
- 关键词：[VLA 与世界模型（loco-manip 161 分类）](../wiki/overview/loco-manip-161-category-09-vla-world-models.md)、[Loco-Manipulation 纵深路线](depth-loco-manipulation.md)、[BFM 纵深路线](depth-bfm.md)

**方向 D：导航 VLA**
- 把语言接地从"怎么动手"扩展到"往哪里走"
- 关键词：[视觉–语言导航（VLN）](../wiki/tasks/vision-language-navigation.md)、[导航纵深路线](depth-navigation.md)

---

## 快速入口汇总

| 阶段 | 核心问题 | 本仓库入口 |
|------|---------|-----------|
| Stage 0 | 具身基础模型分类学 | [VLM/VLN/VLA/VLX/世界模型分类学](../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) |
| Stage 1 | 模仿学习策略基座 | [Diffusion Policy](../wiki/methods/diffusion-policy.md) |
| Stage 2 | VLA 主线 | [VLA](../wiki/methods/vla.md) |
| Stage 3 | 数据与 Scaling | [VLA 开源复现全景 2025](../wiki/overview/vla-open-source-repro-landscape-2025.md) |
| Stage 4 | 部署与整合 | [Xiaomi-Robotics-0](../wiki/entities/xiaomi-robotics-0.md) |
| Stage 5 | 进阶方向 | [ENPIRE](../wiki/methods/enpire.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制算法工程师成长路线](motion-control.md)
- 其它纵深路径：
  - [遥操作（人形全身遥操作 + 手指遥操作 → 示范数据/实时接管）](depth-teleoperation.md)
  - [BFM（人形行为基础模型）](depth-bfm.md) — 姊妹路线：VLA 管任务级语义，BFM 管身体级协调
  - [具身模型测评（认知 → 世界模型保真 → 策略成功率 → sim↔real 校准）](depth-embodied-eval.md) — 验收环节：VLA 成功率/泛化怎么测才不骗人
  - [WAM（世界–动作模型）](depth-wam.md) — 姊妹路线：VLA 管反应式语义策略，WAM 管前向后果耦合
  - [模仿学习与技能迁移](depth-imitation-learning.md) — 本路线 Stage 1 的展开版
  - [Loco-Manipulation（移动操作）](depth-loco-manipulation.md) — Stage 5 方向 C 的展开版
  - [导航（SLAM → VLN → 导航 VLA）](depth-navigation.md) — Stage 5 方向 D 的展开版
  - [动作生成（文本/多模态 → 人形动作）](depth-motion-generation.md) — 语义接口与分层设计的邻接路线
  - [动作重定向（人体动作 → 机器人参考轨迹）](depth-motion-retargeting.md)
  - [人形 RL 运动控制](depth-rl-locomotion.md)
  - [力矩控制电机设计（指标 → 电磁热 → FOC 力矩闭环）](depth-torque-motor-design.md)
  - [传统模型控制（LIP/ZMP → MPC → WBC）](depth-classical-control.md)
  - [人形整机硬件设计（指标预算 → 机械 → 电气 → 通信 → 整机验收）](depth-humanoid-hardware-design.md)
  - [安全控制（CLF/CBF）](depth-safe-control.md)
  - [接触丰富的操作任务](depth-contact-manipulation.md)
  - [感知越障（Perceptive Locomotion）](depth-perceptive-locomotion.md)
  - [人形足球（全向行走 → 感知踢球 → 多机战术）](depth-humanoid-soccer.md)
  - [人形群控展演（群舞同步 → 编队走位 → 群体特技）](depth-humanoid-swarm-performance.md)
  - [人形拳击（动作跟踪 → 潜空间技能 → 对抗自博弈）](depth-humanoid-boxing.md)
  - [Sim2Real（域差画像 → 执行器对齐 → 鲁棒训练 → 真机部署）](depth-sim2real.md)
  - [Real2Sim（真实世界 → 可仿真资产/场景/孪生）](depth-real2sim.md)
  - [ICL（具身上下文学习）](depth-icl.md) — 部署期适应旋钮：不动权重、读一条示范就换映射
- 人形控制全景图：[Humanoid Control Roadmap](../wiki/roadmaps/humanoid-control-roadmap.md)
- 技术栈地图：[tech-map/dependency-graph.md](../tech-map/dependency-graph.md)

## 参考来源

本路线基于以下原始资料的归纳：

- [VLA](../wiki/methods/vla.md) 与 [VLA 知识链汇总](../wiki/overview/hub-vla.md)
- [VLA 开源复现全景 2025](../wiki/overview/vla-open-source-repro-landscape-2025.md)
- "RT-2: Vision-Language-Action Models" (Brohan et al., 2023) — VLA 概念确立
- "π0: A Vision-Language-Action Flow Model" (Black et al., 2024) — flow matching 动作专家代表
- "OpenVLA: An Open-Source Vision-Language-Action Model" (Kim et al., 2024) — 开源 VLA 与 OXE 跨本体路线
- [GlanceWAM / VLA Crew 10 篇技术地图（2026-08-30 策展）](../wiki/overview/glancewam-vla-crew-10-papers-technology-map.md)
