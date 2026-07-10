# 路线（纵深）：如果目标是 VLA 与 BFM（具身基础模型）

**摘要**：面向"想让机器人听懂指令干活（VLA）、想用一个 checkpoint 控住人形全身（BFM）"的纵深路线，从模仿学习策略基座到 VLA 语义策略主线、BFM 行为先验主线，再到高层 VLA + 低层 BFM 的层次化整合，按 Stage 0–6 串通核心方法；本路线是 [运动控制主路线](motion-control.md) 的一条分支。

## 路线一览

```mermaid
flowchart LR
  S0["**Stage 0**<br/>全景与前置<br/><em>VLM/VLN/VLA/BFM 分类学</em>"]
  S1["**Stage 1**<br/>策略基座<br/><em>BC / ACT / Diffusion Policy</em>"]
  S2["**Stage 2**<br/>VLA 主线<br/><em>RT 系列 → OpenVLA → π0</em>"]
  S3["**Stage 3**<br/>数据与 Scaling<br/><em>跨本体 / 人类视频 / WAM</em>"]
  S4["**Stage 4**<br/>BFM 主线<br/><em>人形全身行为先验</em>"]
  S5["**Stage 5**<br/>双栈整合<br/><em>高层 VLA + 低层 BFM</em>"]
  S6["**Stage 6**<br/>进阶方向<br/><em>RL 微调 / 世界模型 / 评测</em>"]

  S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6

  classDef stage fill:#142a3a,stroke:#f1c40f,stroke-width:2px,color:#fff
  class S0,S1,S2,S3,S4,S5,S6 stage
```

## 这条路径怎么用

- 目标读者是有深度学习基础、想进入"具身基础模型"方向的人——不管是操作向 VLA 还是人形全身 BFM
- 先记住两条主线的分工：**VLA 解决任务级语义**（看图、听指令、出动作），**BFM 解决身体级协调**（人形全身控制的可复用行为先验）；工程上二者常按"高层 VLA → 低层 BFM"叠成一个栈
- 每个阶段都有前置知识、核心问题、推荐做什么、推荐读什么、学完输出什么

**和主路线的关系：**
- 本路线是主路线 L5（RL 与模仿学习）之后偏"学习侧"的进阶方向，Stage 0–1 与 [模仿学习纵深](depth-imitation-learning.md) 的策略基座高度重叠
- BFM 主线大量复用 [RL 纵深](depth-rl-locomotion.md) 的训练经验（PPO、reward 设计、sim2real）
- 如果只关心"让机械臂听话干活"，走完 Stage 3 即可；Stage 4–5 面向人形全身控制

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
- [Foundation Policy](../wiki/concepts/foundation-policy.md) 与 [Behavior Foundation Model](../wiki/concepts/behavior-foundation-model.md)（本仓库）— 两条主线的母概念页
- [具身基础模型专题](../wiki/overview/topic-embodied-foundation-model.md)（本仓库）
- [Query：具身大模型家族分类学闭环](../wiki/queries/embodied-fm-taxonomy-loop.md)（本仓库）

### 学完输出什么
- 能一句话说清 VLA 和 BFM 分别是什么、不是什么
- 拿到一篇新论文能放进 VLM / VLN / VLA / WAM / BFM 的正确格子里

---

## Stage 1 模仿学习策略基座

**VLA 的"动作头"和 BFM 的蒸馏管线都建在这一层上。走过 [模仿学习纵深](depth-imitation-learning.md) Stage 0–3 的可以跳。**

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

**VLA 概念由 RT-2（2023）确立：把 VLM 的语义能力直接接到机器人动作上。这是本路线的第一条主干。**

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
- [VLA](../wiki/methods/vla.md) 与 [VLA 专题](../wiki/overview/topic-vla.md)（本仓库）— 主线索引页
- [Robotics Transformer（RT 系列）](../wiki/methods/robotics-transformer-rt-series.md)、[OpenVLA](../wiki/entities/openvla.md)、[Octo](../wiki/methods/octo-model.md)（本仓库）
- [π0](../wiki/methods/π0-policy.md) 与 [π0.7](../wiki/methods/pi07-policy.md)（本仓库）
- [SayCan](../wiki/methods/saycan.md) 与 [DIAL 指令增强](../wiki/methods/dial-instruction-augmentation.md)（本仓库）

### 学完输出什么
- 能画出典型 VLA 的三段式结构（视觉编码 → 语义 backbone → 动作专家）并说清各家差异
- 一份自己任务上的 VLA 微调实验记录

---

## Stage 3 VLA 进阶：数据、Scaling 与部署

**VLA 的瓶颈不在结构在数据：真机演示太贵，人类视频、世界模型、跨本体数据成为主战场。**

### 前置知识
- Stage 2 内容

### 核心问题
- 真机演示之外还有哪些可扩数据源：人类第一视角视频（EgoScale、HumanNet）、互联网视频（mimic-video）
- WAM（World Action Model）如何把"预测未来"与"生成动作"联合建模
- 前向 / 逆动力学解耦预训练（DeFI）解决什么问题
- 真机部署的工程问题：推理延迟、异步 action chunk 执行（Xiaomi-Robotics-0）

### 推荐做什么
- 按开源复现全景挑一条可在消费级 GPU 上跑通的路线，完整复现一次
- 对比"有 / 无人类视频预训练"的下游微调差距（读论文实验即可）

### 推荐读什么
- [VLA 开源复现全景 2025](../wiki/overview/vla-open-source-repro-landscape-2025.md)（本仓库）
- [EgoScale](../wiki/methods/egoscale.md)、[HumanNet](../wiki/entities/humannet.md)、[mimic-video](../wiki/methods/mimic-video.md)（本仓库）
- [World Action Models（WAM）](../wiki/concepts/world-action-models.md) 与 [Pelican-Unified 1.0](../wiki/methods/pelican-unified-1.md)（本仓库）
- [DeFI](../wiki/methods/defi-decoupled-dynamics-vla.md) 与 [Xiaomi-Robotics-0](../wiki/entities/xiaomi-robotics-0.md)（本仓库）

### 学完输出什么
- 能说清 VLA 数据金字塔（真机演示 / 仿真 / 人类视频 / 互联网视频）各层的作用与代价
- 一次完整的开源 VLA 复现或消融记录

---

## Stage 4 BFM 主线：人形全身行为先验

**换战场：从"机械臂听指令"到"人形全身协调"。BFM 的目标是一个 checkpoint 覆盖跟踪、抗扰与多接口控制。**

### 前置知识
- [RL 纵深路线](depth-rl-locomotion.md) Stage 0–2 水平：能在仿真里训练 locomotion 策略
- 了解动捕数据（[AMASS](../wiki/entities/amass.md)）与 motion retargeting 基本概念

### 核心问题
- BFM 与任务专用 RL 的本质区别：跨任务行为先验 + 少 / 零重训适应
- 预训练三线的信号与代价：goal-conditioned（跟踪驱动）、intrinsic-reward（技能发现）、forward–backward（无 reward 表征）
- 跟踪主线谱系：DeepMimic → ASE → PHC → MaskedMimic → HOVER 如何一步步走向"多模式统一"
- 适应两线：微调（LoRA / task token）vs 层次化（高层规划 + BFM 低层执行）

### 推荐做什么
- 用 PHC / MaskedMimic 开源实现在仿真里跑一个人形动作跟踪 demo
- 精读 BFM 综述 taxonomy，把 41 篇论文地图过一遍，标出自己方向的 3 篇精读

### 推荐读什么
- [Behavior Foundation Model](../wiki/concepts/behavior-foundation-model.md)（本仓库）— taxonomy 主入口
- [BFM 41 篇技术地图](../wiki/overview/bfm-41-papers-technology-map.md) 与五个分类页：[FB 表征](../wiki/overview/bfm-category-01-forward-backward-representation.md)、[Goal-conditioned](../wiki/overview/bfm-category-02-goal-conditioned-learning.md)、[Intrinsic-reward](../wiki/overview/bfm-category-03-intrinsic-reward-pretraining.md)、[Adaptation](../wiki/overview/bfm-category-04-adaptation.md)、[Hierarchical](../wiki/overview/bfm-category-05-hierarchical-control.md)（本仓库）
- 代表工作：[BFM4Humanoid](../wiki/entities/paper-behavior-foundation-model-humanoid.md)、[SONIC](../wiki/methods/sonic-motion-tracking.md)、[BFM-Zero](../wiki/entities/paper-bfm-zero.md)、[MetaMotivo](../wiki/entities/paper-bfm-02-metamotivo.md)（本仓库）
- 谱系锚点：[ASE](../wiki/entities/paper-bfm-25-ase.md)、[PHC](../wiki/entities/paper-bfm-22-phc.md)、[MaskedMimic](../wiki/entities/paper-bfm-17-maskedmimic.md)、[HOVER](../wiki/entities/paper-bfm-14-hover.md)、[DIAYN](../wiki/entities/paper-bfm-30-diayn.md)（本仓库）
- 数据侧：[Humanoid-X](../wiki/entities/dataset-bfm-humanoid-x.md) 等 dataset-bfm 系列（本仓库）

### 学完输出什么
- 能把任意一篇 BFM 论文放进"预训练三线 × 适应两线"的格子
- 一个跑通的人形动作跟踪 demo，以及对多模式控制接口（掩码 / 潜变量）的直觉

---

## Stage 5 双栈整合：高层 VLA + 低层 BFM

**当前人形系统的主流工程形态：语义决策与全身执行分层，各自用擅长的数据训练。**

### 前置知识
- Stage 3 与 Stage 4 内容

### 核心问题
- 为什么端到端"语言 → 全身关节"目前难以直接训练（数据稀缺、控制频率、安全性）
- 层次化接口怎么设计：文本 / 潜向量 / motion token / 参考轨迹，各自的表达力与带宽
- LangWBC、LeVERB、GR00T-WholeBodyControl 各自的分层切法差在哪
- 开环级联的 exposure bias 怎么闭环（ReactiveBFM）

### 推荐做什么
- 读 GR00T-WholeBodyControl / LeVERB 的接口定义，画出各自的分层数据流图
- 在仿真里把一个语言指令 pipeline 接到动作跟踪低层（哪怕先用有限状态机中转）

### 推荐读什么
- [LangWBC](../wiki/entities/paper-bfm-37-langwbc.md) 与 [LeVERB](../wiki/entities/paper-bfm-36-leverb.md)（本仓库）
- [GR00T-WholeBodyControl](../wiki/entities/gr00t-wholebodycontrol.md) 与 [Humanoid-VLA](../wiki/entities/paper-loco-manip-161-121-humanoid-vla.md)（本仓库）
- [ReactiveBFM](../wiki/entities/paper-reactivebfm.md) 与 [Perceptive BFM](../wiki/entities/paper-perceptive-bfm.md)（本仓库）
- [行为树 VLA 编排](../wiki/concepts/behavior-tree-vla-orchestration.md)（本仓库）

### 学完输出什么
- 能为"人形 + 语言任务"设计一套分层方案，并说清接口选型的取舍
- 对"什么该端到端、什么该分层"有基于数据 / 频率 / 安全的判断

---

## Stage 6 进阶方向

### 前置知识
- Stage 5 内容

**方向 A：RL 微调与自改进**
- 用 RL / 真机数据闭环继续改进预训练策略
- 关键词：[ENPIRE](../wiki/methods/enpire.md)、[安全真机 RL 微调](../wiki/concepts/safe-real-world-rl-fine-tuning.md)

**方向 B：世界模型融合**
- 把"预测未来"并入策略训练或推理时预演
- 关键词：[Generative World Models](../wiki/methods/generative-world-models.md)、[World Action Models](../wiki/concepts/world-action-models.md)

**方向 C：感知增强与 loco-manipulation**
- 让 BFM 带上环境感知、把 VLA 扩展到全身移动操作
- 关键词：[Perceptive BFM](../wiki/entities/paper-perceptive-bfm.md)、[VLA 与世界模型（loco-manip 161 分类）](../wiki/overview/loco-manip-161-category-09-vla-world-models.md)、[Loco-Manipulation](../wiki/tasks/loco-manipulation.md)

**方向 D：评测、选型与工业级案例**
- 建立自己的评测基线，跟踪工业界的整机方案
- 关键词：[Query：人形动作跟踪方法选型](../wiki/queries/humanoid-motion-tracking-method-selection.md)、[AgiBot BFM-2](../wiki/entities/agibot-bfm-2.md)

---

## 快速入口汇总

| 阶段 | 核心问题 | 本仓库入口 |
|------|---------|-----------|
| Stage 0 | 具身基础模型分类学 | [VLM/VLN/VLA/VLX/世界模型分类学](../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) |
| Stage 1 | 模仿学习策略基座 | [Diffusion Policy](../wiki/methods/diffusion-policy.md) |
| Stage 2 | VLA 主线 | [VLA](../wiki/methods/vla.md) |
| Stage 3 | 数据与 Scaling | [VLA 开源复现全景 2025](../wiki/overview/vla-open-source-repro-landscape-2025.md) |
| Stage 4 | BFM 行为先验 | [Behavior Foundation Model](../wiki/concepts/behavior-foundation-model.md) |
| Stage 5 | 层次化整合 | [GR00T-WholeBodyControl](../wiki/entities/gr00t-wholebodycontrol.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制算法工程师成长路线](motion-control.md)
- 其它纵深路径：
  - [模仿学习与技能迁移](depth-imitation-learning.md) — 本路线 Stage 1 的展开版
  - [人形 RL 运动控制](depth-rl-locomotion.md) — BFM 主线的训练侧前置
  - [传统模型控制（LIP/ZMP → MPC → WBC）](depth-classical-control.md)
  - [安全控制（CLF/CBF）](depth-safe-control.md)
  - [接触丰富的操作任务](depth-contact-manipulation.md)
  - [感知越障（Perceptive Locomotion）](depth-perceptive-locomotion.md)
- 人形控制全景图：[Humanoid Control Roadmap](../wiki/roadmaps/humanoid-control-roadmap.md)
- 技术栈地图：[tech-map/dependency-graph.md](../tech-map/dependency-graph.md)

## 参考来源

本路线基于以下原始资料的归纳：

- [VLA](../wiki/methods/vla.md) 与 [VLA 专题](../wiki/overview/topic-vla.md)
- [Behavior Foundation Model](../wiki/concepts/behavior-foundation-model.md) 与 [BFM 41 篇技术地图](../wiki/overview/bfm-41-papers-technology-map.md)
- "RT-2: Vision-Language-Action Models" (Brohan et al., 2023) — VLA 概念确立
- "π0: A Vision-Language-Action Flow Model" (Black et al., 2024) — flow matching 动作专家代表
- "A Survey of Behavior Foundation Model" (Yuan et al., 2025, arXiv:2506.20487) — BFM taxonomy 主参考
