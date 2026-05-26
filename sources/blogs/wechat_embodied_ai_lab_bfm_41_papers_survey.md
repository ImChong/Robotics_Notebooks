# 智元、众擎都在卷的人形机器人运控基座：41篇论文看懂BFM

> 来源归档（blog / 微信公众号）

- **标题：** 智元、众擎都在卷的人形机器人运控基座：41篇论文看懂BFM
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g
- **发表日期：** 2026-05-26（frontmatter）
- **入库日期：** 2026-05-26
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 + `wechat-article-for-ai`（Camoufox）；正文约 2.7 万字 / 43 图；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **配套仓库：** [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers)（41 篇论文表 + 10 个数据集）
- **配套综述：** [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487)（IEEE TPAMI 2025）
- **关联姊妹篇：** [42 篇 humanoid RL 运动控制「身体系统栈」](wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)、[19 篇 AMP 运动先验专题](wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
- **一句话说明：** 按 **五个问题**（forward-backward 表征、goal-conditioned 覆盖面、intrinsic reward 预训练、adaptation、层次化控制）串读 awesome-bfm-papers 所列 **41 篇** BFM 相关论文；核心判断：BFM 把人形运控从「单技能训练」推向 **可复用、可适配、可调用的身体接口**；智元公开 **BFM-2 运控基座** 叙事与众擎年度 demo 的多动作/抗扰/起身能力，分别体现「明牌押注」与「需求侧验证」。

## 核心摘录（归纳，非全文）

### 问题重框

- **BFM ≠ 更大动作库**：价值在把「动作能力」推进到 **身体接口层**——VLA、世界模型、语言规划最终都要落到走、平衡、起身、接触、扰动恢复等 **可调用身体能力**。
- **读法**：不按时间线堆摘要，而按综述 taxonomy 的五类问题组织；41 篇共同处理 **让机器人身体成为上层智能可复用、适配、调用的底座**。
- **产业语境（文内观点）**：
  - **智元**：把 BFM-2 推为「运控基座模型」，并预告 BFM-3（官方产品叙事）。
  - **众擎**：文内不写「已官方押注 BFM」，但年度视频中的多动作拼接、长时程稳定、倒地起身、抗扰恢复，与「运控基座 / 行为基座」路线需求高度重合。

### 五个问题（对应 awesome 列表分组）

| 组 | 篇数 | 核心问题 | 代表论文 |
|----|------|----------|----------|
| **01 Forward-backward 表征** | 6 | 多任务能否压进 **可调用的身体潜空间** | BFM-Zero、MetaMotivo、FB-AW、Fast Imitation、Learning One Representation、Successor States |
| **02 Goal-conditioned 学习** | 19 | 动作跟踪、全身技能、遥操作、HOI 的 **覆盖面** | SONIC、OpenTrack、AMS、TWIST/TWIST2、BFM4Humanoid、HOVER、InterMimic、MaskedMimic、ASE/CALM/CASE… |
| **03 Intrinsic reward 预训练** | 5 | 无明确任务时身体应先积累 **可迁移探索经验** | APS、Proto-RL、RE3、RND、DIAYN |
| **04 Adaptation** | 3 | 预训练 BFM 如何 **低成本** 进新任务/新动力学/新机 | Task Tokens、Unseen Dynamics、Fast Adaptation |
| **05 Hierarchical control** | 8 | 语言、VLA、扩散、规划器如何 **调用底层身体** | SENTINEL、BeyondMimic、LeVerb、LangWBC、TokenHSI、CLoSD、UniPhys、UniHSI |

### 文内收束判断

- BFM 是 **AMP/运动先验**（自然身体分布）+ **motion tracking / mimic / 遥操作**（动作与交互数据）+ **VLA / 语言 / 扩散 / 规划**（身体 API 化）三股线的交汇。
- **数据侧**（不计入 41 篇）：Humanoid-X、PHUMA、Motion-X++、AMASS、HumanML3D、BABEL 等；关键不在「动作越多越好」，而在能否加工成 **机器人可信、可执行、可迁移** 的训练材料。
- **后续拆解优先级（作者计划）**：BFM-Zero / FB 线；HOVER / MaskedMimic / SONIC 跟踪线；SENTINEL / LangWBC / LeVerb 语言–身体接口线；Humanoid-X / Motion-X++ 数据线。

## 41 篇论文索引（标题以抓取版为准）

### 01 — Forward-backward 表征（6）

| # | 标题 | 年份 | 链接 |
|---|------|------|------|
| 01 | BFM-Zero: Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised RL | 2025 | https://arxiv.org/abs/2511.04131 |
| 02 | Zero-shot Whole-body Humanoid Control via Behavioral Foundation Models (metamotivo) | 2025 | https://arxiv.org/abs/2504.11054 |
| 03 | Finer Behavioral Foundation Models via Auto-regressive Features and Advantage Weighting | 2024 | https://arxiv.org/abs/2412.04368 |
| 04 | Fast Imitation via Behavior Foundation Models | 2024 | NeurIPS |
| 05 | Learning One Representation to Optimize All Rewards | 2021 | NeurIPS |
| 06 | Learning Successor States and Goal-Dependent Values: A Mathematical Viewpoint | 2021 | arXiv |

### 02 — Goal-conditioned 学习（19）

| # | 标题 | 年份 | 链接 |
|---|------|------|------|
| 07 | Sonic: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control | 2025 | https://arxiv.org/abs/2511.07820 |
| 08 | Track Any Motions under Any Disturbances (OpenTrack) | 2025 | https://arxiv.org/abs/2509.13833 |
| 09 | Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data (AMS) | 2025 | https://arxiv.org/abs/2511.17373 |
| 10 | TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System | 2025 | https://arxiv.org/abs/2505.02833 |
| 11 | TWIST: Teleoperated Whole-Body Imitation System | 2025 | CoRL |
| 12 | CLONE: Closed-Loop Whole-Body Humanoid Teleoperation for Long-Horizon Tasks | 2025 | CoRL |
| 13 | Behavior Foundation Model for Humanoid Robots | 2025 | https://arxiv.org/pdf/2509.13780 |
| 14 | HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots | 2025 | https://arxiv.org/abs/2410.21229 |
| 15 | InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions | 2025 | https://arxiv.org/abs/2502.20390 |
| 16 | ModSkill: Physical Character Skill Modularization | 2025 | https://arxiv.org/abs/2502.14140 |
| 17 | MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting | 2024 | TOG |
| 18 | H-GAP: Humanoid Control with a Generalist Planner | 2024 | https://arxiv.org/abs/2312.02682 |
| 19 | CALM: Conditional Adversarial Latent Models for Directable Virtual Characters | 2024 | SIGGRAPH |
| 20 | MoConVQ: Unified Physics-Based Motion Control via Scalable Discrete Representations | 2023 | TOG |
| 21 | CASE: Learning Conditional Adversarial Skill Embeddings for Physics-Based Characters | 2023 | SIGGRAPH Asia |
| 22 | PHC: Perpetual Humanoid Control for Real-Time Simulated Avatars | 2023 | ICCV |
| 23 | TeamPlay: From Motor Control to Team Play in Simulated Humanoid Football | 2021 | Science Robotics |
| 24 | MTM: Masked Trajectory Models for Prediction, Representation, and Control | 2023 | ICML |
| 25 | ASE: Large-scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters | 2022 | TOG |

### 03 — Intrinsic reward 预训练（5）

| # | 标题 | 年份 | 链接 |
|---|------|------|------|
| 26 | Active Pretraining with Successor Features (APS) | 2021 | ICML |
| 27 | Reinforcement Learning with Prototypical Representations (Proto-RL) | 2021 | ICML |
| 28 | State Entropy Maximization with Random Encoders for Efficient Exploration (RE3) | 2020 | ICML |
| 29 | Exploration by Random Network Distillation (RND) | 2019 | ICLR |
| 30 | Diversity is All You Need: Learning Skills without a Reward Function (DIAYN) | 2018 | ICLR |

### 04 — Adaptation（3）

| # | 标题 | 年份 | 链接 |
|---|------|------|------|
| 31 | Task Tokens: A Flexible Approach to Adapting Behavior Foundation Models | 2025 | https://arxiv.org/abs/2503.22886 |
| 32 | Zero-Shot Adaptation of Behavioral Foundation Models to Unseen Dynamics | 2025 | https://arxiv.org/abs/2505.13150 |
| 33 | Fast Adaptation With Behavioral Foundation Models | 2025 | CoRL |

### 05 — Hierarchical control（8）

| # | 标题 | 年份 | 链接 |
|---|------|------|------|
| 34 | SENTINEL: A Fully End-to-End Language-Action Model for Humanoid Whole Body Control | 2025 | https://arxiv.org/abs/2511.19236 |
| 35 | BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion | 2025 | https://arxiv.org/abs/2508.08241 |
| 36 | LeVerb: Humanoid Whole-Body Control with Latent Vision-Language Instruction | 2025 | https://arxiv.org/abs/2506.13751 |
| 37 | LangWBC: Language-Directed Humanoid Whole-Body Control via End-to-end Learning | 2025 | https://arxiv.org/abs/2504.21738 |
| 38 | Tokenhsi: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization | 2025 | https://arxiv.org/abs/2503.19901 |
| 39 | CloSD: Closing the Loop between Simulation and Diffusion for Multi-task Character Control | 2024 | https://arxiv.org/abs/2410.03441 |
| 40 | UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-based Character Control | 2024 | https://arxiv.org/abs/2504.12540 |
| 41 | Unified Human-Scene Interaction via Prompted Chain-of-Contacts (UniHSI) | 2023 | https://arxiv.org/abs/2309.07918 |

### 数据集（文内附录，10 项，不计入 41 篇）

AMASS、KIT-ML、LAFAN、BABEL、HumanML3D、PoseScript、Motion-X、Motion-X++、PHUMA、Humanoid-X — 规模与入口见 awesome-bfm-papers README；与本库 [AMASS](../../wiki/entities/amass.md) 等实体互参。

## 对 wiki 的映射

- [bfm_awesome_41_catalog.md](../papers/bfm_awesome_41_catalog.md)（41 论文 + 10 数据集分别入库的 source 总索引）
- [bfm-41-papers-technology-map](../../wiki/overview/bfm-41-papers-technology-map.md)（本次升格主页面）
- [behavior-foundation-model](../../wiki/concepts/behavior-foundation-model.md)、[paper-behavior-foundation-model-humanoid](../../wiki/entities/paper-behavior-foundation-model-humanoid.md)
- [humanoid-rl-motion-control-body-system-stack](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md)、[awesome_bfm_papers](../repos/awesome_bfm_papers.md)

## 可信度与使用边界

- 公众号为 **策展导航 + 产业观察**；41 篇逐条点评以抓取正文为准，公式与指标请对照 arXiv / 项目页。
- 智元/众擎产品叙事为 **文内解读**，非官方技术白皮书；商业宣传与商务信息已剥离。
- 微信 CDN 图不入库；完整论文表以 [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 为准。
