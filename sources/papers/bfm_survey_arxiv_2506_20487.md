# BFM 综述：A Survey of Behavior Foundation Model（arXiv:2506.20487）

> 来源归档（ingest）

- **标题：** A Survey of Behavior Foundation Model: Next-Generation Whole-Body Control System of Humanoid Robots
- **类型：** survey / humanoid whole-body control
- **arXiv：** <https://arxiv.org/abs/2506.20487>
- **PDF：** <https://arxiv.org/pdf/2506.20487>
- **期刊：** IEEE Transactions on Pattern Analysis and Machine Intelligence（TPAMI），2025
- **作者：** Mingqi Yuan, Tao Yu, Wenqi Ge 等（LimX Dynamics、东方理工、港大、EPFL、南科大、浙大等）
- **配套精选列表：** <https://github.com/yuanmingqi/awesome-bfm-papers>（用户给定镜像：<https://github.com/friedrichyuan/awesome-bfm-papers>）
- **入库日期：** 2026-05-26
- **最近刷新：** 2026-07-11（对照 awesome 列表 42 篇；升格 wiki 实体页）
- **一句话说明：** 首篇系统梳理 **人形 WBC 行为基础模型（BFM）** 的综述：从传统模型法 / 任务专用学习法演进到 **大规模预训练行为先验 + 零样本或快速适应**；按 **goal-conditioned、intrinsic-reward、forward-backward** 三线预训练与 **微调、层次化控制** 两线适应组织文献，并讨论数据集、真机部署与开放问题。

## 摘要级要点

- **动机：** 人形 WBC 面临高维动力学、欠驱动、多任务耦合；任务专用 RL/IL **换场景就要重训**，难以规模化。
- **BFM 定义（综述扩展）：** 在动态环境中控制智能体行为的 **基础模型子类**——在 **大规模行为数据**（演示、交互轨迹等）上预训练，编码 **可复用原语技能与广义行为先验**，测试时对广泛 reward / 目标 **近似最优或快速适应**。
- **与 VLA 的分工：** VLA 整合视觉–语言–动作，多面向 **操作/相对稳定平台**；BFM 主攻 **locomotion、操作、交互的全身低层控制**，面向 **全尺寸人形** 的复杂 WBC。
- **演化叙事（Fig.2）：** 传统 MPC/QP-WBC → 学习式任务专用控制器（DeepMimic、AMP、遥操作 IL）→ **BFM 通用ist**。

## 预训练 taxonomy（Section III-A，与 awesome 列表一致）

### 1）Goal-conditioned learning

- 策略显式条件于 **目标状态 / 目标嵌入**；跟踪类把 dense reference（关节角、姿态）作为逐步目标，比「整段模仿」更易泛化。
- **代表脉络：** TeamPlay → ASE → CALM/CASE；PHC、MaskedMimic、InterMimic；真机 * 标记：HOVER、TWIST/TWIST2、AMS、Any2Track、**BFM4Humanoid**、**SONIC** 等。
- **本库映射：** [BFM 论文实体](../../wiki/entities/paper-behavior-foundation-model-humanoid.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[humanoid motion tracking 选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)。

### 2）Intrinsic reward-driven learning

- 优化 $\sum \gamma^t r^{int}(s_t)$，用好奇心、技能发现、覆盖度等 **自监督信号** 探索；经典含 DIAYN、RND、ICM、ProtoRL、RE3。
- 与人形 WBC 列表中多作 **技能发现历史线**，真机 BFM 条目较少。

### 3）Forward-backward representation learning

- 在 **无 reward** 转移上学 **前向嵌入 F** 与 **后向嵌入 B**；测试时与具体 reward 组合推断策略，支持模仿、奖励推断、分布匹配等多 IL 范式。
- **代表：** FB、FB-IL、FB-AWARE、Motivo、**BFM-Zero**（人形真机 *）。
- **与 CVAE-BFM 对照：** 本库 [BFM 实体页](../../wiki/entities/paper-behavior-foundation-model-humanoid.md) 已述 **特权蒸馏 + 掩码 CVAE** vs **无监督 RL + FB** 谱系两端。

## 适应 taxonomy（Section III-B）

| 路线 | 机制 | 代表 |
|------|------|------|
| **微调** | FFT、LoRA、潜空间 / belief 修改、Task Tokens | TokenHSI、ReLA、LoLA、Belief-FB |
| **层次化控制** | 高层 LLM/扩散生成子任务 → BFM 低层跟踪执行 | UniHSI、CloSD、UniPhys、LangWBC、LeVERB、BeyondMimic、SENTINEL |

## 数据集（Table / awesome § Datasets）

综述与列表强调 **AMASS、HumanML3D、BABEL、LAFAN1、Motion-X、PoseScript、KIT-ML** 及 2025 的 **Humanoid-X、PHUMA、Motion-X++** 等——支撑 goal-conditioned 与文本条件运动生成的 **规模与多样性**。

## 局限与开放问题（综述 Section IV–V 摘要）

- 数据成本、Sim2Real、安全与伦理、**层次栈接口标准化**（语言/子目标 → 低层 BFM）仍不成熟。
- **Locomotion 向「基础模型」** 与操作向 VLA 的 **统一评测** 尚未建立。

## 对 wiki 的映射

- 沉淀：[`wiki/concepts/behavior-foundation-model.md`](../../wiki/concepts/behavior-foundation-model.md)
- 综述实体：[`wiki/entities/paper-bfm-survey-tpami-2025.md`](../../wiki/entities/paper-bfm-survey-tpami-2025.md)
- 原始列表：[`sources/repos/awesome_bfm_papers.md`](../repos/awesome_bfm_papers.md)
- 互链升级：foundation-policy、whole-body-control、paper-behavior-foundation-model-humanoid、humanoid-rl-motion-control-body-system-stack

## 当前提炼状态

- [x] 摘要、BFM 定义、双 taxonomy、与 VLA 分工
- [x] 与本库已有 BFM / SONIC 页的对照指针
- [x] 升格 wiki 实体页 `paper-bfm-survey-tpami-2025.md`（2026-07-11）
- [ ] 单篇深读 BFM-Zero / MetaMotivo（留待后续 ingest）
