# awesome-bfm-papers

- **URL（本次 ingest）：** <https://github.com/friedrichyuan/awesome-bfm-papers>
- **综述配套维护仓（arXiv 摘要标注）：** <https://github.com/yuanmingqi/awesome-bfm-papers>
- **Maintainer：** Yuan Mingqi 等（与 TPAMI 2025 BFM 综述作者团队一致）
- **类型：** repo / curated list
- **关联论文：** [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487)（IEEE TPAMI 2025）
- **入库日期：** 2026-05-26
- **最近刷新：** 2026-07-11（列表增至 **42 篇**；Adaptation 微调组新增 Any2Any arXiv:2605.23733）
- **Tags：** #bfm #behavior-foundation-model #humanoid #whole-body-control #survey #curated-list

## 一句话说明

面向 **Behavior(al) Foundation Model（BFM）** 的 **持续更新论文/项目精选列表**，与 Yuan 等 TPAMI 综述共用 taxonomy（预训练三线 + 适应两线），是人形 WBC 基础模型方向的 **外部索引入口**。

## 核心内容

### 定义（README）

BFM 从 **大规模、多样化行为数据** 学习 **广义行为先验（broad behavior priors）**，再 **便捷适应** 多种下游任务——与 NLP 基础模型「大规模预训练 → 下游微调/提示」范式类比，但对象是人形 **全身控制（WBC）** 而非静态语料。

### 分类结构（与综述 Fig.3 对齐）

**预训练（Pre-training）**

| 子类 | 要点 | 列表代表（2025 快照） |
|------|------|------------------------|
| **Forward-backward 表示学习** | 无外在 reward 的转移上训 FEN/BEN；测试时与具体 reward 组合推策略 | BFM-Zero、MetaMotivo、FB / FB-IL 系列 |
| **Goal-conditioned 学习** | 外在 reward + 大规模人体/动捕数据；跟踪、多模式 WBC、遥操作 | SONIC、Any2Track、AMS、BFM4Humanoid、HOVER、MaskedMimic、ASE/CALM/CASE、PHC、TWIST 系列 |
| **Intrinsic reward 驱动** | 自监督内在奖励探索技能空间（DIAYN、RND、ICM 等） | 多为技能发现经典，与人形 WBC 列表中作历史脉络 |

**适应（Adaptation）**

| 子类 | 要点 | 列表代表 |
|------|------|----------|
| **微调技术** | FFT、LoRA、潜空间 / task token、跨具身 PEFT 适应 | Task Tokens、Fast Adaptation with BFM、Zero-Shot Dynamics Adaptation、**Any2Any**（arXiv:2605.23733） |
| **层次化控制** | 高层规划器（LLM / 扩散）→ BFM 低层执行 | SENTINEL、BeyondMimic、LangWBC、LeVERB、CloSD、UniPhys、TokenHSI |

### 数据集表（README § Datasets）

覆盖 **AMASS、HumanML3D、BABEL、LAFAN、Motion-X / Motion-X++、PHUMA、Humanoid-X** 等人体/人形运动语料规模与入口——与本库 [AMASS](../../wiki/entities/amass.md) 实体及 motion tracking 数据链互参。

### 与本库已有条目的重叠（不必重复造页）

下列工作在本库 **已有** wiki 页或 methods 页，本 source 主要作 **谱系索引**：

- [BFM（arXiv:2509.13780）](../../wiki/entities/paper-behavior-foundation-model-humanoid.md)
- [SONIC](../../wiki/methods/sonic-motion-tracking.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md)、[AMS](../../wiki/methods/ams.md)（若已收录）
- [Foundation Policy](../../wiki/concepts/foundation-policy.md)、[Whole-Body Control](../../wiki/concepts/whole-body-control.md)

### 值得后续单篇 ingest 的候选（列表有、本库尚无独立页）

- **BFM-Zero**（arXiv:2511.04131，LeCAR-Lab）— FB 表示 + 无监督 RL，与 CVAE-BFM 对照
- **MetaMotivo / Zero-shot Whole-body Humanoid Control**（arXiv:2504.11054）
- **SENTINEL、LangWBC、LeVERB** — 语言–全身层次化栈
- **HOVER、MaskedMimic、InterMimic** — 多模式 / 掩码 / HOI 向 WBC

## 对 wiki 的映射

- 沉淀概念页：[`wiki/concepts/behavior-foundation-model.md`](../../wiki/concepts/behavior-foundation-model.md) — BFM 定义、三线预训练 + 两线适应 taxonomy、与本库实体互链
- 综述实体页：[`wiki/entities/paper-bfm-survey-tpami-2025.md`](../../wiki/entities/paper-bfm-survey-tpami-2025.md) — TPAMI 2025 综述归纳（arXiv:2506.20487）
- 公众号五类问题导读：[`wiki/overview/bfm-41-papers-technology-map.md`](../../wiki/overview/bfm-41-papers-technology-map.md)（源：[`wechat_embodied_ai_lab_bfm_41_papers_survey.md`](../blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)）
- **分篇 source 索引（2026-05-27）：** [`bfm_awesome_41_catalog.md`](../papers/bfm_awesome_41_catalog.md) — 41 论文 + 10 数据集各对应 `papers/bfm_awesome_*.md`
- 交叉更新：
  - [`wiki/concepts/foundation-policy.md`](../../wiki/concepts/foundation-policy.md) — 区分 VLA 操作向 vs BFM 人形 WBC 向
  - [`wiki/entities/paper-behavior-foundation-model-humanoid.md`](../../wiki/entities/paper-behavior-foundation-model-humanoid.md) — 回链综述与精选列表
  - [`wiki/concepts/whole-body-control.md`](../../wiki/concepts/whole-body-control.md) — Learning-based BFM 段落补 taxonomy 入口

## 参考来源（原始）

- 仓库 README：<https://github.com/friedrichyuan/awesome-bfm-papers/blob/main/README.md>（2026-07-11 抓取）
- 配套综述 PDF：<https://arxiv.org/pdf/2506.20487>
