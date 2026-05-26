---
type: overview
tags: [bfm, behavior-foundation-model, humanoid, whole-body-control, survey, motion-tracking, foundation-model]
status: complete
updated: 2026-05-26
related:
  - ../concepts/behavior-foundation-model.md
  - ../entities/paper-behavior-foundation-model-humanoid.md
  - ./humanoid-rl-motion-control-body-system-stack.md
  - ./humanoid-amp-motion-prior-survey.md
  - ../methods/sonic-motion-tracking.md
  - ../methods/beyondmimic.md
  - ../methods/ams.md
  - ../methods/any2track.md
  - ../queries/humanoid-motion-tracking-method-selection.md
  - ../concepts/foundation-policy.md
sources:
  - ../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md
  - ../../sources/repos/awesome_bfm_papers.md
  - ../../sources/papers/bfm_survey_arxiv_2506_20487.md
  - ../../sources/papers/bfm_awesome_41_catalog.md
summary: "依据具身智能研究室 BFM 专题长文，把 awesome-bfm-papers 所列 41 篇论文整理为「五类问题」技术地图；核心判断：BFM 把人形运控从技能训练推向可复用、可适配、可调用的身体接口，与智元 BFM-2 运控基座叙事及众擎类 demo 需求同向。"
---

# BFM 技术地图：41 篇论文的五类问题视角

> **本页定位**：为 [具身智能研究室 · BFM 41 篇专题](https://mp.weixin.qq.com/s/Ei32la_vo0UW9Y_QCAqB2g) 提供 **按五类问题组织的阅读坐标**；不复述每篇论文细节，只保留 **产业语境、五组论文地图、与 taxonomy / 身体系统栈的挂接**。概念定义与 Mermaid taxonomy 见 [Behavior Foundation Model](../concepts/behavior-foundation-model.md)；姊妹篇 [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)、[AMP 运动先验综述](./humanoid-amp-motion-prior-survey.md)。

## 一句话观点

**BFM 最值得看的，不是「动作库更大」，而是把身体能力做成上层智能可调用的接口**——走、平衡、起身、接触、抗扰恢复要先在底层封装好，语言 / VLA / 世界模型 / 规划器才能稳定调用；41 篇论文共同回答 **「身体如何成为运控基座」**，而非单点 demo。

## 为什么单独做这张地图

- [BFM 概念页](../concepts/behavior-foundation-model.md) 已给出 **预训练三线 + 适应两线** 的 taxonomy（对齐 TPAMI 2025 综述）。
- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) 提供 **活索引**，但读者仍需要 **按问题线索** 的导航——公众号长文把 41 篇全部串读，并补上 **智元 BFM-2 / 众擎 demo** 等产业观察。
- 与 [八层身体系统栈](./humanoid-rl-motion-control-body-system-stack.md) 的关系：八层栈回答「humanoid RL 系统在搭什么」；本页回答 **「运控基座 / BFM」这条横切面在 2025–2026 的论文簇如何铺开」**。

## 流程总览：五类问题 → 身体 API

```mermaid
flowchart TB
  subgraph G1["01 Forward-backward（6）"]
    FBZ["BFM-Zero / MetaMotivo\n潜空间 prompt · zero-shot"]
    FBT["FB-AW / Fast Imitation\n细粒度表征 · 快速模仿"]
  end
  subgraph G2["02 Goal-conditioned（19）"]
    TRK["SONIC / OpenTrack / AMS\n跟踪覆盖面 · 抗扰"]
    DATA["TWIST / TWIST2 / CLONE\n遥操作 · 数据采集"]
    SYS["BFM4Humanoid / HOVER / MaskedMimic\n多接口全身控制"]
  end
  subgraph G3["03 Intrinsic reward（5）"]
    EXP["APS / Proto-RL / RE3 / RND / DIAYN\n无任务预探索"]
  end
  subgraph G4["04 Adaptation（3）"]
    ADP["Task Tokens / Unseen Dynamics\nFast Adaptation"]
  end
  subgraph G5["05 Hierarchical（8）"]
    LANG["SENTINEL / LangWBC / LeVerb\n语言–身体"]
    GEN["BeyondMimic / CLoSD / UniPhys\n扩散 · 规划–控制"]
  end
  AGI["上层：VLA · 世界模型 · 任务规划"]
  G1 --> BODY["BFM checkpoint / 身体潜空间"]
  G2 --> BODY
  G3 --> BODY
  BODY --> ADP
  ADP --> G5
  G5 --> AGI
  AGI -.->|"调用"| BODY
```

## 产业语境（策展，非官方技术规格）

| 主体 | 文内观察 | 与本页关系 |
|------|----------|------------|
| **智元** | 公开把 **BFM-2** 推为「运控基座模型」，预告 BFM-3 | 与 **01 FB 线 + 02 跟踪覆盖面** 叙事直接对齐 |
| **众擎** | 年度 demo：多动作拼接、长时程、倒地起身、抗扰 | 文内视为「运控基座需求侧验证」，**不写成已官方冠名 BFM** |
| **学术索引** | [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) + [综述 arXiv:2506.20487](https://arxiv.org/abs/2506.20487) | 41 篇编号与分组以仓库 README 为准 |

## 原始资料索引（41 论文 + 10 数据集）

- **总表：** [bfm_awesome_41_catalog.md](../../sources/papers/bfm_awesome_41_catalog.md) — 每篇/每项对应独立 `sources/papers/bfm_awesome_<slug>.md`（策展摘录 + 公众号导读要点，非 PDF 全文）。
- **Wiki 实体：** 每篇/每项均有站内详情页，见下节 [Wiki 实体索引](#wiki-实体索引站内详情页)；图谱与搜索已收录。
- **编译导读：** [wechat_embodied_ai_lab_bfm_41_papers_survey.md](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)
- **深读已有：** #13 → [bfm_humanoid_arxiv_2509_13780.md](../../sources/papers/bfm_humanoid_arxiv_2509_13780.md)；AMASS → [amass-dataset.md](../../sources/sites/amass-dataset.md)

## Wiki 实体索引（站内详情页）

> 41 篇论文与 10 个数据集均已升格为 `wiki/entities/` 详情页（可搜索、进图谱）；#13 与 AMASS 复用既有深读页。

### 论文（41）

| # | 工作 | Wiki 实体 | Source |
|---|------|-----------|--------|
| 01 | BFM-Zero | [paper-bfm-01-bfm-zero](../entities/paper-bfm-01-bfm-zero.md) | [source](../../sources/papers/bfm_awesome_bfm_zero_arxiv_2511_04131.md) |
| 02 | Zero-shot Whole-body Humanoid Control via Behavioral Foundation Models | [paper-bfm-02-metamotivo](../entities/paper-bfm-02-metamotivo.md) | [source](../../sources/papers/bfm_awesome_metamotivo_arxiv_2504_11054.md) |
| 03 | Finer Behavioral Foundation Models via Auto-regressive Features and Advantage Weighting | [paper-bfm-03-fb-aw](../entities/paper-bfm-03-fb-aw.md) | [source](../../sources/papers/bfm_awesome_fb_aw_arxiv_2412_04368.md) |
| 04 | Fast Imitation via Behavior Foundation Models | [paper-bfm-04-fast-imitation-bfm](../entities/paper-bfm-04-fast-imitation-bfm.md) | [source](../../sources/papers/bfm_awesome_fast_imitation_bfm_neurips_2024.md) |
| 05 | Learning One Representation to Optimize All Rewards | [paper-bfm-05-learning-one-representation](../entities/paper-bfm-05-learning-one-representation.md) | [source](../../sources/papers/bfm_awesome_learning_one_representation_neurips_2021.md) |
| 06 | Learning Successor States and Goal-Dependent Values | [paper-bfm-06-successor-states](../entities/paper-bfm-06-successor-states.md) | [source](../../sources/papers/bfm_awesome_successor_states_arxiv_2101_07123.md) |
| 07 | Sonic | [paper-bfm-07-sonic](../entities/paper-bfm-07-sonic.md) | [source](../../sources/papers/bfm_awesome_sonic_arxiv_2511_07820.md) |
| 08 | Track Any Motions under Any Disturbances | [paper-bfm-08-opentrack](../entities/paper-bfm-08-opentrack.md) | [source](../../sources/papers/bfm_awesome_opentrack_arxiv_2509_13833.md) |
| 09 | Agility Meets Stability | [paper-bfm-09-ams](../entities/paper-bfm-09-ams.md) | [source](../../sources/papers/bfm_awesome_ams_arxiv_2511_17373.md) |
| 10 | TWIST2 | [paper-bfm-10-twist2](../entities/paper-bfm-10-twist2.md) | [source](../../sources/papers/bfm_awesome_twist2_arxiv_2505_02833.md) |
| 11 | TWIST | [paper-bfm-11-twist](../entities/paper-bfm-11-twist.md) | [source](../../sources/papers/bfm_awesome_twist_corl_2025.md) |
| 12 | CLONE | [paper-bfm-12-clone](../entities/paper-bfm-12-clone.md) | [source](../../sources/papers/bfm_awesome_clone_corl_2025.md) |
| 13 | Behavior Foundation Model for Humanoid Robots | [paper-behavior-foundation-model-humanoid](../entities/paper-behavior-foundation-model-humanoid.md) | [source](../../sources/papers/bfm_awesome_bfm_humanoid_arxiv_2509_13780.md) |
| 14 | HOVER | [paper-bfm-14-hover](../entities/paper-bfm-14-hover.md) | [source](../../sources/papers/bfm_awesome_hover_arxiv_2410_21229.md) |
| 15 | InterMimic | [paper-bfm-15-intermimic](../entities/paper-bfm-15-intermimic.md) | [source](../../sources/papers/bfm_awesome_intermimic_arxiv_2502_20390.md) |
| 16 | ModSkill | [paper-bfm-16-modskill](../entities/paper-bfm-16-modskill.md) | [source](../../sources/papers/bfm_awesome_modskill_arxiv_2502_14140.md) |
| 17 | MaskedMimic | [paper-bfm-17-maskedmimic](../entities/paper-bfm-17-maskedmimic.md) | [source](../../sources/papers/bfm_awesome_maskedmimic_tog_2024.md) |
| 18 | H-GAP | [paper-bfm-18-hgap](../entities/paper-bfm-18-hgap.md) | [source](../../sources/papers/bfm_awesome_hgap_arxiv_2312_02682.md) |
| 19 | CALM | [paper-bfm-19-calm](../entities/paper-bfm-19-calm.md) | [source](../../sources/papers/bfm_awesome_calm_siggraph_2024.md) |
| 20 | MoConVQ | [paper-bfm-20-moconvq](../entities/paper-bfm-20-moconvq.md) | [source](../../sources/papers/bfm_awesome_moconvq_tog_2023.md) |
| 21 | CASE | [paper-bfm-21-case](../entities/paper-bfm-21-case.md) | [source](../../sources/papers/bfm_awesome_case_arxiv_2309_11351.md) |
| 22 | PHC | [paper-bfm-22-phc](../entities/paper-bfm-22-phc.md) | [source](../../sources/papers/bfm_awesome_phc_arxiv_2305_06456.md) |
| 23 | TeamPlay | [paper-bfm-23-teamplay](../entities/paper-bfm-23-teamplay.md) | [source](../../sources/papers/bfm_awesome_teamplay_arxiv_2105_12196.md) |
| 24 | MTM | [paper-bfm-24-mtm](../entities/paper-bfm-24-mtm.md) | [source](../../sources/papers/bfm_awesome_mtm_arxiv_2305_02968.md) |
| 25 | ASE | [paper-bfm-25-ase](../entities/paper-bfm-25-ase.md) | [source](../../sources/papers/bfm_awesome_ase_arxiv_2205_01906.md) |
| 26 | Active Pretraining with Successor Features | [paper-bfm-26-aps](../entities/paper-bfm-26-aps.md) | [source](../../sources/papers/bfm_awesome_aps_icml_2021.md) |
| 27 | Reinforcement Learning with Prototypical Representations | [paper-bfm-27-proto-rl](../entities/paper-bfm-27-proto-rl.md) | [source](../../sources/papers/bfm_awesome_proto_rl_icml_2021.md) |
| 28 | State Entropy Maximization with Random Encoders for Efficient Exploration | [paper-bfm-28-re3](../entities/paper-bfm-28-re3.md) | [source](../../sources/papers/bfm_awesome_re3_icml_2020.md) |
| 29 | Exploration by Random Network Distillation | [paper-bfm-29-rnd](../entities/paper-bfm-29-rnd.md) | [source](../../sources/papers/bfm_awesome_rnd_iclr_2019.md) |
| 30 | Diversity is All You Need | [paper-bfm-30-diayn](../entities/paper-bfm-30-diayn.md) | [source](../../sources/papers/bfm_awesome_diayn_iclr_2018.md) |
| 31 | Task Tokens | [paper-bfm-31-task-tokens](../entities/paper-bfm-31-task-tokens.md) | [source](../../sources/papers/bfm_awesome_task_tokens_arxiv_2503_22886.md) |
| 32 | Zero-Shot Adaptation of Behavioral Foundation Models to Unseen Dynamics | [paper-bfm-32-unseen-dynamics](../entities/paper-bfm-32-unseen-dynamics.md) | [source](../../sources/papers/bfm_awesome_unseen_dynamics_arxiv_2505_13150.md) |
| 33 | Fast Adaptation With Behavioral Foundation Models | [paper-bfm-33-fast-adaptation-bfm](../entities/paper-bfm-33-fast-adaptation-bfm.md) | [source](../../sources/papers/bfm_awesome_fast_adaptation_bfm_corl_2025.md) |
| 34 | SENTINEL | [paper-bfm-34-sentinel](../entities/paper-bfm-34-sentinel.md) | [source](../../sources/papers/bfm_awesome_sentinel_arxiv_2511_19236.md) |
| 35 | BeyondMimic | [paper-bfm-35-beyondmimic](../entities/paper-bfm-35-beyondmimic.md) | [source](../../sources/papers/bfm_awesome_beyondmimic_arxiv_2508_08241.md) |
| 36 | LeVerb | [paper-bfm-36-leverb](../entities/paper-bfm-36-leverb.md) | [source](../../sources/papers/bfm_awesome_leverb_arxiv_2506_13751.md) |
| 37 | LangWBC | [paper-bfm-37-langwbc](../entities/paper-bfm-37-langwbc.md) | [source](../../sources/papers/bfm_awesome_langwbc_arxiv_2504_21738.md) |
| 38 | Tokenhsi | [paper-bfm-38-tokenhsi](../entities/paper-bfm-38-tokenhsi.md) | [source](../../sources/papers/bfm_awesome_tokenhsi_arxiv_2503_19901.md) |
| 39 | CloSD | [paper-bfm-39-closd](../entities/paper-bfm-39-closd.md) | [source](../../sources/papers/bfm_awesome_closd_arxiv_2410_03441.md) |
| 40 | UniPhys | [paper-bfm-40-uniphys](../entities/paper-bfm-40-uniphys.md) | [source](../../sources/papers/bfm_awesome_uniphys_arxiv_2504_12540.md) |
| 41 | Unified Human-Scene Interaction via Prompted Chain-of-Contacts | [paper-bfm-41-unihsi](../entities/paper-bfm-41-unihsi.md) | [source](../../sources/papers/bfm_awesome_unihsi_arxiv_2309_07918.md) |

### 数据集（10）

| 数据集 | Wiki 实体 | Source |
|--------|-----------|--------|
| Humanoid-X | [dataset-bfm-humanoid-x](../entities/dataset-bfm-humanoid-x.md) | [source](../../sources/papers/bfm_awesome_dataset_humanoid_x_arxiv_2501_05098.md) |
| PHUMA | [dataset-bfm-phuma](../entities/dataset-bfm-phuma.md) | [source](../../sources/papers/bfm_awesome_dataset_phuma_arxiv_2510_26236.md) |
| Motion-X++ | [dataset-bfm-motion-xpp](../entities/dataset-bfm-motion-xpp.md) | [source](../../sources/papers/bfm_awesome_dataset_motion_xpp_arxiv_2501_05098.md) |
| Motion-X | [dataset-bfm-motion-x](../entities/dataset-bfm-motion-x.md) | [source](../../sources/papers/bfm_awesome_dataset_motion_x_neurips_2023.md) |
| PoseScript | [dataset-bfm-posescript](../entities/dataset-bfm-posescript.md) | [source](../../sources/papers/bfm_awesome_dataset_posescript_eccv_2022.md) |
| HumanML3D | [dataset-bfm-humanml3d](../entities/dataset-bfm-humanml3d.md) | [source](../../sources/papers/bfm_awesome_dataset_humanml3d_cvpr_2022.md) |
| BABEL | [dataset-bfm-babel](../entities/dataset-bfm-babel.md) | [source](../../sources/papers/bfm_awesome_dataset_babel_cvpr_2021.md) |
| LAFAN | [dataset-bfm-lafan](../entities/dataset-bfm-lafan.md) | [source](../../sources/papers/bfm_awesome_dataset_lafan_tog_2020.md) |
| AMASS | [amass](../entities/amass.md) | [source](../../sources/papers/bfm_awesome_dataset_amass_iccv_2019.md) |
| KIT-ML | [dataset-bfm-kit-ml](../entities/dataset-bfm-kit-ml.md) | [source](../../sources/papers/bfm_awesome_dataset_kit_ml_arxiv_2016.md) |

## 五组论文地图（41 篇）

> **Source** 列指向独立原始资料页；**本库** 列标已有 wiki 深读页，其余为后续升格候选。

### 01 — Forward-backward 表征（6 篇）

| # | 工作 | 要点 | Source | 本库 |
|---|------|------|--------|------|
| 01 | **BFM-Zero** | 无监督 RL + latent prompt；未见动作跟踪、奖励优化、恢复 | [01](../../sources/papers/bfm_awesome_bfm_zero_arxiv_2511_04131.md) | [BFM 实体 § 同期工作](../entities/paper-behavior-foundation-model-humanoid.md) |
| 02 | **MetaMotivo** | Zero-shot whole-body；与 BFM-Zero 对照读 | [02](../../sources/papers/bfm_awesome_metamotivo_arxiv_2504_11054.md) | — |
| 03 | **FB-AW / FB-AWARE** | 潜空间要「细」才可被上层精确调用 | [03](../../sources/papers/bfm_awesome_fb_aw_arxiv_2412_04368.md) | — |
| 04 | **Fast Imitation** | 有基座后新动作应少走弯路 | [04](../../sources/papers/bfm_awesome_fast_imitation_bfm_neurips_2024.md) | — |
| 05 | **Learning One Representation** | 统一 FB 嵌入服务多 reward | [05](../../sources/papers/bfm_awesome_learning_one_representation_neurips_2021.md) | — |
| 06 | **Successor States** | 未来状态分布的数学底座 | [06](../../sources/papers/bfm_awesome_successor_states_arxiv_2101_07123.md) | — |

### 02 — Goal-conditioned 学习（19 篇）

| # | 工作 | 要点 | 本库 |
|---|------|------|------|
| 07 | **SONIC** | Supersizing motion tracking；动作覆盖面 | [sonic-motion-tracking](../methods/sonic-motion-tracking.md) |
| 08 | **OpenTrack** | 任意动作 + 扰动下跟踪 | — |
| 09 | **AMS** | 异构数据下敏捷与稳定 | [ams](../methods/ams.md) |
| 10–12 | **TWIST2 / TWIST / CLONE** | 全身数据采集与长时程遥操作闭环 | [teleoperation](../tasks/teleoperation.md) |
| 13 | **BFM for Humanoid Robots** | CVAE + 掩码 + 多接口统一 | [paper-behavior-foundation-model-humanoid](../entities/paper-behavior-foundation-model-humanoid.md) |
| 14 | **HOVER** | 神经全身控制器 · 上层接口 | — |
| 15 | **InterMimic** | 人-物交互 WBC | — |
| 16–25 | **ModSkill … ASE** | 技能模块化、masked inpainting、planner、CALM/CASE、PHC、MTM、ASE 等 | [protomotions](../entities/protomotions.md)（MaskedMimic 生态） |

### 03 — Intrinsic reward 预训练（5 篇）

| # | 工作 | 要点 | 本库 |
|---|------|------|------|
| 26–30 | **APS / Proto-RL / RE3 / RND / DIAYN** | 任务到来前的探索与技能分化 | [behavior-foundation-model](../concepts/behavior-foundation-model.md) § intrinsic 线 |

### 04 — Adaptation（3 篇）

| # | 工作 | 要点 | 本库 |
|---|------|------|------|
| 31 | **Task Tokens** | 轻量 task 条件适配 | — |
| 32 | **Unseen Dynamics** | 负载/地面/参数变化 | [sim2real](../concepts/sim2real.md) |
| 33 | **Fast Adaptation** | 下游样本与工程成本 | — |

### 05 — Hierarchical control（8 篇）

| # | 工作 | 要点 | 本库 |
|---|------|------|------|
| 34 | **SENTINEL** | 端到端 language-action WBC | [身体系统栈 §7](./humanoid-rl-motion-control-body-system-stack.md) |
| 35 | **BeyondMimic** | Guided diffusion 全身控制 | [beyondmimic](../methods/beyondmimic.md) |
| 36–37 | **LeVerb / LangWBC** | 视觉-语言 → 全身 latent / 端到端 | [vla](../methods/vla.md)、[DAJI](../entities/paper-daji-anticipatory-joint-intent.md) |
| 38–41 | **TokenHSI / CLoSD / UniPhys / UniHSI** | 场景交互 token、仿真–扩散闭环、规划–控制、contact chain | — |

### 数据集（10 项，不计入 41 篇）

文内强调：**上限在数据能否变成机器人可信、可执行、可迁移的训练材料**。独立 source 见 [catalog § 数据集](../../sources/papers/bfm_awesome_41_catalog.md#数据集10)（Humanoid-X、PHUMA、Motion-X++、Motion-X、HumanML3D、BABEL、LAFAN、PoseScript、KIT-ML；AMASS 另见 [amass-dataset](../../sources/sites/amass-dataset.md)）。

## 与 taxonomy / 身体系统栈的对照

| 本页五组 | [BFM 概念页](../concepts/behavior-foundation-model.md) taxonomy | [八层身体系统栈](./humanoid-rl-motion-control-body-system-stack.md) |
|----------|---------------------------------------------------------------------|----------------------------------------------------------------------|
| 01 FB | Forward–backward 预训练 | 控制层 · 身体潜空间 / prompt |
| 02 Goal-conditioned | Goal-conditioned 预训练 | 数据 + 跟踪 + 控制（层 1–3） |
| 03 Intrinsic | Intrinsic-reward 预训练 | 控制层 · 探索先验 |
| 04 Adaptation | 微调线 | 跨任务 / 跨动力学部署 |
| 05 Hierarchical | 层次化 + 微调 | 任务接口层（层 7）· VLA 调用 |

## 四个后续深读方向（沿用文内计划）

1. **BFM-Zero + forward-backward** — 与 CVAE-BFM 的方法谱系两端。
2. **HOVER / MaskedMimic / SONIC** — 全身跟踪与多接口执行器。
3. **SENTINEL / LangWBC / LeVerb** — 语义到身体接口。
4. **Humanoid-X / Motion-X++** — 身体数据 scaling。

## 关联页面

- [Behavior Foundation Model](../concepts/behavior-foundation-model.md) — 定义与 taxonomy
- [BFM（人形 WBC 论文实体）](../entities/paper-behavior-foundation-model-humanoid.md)
- [人形运动跟踪方法选型](../queries/humanoid-motion-tracking-method-selection.md)
- [Foundation Policy](../concepts/foundation-policy.md) — 与 VLA 操作向基础策略的边界
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)、[AMP 运动先验综述](./humanoid-amp-motion-prior-survey.md)

## 参考来源

- [BFM 41 篇 + 10 数据集 source 总索引](../../sources/papers/bfm_awesome_41_catalog.md)
- [具身智能研究室 · BFM 41 篇微信公众号编译稿](../../sources/blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md)
- [awesome-bfm-papers 精选列表](../../sources/repos/awesome_bfm_papers.md)
- [BFM 综述（arXiv:2506.20487）](../../sources/papers/bfm_survey_arxiv_2506_20487.md)

## 推荐继续阅读

- [awesome-bfm-papers](https://github.com/friedrichyuan/awesome-bfm-papers) — 41 篇完整表与数据集表
- [A Survey of Behavior Foundation Model](https://arxiv.org/abs/2506.20487) — TPAMI 2025 全文
- [BFM-Zero 项目页](https://lecar-lab.github.io/BFM-Zero/) — promptable 身体基座对照
