# Bioinspired multimodal robotics（Science Robotics, 2026）

> 来源归档（ingest）

- **标题：** Bioinspired multimodal robotics
- **类型：** paper / Review / bioinspired / multimodal locomotion / soft robotics / physical intelligence
- **期刊：** Science Robotics, 2026（Vol. 11, Issue 116）
- **DOI：** <https://doi.org/10.1126/scirobotics.aea7639>
- **PubMed：** <https://pubmed.ncbi.nlm.nih.gov/42485442/>（PMID:42485442）
- **eprint（作者分享链）：** <https://www.science.org/eprint/DSWZCRX276ZXEYUWFPQN/full?activationRedirect=/doi/full/10.1126/scirobotics.aea7639>（入库时环境返回 403，正文以 Crossref/PubMed 摘要 + 公开二手报道核对）
- **项目页 / GitHub：** **无**（综述文章；截至 **2026-07-25** 未见独立项目页、代码仓库或数据集发布）
- **作者：** Ziyu Ren、Youning Duo、Haoyuan Xu、Yihui Zhang、Xingjian Liu、Jamie Paik、Auke Ijspeert、Li Wen（文力）
- **机构：** 北京航空航天大学（Beihang / BUAA）；清华大学；大连理工大学；洛桑联邦理工学院（EPFL，含 Reconfigurable Robotics Lab / Biorobotics Lab）
- **代码与数据：** **未开源 / 不适用**（综述，无单一可运行实现）
- **入库日期：** 2026-07-25
- **一句话说明：** Science Robotics **Review**：界定仿生多模态机器人（≥2 种仿生运动模态且可切换），梳理机体设计（软材料 / 结构复用 / 多机协作）与控制范式迁移（图搜索·分立控制器 → 学习框架），并提出 **五项量化评测指标**，主张 **物理智能 + 计算智能** 融合路线图。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.1126/scirobotics.aea7639](https://doi.org/10.1126/scirobotics.aea7639) | Science Robotics 原文入口 |
| PubMed | [42485442](https://pubmed.ncbi.nlm.nih.gov/42485442/) | 摘要全文（开放） |
| Crossref | [works API](https://api.crossref.org/works/10.1126/scirobotics.aea7639) | 元数据与 JATS 摘要 |
| 二手报道 | [Interesting Engineering 综述解读](https://interestingengineering.com/ai-robotics/bioinspired-robotics-the-future-of-robots-that-can-walk-fly-swim-and-climb) | 五项指标与挑战的科普复述（非原文） |
| 北航文力组实例 | [`aerial_aquatic_remora_scirobotics_2022.md`](./aerial_aquatic_remora_scirobotics_2022.md) | 空–水多模态 + 结构复用（被动桨） |
| 北航文力组实例 | [`miniature_deep_sea_morphable_scirobotics_2025.md`](./miniature_deep_sea_morphable_scirobotics_2025.md) | 深海软体三模态 |
| 运动任务 | [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md) | 多模态 locomotion 任务中心 |
| 混合运动任务 | [`wiki/tasks/hybrid-locomotion.md`](../../wiki/tasks/hybrid-locomotion.md) | 轮腿 / 可变形态混合运动 |

## 摘要级要点

- **定义：** 仿生多模态机器人 = 在同一系统上整合并切换 **两种及以上** 仿生运动模态（走 / 飞 / 游 / 爬 / 跳等）的机器人。
- **目标不是「模态堆叠」：** 主目标是在非结构化、动态环境中提升 **整体任务效能**（例如远程飞行 + 地面精检、两栖搜救）。
- **机体设计三线索：**
  1. **软材料 / 柔性结构** — 形变适配环境；
  2. **结构复用（structure repurposing）** — 同一组件跨模态承担不同功能，降低「死重」；
  3. **多机器人系统** — 用协作替代单体堆叠全部模态。
- **模态切换：** 主动与被动结构重配置均可支持无缝切换；被动智能（环境力学触发形变）与主动变形互补。
- **规划与控制：** 从传统图搜索路径规划 + 分立模态控制器，转向 **学习框架**（RL、VLA、世界模型等）；传统方法难以处理模态切换时的动力学剧变。
- **五项评测指标（填补标准化空白）：**
  1. **Number of modes** — 模态数量；
  2. **Marginal cost of modality** — 新增模态的边际成本；
  3. **Component repurpose percentage** — 跨模态组件复用比例；
  4. **Transition cost** — 模态切换的时间 / 能量代价；
  5. **Performance improvement** — 多模态相对单模态的综合性能增益。
- **工程瓶颈：** 机载空间/重量预算、跨模态「死重」、刚度–柔度权衡、异构驱动集成、模态互相干扰。
- **未来路线图：** **物理智能（adaptive hardware）** 与 **计算智能（感知 / 规划 / 控制学习）** 深度融合，以支撑复杂环境刺激下的实时行为适配。

## 核心摘录（面向 wiki 编译）

### 1) 领域定义与历史线索

- 动物多模态运动是非结构化环境生存的核心能力；仿生多模态机器人以此为驱动力。
- 演进方向：从「拼装式多机构叠加」→「共享结构 + 智能控制的一体化系统」。
- **对 wiki 的映射：** 与轮腿 / 可变形态 [hybrid-locomotion](../../wiki/tasks/hybrid-locomotion.md) 对照——本综述覆盖更广的跨介质仿生模态族，而非仅轮–腿。

### 2) 五项定量指标（设计有效性 + 运行性能）

| 指标（英文） | 设计意图（摘要编译） |
|--------------|----------------------|
| Number of modes | 能力覆盖面；但单独追求数量无意义 |
| Marginal cost of modality | 每新增一模态付出的质量 / 体积 / 功耗增量 |
| Component repurpose percentage | 共享结构程度；越高则「死重」越低 |
| Transition cost | 切换时间或能量；决定任务级可用性 |
| Performance improvement | 相对单模态基线的任务级增益 |

- **对 wiki 的映射：** 主沉淀页用该表作评测框架；工程选型时优先看 **marginal cost + transition cost + repurpose %**，而非模态数。

### 3) 机体与控制的耦合观点

- 软材料、结构复用、多机协作共同降低「每模态一套硬件」的代价。
- 控制侧：离散模态 FSM / 图规划在动力学突变处脆弱；学习框架更适合连续切换与环境适应。
- 路线图明确主张：**physical intelligence × computational intelligence**，而非只堆算法或只堆机构。
- **对 wiki 的映射：** 与 [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[VLA](../../wiki/methods/vla.md) 交叉；与北航文力组实例页（印鱼两栖、深海可变形、章鱼臂）互链作「结构复用 / 被动智能」案例。

### 4) 开源与复现边界

- **综述文章**，无单一官方代码 / 数据集 / 项目页。
- 入库日（2026-07-25）eprint 链接在本环境 **403**；正文细节以开放摘要与元数据为准，二手报道仅作指标命名交叉核对。
- **对 wiki 的映射：** 实体页「源码运行时序图」写明 **不适用**；「局限」注明全文付费墙与无代码。

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-bioinspired-multimodal-robotics.md`](../../wiki/entities/paper-bioinspired-multimodal-robotics.md)**
- 交叉：**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**、**[`wiki/tasks/hybrid-locomotion.md`](../../wiki/tasks/hybrid-locomotion.md)**
- 北航文力组系列实例：**[`wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md`](../../wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md)**、**[`wiki/entities/paper-miniature-deep-sea-morphable-robot.md`](../../wiki/entities/paper-miniature-deep-sea-morphable-robot.md)**、**[`wiki/entities/paper-octopus-inspired-esoam-soft-arm.md`](../../wiki/entities/paper-octopus-inspired-esoam-soft-arm.md)**
- 仿生多步态对照：**[`wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md`](../../wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md)**

## 当前提炼状态

- [x] 论文摘要填写（Crossref / PubMed）
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源 / 关联段落已添加 ingest 链接
- [x] 项目页 / 源码开放核查：无项目页；综述无代码（写入局限）
