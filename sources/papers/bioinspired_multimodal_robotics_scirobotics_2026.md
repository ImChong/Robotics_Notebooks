# Bioinspired multimodal robotics（Science Robotics, 2026）

> 来源归档（ingest）

- **标题：** Bioinspired multimodal robotics
- **类型：** paper / Review / bioinspired / multimodal locomotion / soft robotics / physical intelligence
- **期刊：** Science Robotics, 2026（Vol. 11, Issue 116）
- **DOI：** <https://doi.org/10.1126/scirobotics.aea7639>
- **PubMed：** <https://pubmed.ncbi.nlm.nih.gov/42485442/>（PMID:42485442）
- **eprint（作者分享链）：** <https://www.science.org/eprint/DSWZCRX276ZXEYUWFPQN/full?activationRedirect=/doi/full/10.1126/scirobotics.aea7639>（入库时环境返回 403；正文以 Crossref/PubMed 摘要 + 微信导读 + 公开二手报道核对）
- **项目页 / GitHub：** **无**（综述文章；截至 **2026-07-25** 未见独立项目页、代码仓库或数据集发布）
- **作者：** Ziyu Ren†、Youning Duo†、Haoyuan Xu†、Yihui Zhang、Xingjian Liu、Jamie Paik、Auke Ijspeert、Li Wen（文力）（† 共同一作，据微信导读）
- **机构：** 北京航空航天大学（Beihang / BUAA）；清华大学；大连理工大学；洛桑联邦理工学院（EPFL，含 Reconfigurable Robotics Lab / Biorobotics Lab）
- **代码与数据：** **未开源 / 不适用**（综述，无单一可运行实现）
- **入库日期：** 2026-07-25
- **最后更新：** 2026-07-25（补微信导读：指标数值例、切换分类、三模块架构）
- **一句话说明：** Science Robotics **Review**：界定仿生多模态机器人（≥2 种运动模态或移动+操作、至少一种仿生、可切换），梳理机体设计（软材料 / 结构复用 / 多机协作）与控制范式迁移，并提出 **五项量化评测指标**，主张 **物理智能 + 计算智能** 融合路线图。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.1126/scirobotics.aea7639](https://doi.org/10.1126/scirobotics.aea7639) | Science Robotics 原文入口 |
| PubMed | [42485442](https://pubmed.ncbi.nlm.nih.gov/42485442/) | 摘要全文（开放） |
| Crossref | [works API](https://api.crossref.org/works/10.1126/scirobotics.aea7639) | 元数据与 JATS 摘要 |
| 微信导读 | [`wechat_robot_lecture_bioinspired_multimodal_2026-07-25.md`](../blogs/wechat_robot_lecture_bioinspired_multimodal_2026-07-25.md) | 机器人大讲堂深度复述（指标公式/样机数值/切换分类/三模块架构） |
| 二手报道 | [Interesting Engineering 综述解读](https://interestingengineering.com/ai-robotics/bioinspired-robotics-the-future-of-robots-that-can-walk-fly-swim-and-climb) | 五项指标与挑战的科普复述（非原文） |
| 北航文力组实例 | [`aerial_aquatic_remora_scirobotics_2022.md`](./aerial_aquatic_remora_scirobotics_2022.md) | 空–水多模态 + 结构复用（被动桨）；导读给出 Tair-water/Twater-air |
| 北航文力组实例 | [`miniature_deep_sea_morphable_scirobotics_2025.md`](./miniature_deep_sea_morphable_scirobotics_2025.md) | 深海软体三模态；导读 CRP=0.4 例 |
| 运动任务 | [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md) | 多模态 locomotion 任务中心 |
| 混合运动任务 | [`wiki/tasks/hybrid-locomotion.md`](../../wiki/tasks/hybrid-locomotion.md) | 轮腿 / 可变形态混合运动 |

## 摘要级要点

- **定义（摘要 + 微信导读对齐）：** 同一平台集成 **≥2 种运动模态**，**或** 同时具备移动与操作；**至少一种模态受生物启发**，并能在模态间切换。
- **目标不是「模态堆叠」：** 主目标是在非结构化、动态环境中提升 **整体任务效能**（例如远程飞行 + 地面精检、两栖搜救）。
- **三阶段历史：** 工程化堆叠（1959–世纪末）→ 仿生嵌入本体（PolyBot / AZIMUT）→ 计算智能 + 物理智能融合（CPG / 分层 RL / 软体磁控等）。
- **六大设计权衡：** 空间分配、质量死重、模态协同、刚度矛盾、驱动冲突、变形能力。
- **机体设计三线索：** 软材料 / 结构复用 / 多机器人集群涌现。
- **模态切换：** 无结构变换 vs 结构变换（0M–1M / 1M–1M / 1M–MM / MM–MM）；主动 vs 被动；现有工作偏主动 + 1M–1M/1M–MM。
- **规划与控制：** 图搜索 + 分立控制器 → 学习框架；综述提出 **全局规划 / 执行 / 多模态感知** 三模块一体化架构。
- **五项评测指标：** Nmode、MCM、CRP、Tij、PI（设计效率 + 运动性能两维）。
- **未来路线图：** **物理智能（adaptive hardware）** 与 **计算智能（感知 / 规划 / 控制学习）** 深度融合。

## 核心摘录（面向 wiki 编译）

### 1) 领域定义与历史线索

- 动物多模态运动是非结构化环境生存的核心能力；仿生多模态机器人以此为驱动力。
- 演进：工程化堆叠 → 仿生形态自适应 → 计算智能 × 物理智能耦合。
- **对 wiki 的映射：** 与 [hybrid-locomotion](../../wiki/tasks/hybrid-locomotion.md) 对照——本综述覆盖更广跨介质仿生模态族。

### 2) 五项定量指标（含微信导读样机数值）

| 指标 | 意图 | 导读样机例 |
|------|------|------------|
| Nmode | 功能多样性基准 | Tribot = 5 |
| MCM | 新增模态是否划算 | ANYmal 加轮 ≈ 0.38 |
| CRP | 共享部件占比 | 深海变构形 0.4；片状磁控软体 1 |
| Tij | 切换时间/能量矩阵 | 空→水 0.13 s；水→空 0.35 s（空–水吸附机器人） |
| PI | 多模态相对单模态最佳的提升比 | Hopcopter 续航 PI=3.29 |

- **对 wiki 的映射：** 主沉淀页作评测框架；选型优先 **MCM + Tij + CRP + PI**，而非只比 Nmode。

### 3) 切换分类与三模块架构

- 切换：无结构变换 / 结构变换（nM–mM）；主动 / 被动；空白区：**0M–1M、MM–MM**。
- 架构：全局规划（大脑）→ 执行（分层 or 端到端小脑）→ 多模态感知闭环；展望 VLA / 世界模型全局端到端。
- **对 wiki 的映射：** 与 [reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[VLA](../../wiki/methods/vla.md) 交叉。

### 4) 开源与复现边界

- **综述文章**，无单一官方代码 / 数据集 / 项目页。
- eprint 在本环境 **403**；细节以开放摘要 + [微信导读](../blogs/wechat_robot_lecture_bioinspired_multimodal_2026-07-25.md) 为准。
- **对 wiki 的映射：** 「源码运行时序图」**不适用**。

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-bioinspired-multimodal-robotics.md`](../../wiki/entities/paper-bioinspired-multimodal-robotics.md)**
- 导读源：**[`sources/blogs/wechat_robot_lecture_bioinspired_multimodal_2026-07-25.md`](../blogs/wechat_robot_lecture_bioinspired_multimodal_2026-07-25.md)**
- 交叉：**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**、**[`wiki/tasks/hybrid-locomotion.md`](../../wiki/tasks/hybrid-locomotion.md)**
- 北航文力组系列实例：**[`wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md`](../../wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md)**、**[`wiki/entities/paper-miniature-deep-sea-morphable-robot.md`](../../wiki/entities/paper-miniature-deep-sea-morphable-robot.md)**、**[`wiki/entities/paper-octopus-inspired-esoam-soft-arm.md`](../../wiki/entities/paper-octopus-inspired-esoam-soft-arm.md)**
- 仿生多步态对照：**[`wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md`](../../wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md)**

## 当前提炼状态

- [x] 论文摘要填写（Crossref / PubMed）
- [x] 微信导读抓取与指标/架构补全（2026-07-25）
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源 / 关联段落已添加 ingest 链接
- [x] 项目页 / 源码开放核查：无项目页；综述无代码（写入局限）
