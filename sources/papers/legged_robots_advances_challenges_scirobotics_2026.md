# Advances, challenges, and opportunities for legged robots（Science Robotics, 2026）

> 来源归档（ingest）

- **标题：** Advances, challenges, and opportunities for legged robots
- **类型：** paper / Review / legged robots / humanoid / quadruped / ethics & policy
- **期刊：** Science Robotics, 2026（Vol. 11, Issue 116；article eaee0787）
- **DOI：** <https://doi.org/10.1126/scirobotics.aee0787>
- **PubMed：** <https://pubmed.ncbi.nlm.nih.gov/42525724/>（PMID:42525724）
- **OpenAlex：** <https://openalex.org/W7171713488>
- **项目页 / GitHub：** **无**（领域综述；截至 **2026-07-31** 未见独立项目页、代码仓库或数据集发布）
- **作者：** Jonas Frey、Matías Mattamala、Hae-Won Park、Mayank Mittal、Georg Martius、Maike Osborne、Robert Sparrow、Marco Hutter
- **机构：** ETH Zurich；Stanford；UC Berkeley；University of Edinburgh；KAIST；NVIDIA；University of Tübingen；Max Planck Institute for Intelligent Systems；University of Oxford；Monash University；RAI Institute
- **代码与数据：** **未开源 / 不适用**（综述，无单一可运行实现）
- **入库日期：** 2026-07-31
- **最后更新：** 2026-07-31
- **一句话说明：** Science Robotics **Review**：沿 **硬件 / locomotion / 自主 / 数据 / 应用** 五柱评估人形与四足腿式系统的能力与开放挑战，并展望伦理、经济、政策与社会影响。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.1126/scirobotics.aee0787](https://doi.org/10.1126/scirobotics.aee0787) | Science Robotics 原文入口（付费墙） |
| PubMed | [42525724](https://pubmed.ncbi.nlm.nih.gov/42525724/) | 摘要全文（开放） |
| Crossref | [works API](https://api.crossref.org/works/10.1126/scirobotics.aee0787) | 元数据与 JATS 摘要 |
| OpenAlex | [W7171713488](https://openalex.org/W7171713488) | 引用图（约 113 篇参考文献） |
| 机构通稿解读 | [`techxplore_legged_robots_ethics_monash_2026-07-30.md`](../blogs/techxplore_legged_robots_ethics_monash_2026-07-30.md) | Monash / TechXplore：伦理–经济–政策侧复述 |
| 同期综述对照 | [`bioinspired_multimodal_robotics_scirobotics_2026.md`](./bioinspired_multimodal_robotics_scirobotics_2026.md) | 同刊 Issue 116 仿生多模态 Review |
| 运动任务 | [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md) | 腿式 locomotion 任务中心 |
| 四足实体 | [`wiki/entities/quadruped-robot.md`](../../wiki/entities/quadruped-robot.md) | 四足平台总览 |

## 摘要级要点

- **对象：** 人形（humanoid）与四足（quadrupedal）腿式机器人，而非跨介质多模态族。
- **评估轴（五柱）：** hardware · locomotion · autonomy · data · applications。
- **目标：** 识别近期进展与阻碍大规模部署 / 新用例的关键开放挑战。
- **展望：** 伦理考量、经济潜力、政策含义与更广社会影响（非纯技术路线图）。
- **开源边界：** 综述无官方代码 / 数据集 / 项目页；复现应落到被引用的具体系统论文。

## 核心摘录（面向 wiki 编译）

### 1) 五柱评估框架（摘要主结构）

- 作者明确以 **硬件、运动、自主、数据、应用** 五轴盘点当前能力，而不是只写控制算法。
- 这与本库「硬件 101 / locomotion / 感知导航 / 数据与仿真 / 落地任务」分层可读作同一坐标系。
- **对 wiki 的映射：** 主沉淀页以五柱组织「核心原理」；交叉到 [locomotion](../../wiki/tasks/locomotion.md)、[quadruped-robot](../../wiki/entities/quadruped-robot.md)、[sim2real](../../wiki/concepts/sim2real.md)。

### 2) 技术侧线索（由参考文献图 + 作者阵容推断，非全文逐节复述）

OpenAlex 引用约 **113** 篇，重心包括：

| 柱 | 代表性被引线索（节选） | 工程读法 |
|----|------------------------|----------|
| Hardware | ANYmal、MIT Cheetah / SEA / 本体感受执行器、ARTEMIS、软体/电液腿 | 执行器带宽与冲击耐受仍是上限 |
| Locomotion | Hwangbo 2019、Lee 2020 challenging terrain、RMA、ANYmal parkour、DTC、实机人形 RL loco、Ha et al. IJRR 2025 学习型腿式综述 | Sim2Real RL + 感知 loco 已是主叙事 |
| Autonomy | CERBERUS / SubT、森林清查、AutoInspect、可通行性估计、Holistic Fusion | 长程野外自主 ≠ 室内 demo |
| Data | Ego4D、SubT-MRS、NeRF / 3DGS、物理仿真器角色 | 数据与世界模型成为瓶颈叙事 |
| Applications | 工业巡检、农业、行星模拟、建筑、军事与养老照料相关伦理文献 | 用例分化决定安全/伦理优先级 |

- **对 wiki 的映射：** 勿把推断写成「原文已给出的章节标题」；在 wiki 中标注「摘要五柱 + 引用图/通稿补充」。

### 3) 伦理–经济–政策展望（Monash / TechXplore 通稿）

机构通稿（共同作者 Robert Sparrow）强调：

- **养老与陪伴：** 机器人照料能否替代人际连接；故障后的情感依赖风险。
- **监视与数据：** 家中带摄像头的腿式平台 → 谁控制亲密数据。
- **外形与偏见：** 人形常被设计为男/女气质；待遇机器人可能外溢到对人或动物的态度。
- **军事：** 可能降低杀戮心理门槛与冲突阈值；战场问责与人权框架滞后。
- **就业：** 此前自动化主要冲击制造；腿式系统可能冲击服务业（部分发达经济体约 **80%** 就业）。
- **民主授权：** 「机器人最重要的不是腿，而是规范其发展的政策与民主授权。」
- **对 wiki 的映射：** 「局限与风险 / 结论」写入社会层，避免只剩技术 checklist。

### 4) 开源与复现边界

- **综述文章**，无单一官方代码 / 数据集 / 项目页（截至 2026-07-31）。
- Science.org PDF 在本环境 **403**；细节以开放摘要 + 机构通稿 + 引用图为准。
- **对 wiki 的映射：** 「源码运行时序图」**不适用**。

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-legged-robots-advances-challenges.md`](../../wiki/entities/paper-legged-robots-advances-challenges.md)**
- 通稿源：**[`sources/blogs/techxplore_legged_robots_ethics_monash_2026-07-30.md`](../blogs/techxplore_legged_robots_ethics_monash_2026-07-30.md)**
- 交叉：**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**、**[`wiki/entities/quadruped-robot.md`](../../wiki/entities/quadruped-robot.md)**、**[`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)**、**[`wiki/entities/paper-bioinspired-multimodal-robotics.md`](../../wiki/entities/paper-bioinspired-multimodal-robotics.md)**

## 当前提炼状态

- [x] 论文摘要填写（Crossref / PubMed / OpenAlex）
- [x] 机构通稿抓取（TechXplore / SMBtech，伦理–政策侧）
- [x] 引用图抽样（OpenAlex referenced_works）辅助五柱技术线索
- [x] wiki 页面映射确认
- [x] 项目页 / 源码开放核查：无项目页；综述无代码（写入局限）
- [ ] 全文付费墙解除后可补各节精确主张与图表数值
