# Tactile-Guided Robot Learning: From Representation Learning to Foundation Models（综述，arXiv:2610.20134）

> 来源归档（ingest）

- **标题：** Tactile-Guided Robot Learning: From Representation Learning to Foundation Models
- **类型：** paper / survey / tactile sensing / robot learning / foundation models
- **arXiv：** <https://arxiv.org/abs/2610.20134>（v1，提交 2026-10-14，cs.RO）
- **项目页：** <https://sites.google.com/view/touch-in-robot-learning-review/home>（配套文献数据库：按条目链接论文、项目页、代码仓库等资源）
- **作者：** Kiyanoush Nazari, Luchen Li, Haoshu Fang, Nima Fazeli, Zhibin Li, Fumiya Iida, Thomas George Thuruthel
- **机构：** 未核实（页面未列机构）
- **入库日期：** 2026-10-15
- **一句话说明：** 按触觉在学习与控制流程中的功能组织文献，覆盖触觉表示与策略学习、触觉基础模型、预测与自适应控制，以及跨物体、任务、传感器、形态与 sim2real 的泛化；提出触觉基础模型与世界模型方向。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-15）：** 站点为「触觉机器人学习文献数据库」，导航到首页；正文未给 GitHub 或代码链接；具体条目内的外部链接在抽取中不可见，未逐条核实。
- **arXiv 页：** 无代码仓库。
- **结论：** **未开源**（综述，本身无可运行代码；配套文献库为资源索引）。

## 摘录 1：综述框架（摘要）

- **组织方式：** 按触觉的功能角色（表示学习、策略学习、预测与自适应控制、基础模型）分章，而非按传感器或任务。
- **泛化维度：** 物体、任务、传感器、形态（embodiment）、sim-to-real。
- **开放问题（作者列出）：** 学习可迁移的物理表示；把语义推理与接触级控制结合；跨异构数据与硬件扩展。
- **主张：** 触觉正从任务专用反馈信号转变为通用具身智能的核心模态。

## 摘录 2：自监督表征分类（Figure 2）

- **核对日期：** 2026-10-26
- **图注：** “Self-supervised representation learning paradigms for tactile sensing.”
- **摘录要点：** Figure 2 将范式分为三组：重建式（Autoencoder、VAE、Masked AE、隐式场）；对比式（同一传感器增强视图、跨传感器同一接触、同步触觉-视觉、冻结基础模型锚点）；非对比与预测式（自蒸馏、潜变量预测、动作条件未来预测）。图是方法分类示意，不是统一性能比较。
- **对 wiki 的映射：** [现有综述实体页](../../wiki/entities/paper-tactile-guided-robot-learning-survey.md)、[Sparsh](../../wiki/entities/paper-sparsh.md)。

## 配套文献库核查（2026-10-26）

- 官方配套站点介绍其为学习型触觉机器人操作的策展论文库，主题覆盖触觉表征、策略、基础模型、自适应控制与泛化。
- 条目可链接论文、项目页、代码仓库及其他资源；数据库不是本文的代码仓库。
- 归档见 [配套站点来源页](../sites/touch-in-robot-learning-review.md)。

## 建议 wiki 动作

- 升格 [`wiki/entities/paper-tactile-guided-robot-learning-survey.md`](../../wiki/entities/paper-tactile-guided-robot-learning-survey.md)：以「研究地图」读法组织，与已有触觉概念页和触觉综述页互链。
- 综述页无可运行代码，源码运行时序图标注为不适用。
