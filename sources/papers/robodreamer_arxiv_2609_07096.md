# RoboDreamer（arXiv:2609.07096）

> 来源归档（ingest）

- **标题：** RoboDreamer：基于预测状态空间模型的前瞻式人形运动控制
- **英文标题：** RoboDreamer: Anticipatory Humanoid Locomotion with Predictive State-Space Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.07096>
- **PDF：** <https://arxiv.org/pdf/2609.07096>
- **开源：** 截至入库日 **未见** 官方仓库（步骤 2.5：项目页/arXiv 未给出可运行代码链接）。
- **入库日期：** 2026-09-14
- **策展索引：** [wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md](../blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)

## 核心论文摘录

### 1) 两阶段 Teacher-Student

- 先训预测状态空间模型，再蒸馏可部署策略。
- **对 wiki 的映射：** [../../wiki/entities/paper-robodreamer-anticipatory-humanoid-locomotion.md](../../wiki/entities/paper-robodreamer-anticipatory-humanoid-locomotion.md)

### 2) 随机掩码 + 下一观测一致性

- 强迫模型补全缺失感知，提升前瞻质量。
- **对 wiki 的映射：** [../../wiki/entities/paper-robodreamer-anticipatory-humanoid-locomotion.md](../../wiki/entities/paper-robodreamer-anticipatory-humanoid-locomotion.md)

### 3) Mamba 长序列 + 推理动作细化

- 长时依赖与在线 action refinement。
- **对 wiki 的映射：** [../../wiki/entities/paper-robodreamer-anticipatory-humanoid-locomotion.md](../../wiki/entities/paper-robodreamer-anticipatory-humanoid-locomotion.md)

## 步骤 2.5 开源核查

- 已检索 arXiv 摘要与常见项目页关键词（GitHub/code）；**未发现**可运行官方实现。
- 若后续发布代码，应同步 `sources/repos/` 与本 wiki 页「工程实践」与「源码运行时序图」。

## 当前提炼状态

- [x] 公众号周更 ingest 映射
- [x] wiki 实体页
