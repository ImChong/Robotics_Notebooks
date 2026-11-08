## [2026-11-08] ingest | DAWN 对比学习投影头与深度鲁棒性机制复核

- **意图：** 对现有 DAWN 项目条目补齐 Projection Head 的理论动机、正负样本构造及与随机潜状态的关系。
- **一手资料：** arXiv:2609.29092v1 全文、<https://dawn-parkour.github.io/>、<https://github.com/DocyNoah/dawn-parkour> 的世界模型和 runner 关键训练源码。
- **开源结论：** 官方已发布训练/仿真回放；未见独立预训练权重或真机通信部署入口。
- **关键更新：** 论文式 (9) 与实现的 BT×BT 双向 CE 负样本组织差异；对应方法流程图、源码训练时序图与消融数字。
- **知识页：** [DAWN](../wiki/entities/paper-dawn.md)。
