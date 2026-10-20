# galilai-group/cjepa

> 来源归档（ingest）

- **仓库：** <https://github.com/galilai-group/cjepa>
- **对应论文：** [Causal-JEPA（arXiv:2602.11389）](../papers/causal_jepa_arxiv_2602_11389.md)
- **许可证：** README 未声明
- **依赖（README）：** Stable-WorldModel、Stable-Pretraining；外部 VideoSAUR、SAVi、ALOE
- **入口（README）：** 训练 `script/clevrer/train_cjepa_from_slot.sh`、`script/pusht/train_cjepa_from_slot.sh`；评测 `scripts/pusht/test_planning.sh`、`scripts/clevrer/rollout_from_slot.sh`；视觉问答 `scripts/clevrer/{train,test}_aloe.sh`
- **权重：** README 链接 Hugging Face 上的对象编码器、预提取 slot 与 C-JEPA 模型
- **入库日期：** 2026-10-20

## 对 wiki 的映射

- [paper-causal-jepa](../../wiki/entities/paper-causal-jepa.md) 的「源码运行时序图」与「工程实践」
