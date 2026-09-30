# REMAC（hatchetProject）

> 来源归档（repo）

- **标题：** REMAC
- **类型：** repo
- **链接：** <https://github.com/hatchetProject/REMAC>
- **关联论文：** [remac_arxiv_2601_20130.md](../papers/remac_arxiv_2601_20130.md)
- **项目页：** <https://remac-async.github.io/>
- **入库日期：** 2026-09-30
- **一句话说明：** ICLR 2026 REMAC 官方 Kinetix 实现：Stage1 按 RTC 管线训 base flow，Stage2 LoRA 微调 masked action chunking。

## 开源状态

- **已开源**：`uv sync` 环境；含 Kinetix 子模块。

## README 入口

1. Stage1 base：`src_lora/train_expert.py` → `generate_data.py` → flow 训练（对齐 [real-time-chunking-kinetix](real-time-chunking-kinetix.md) 思路）。
2. Stage2：`LoRA` REMAC 微调（见仓内 Usage）。

## 对 wiki 的映射

- [paper-remac](../../wiki/entities/paper-remac.md)
