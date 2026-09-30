# openpi-rtc（EII-Tokyo）

> 来源归档（repo）

- **标题：** openpi-rtc
- **类型：** repo
- **链接：** <https://github.com/EII-Tokyo/openpi-rtc>
- **关联论文：** [real_time_chunking_arxiv_2506_07339.md](../papers/real_time_chunking_arxiv_2506_07339.md)
- **上游：** [openpi](openpi.md)（Physical Intelligence）
- **入库日期：** 2026-09-30
- **一句话说明：** 在 openpi 上实现推理期 RTC：Action Chunk Broker + `pi0.py` 的 guided_inference；ALOHA `pi0_aloha_pen_uncap` 报告拔笔任务完成时间约 10s→9s。

## 开源状态

- **已开源**：社区维护 fork，非 PI 官方；Docker + ALOHA udev 文档在 README。

## README 入口

| 模块 | 路径 |
|------|------|
| Chunk 调度 | `packages/openpi-client/src/openpi_client/action_chunk_broker.py` |
| 引导推理 | `src/openpi/models/pi0.py`（`guided_inference`） |

## 对 wiki 的映射

- [paper-real-time-chunking](../../wiki/entities/paper-real-time-chunking.md)
- [π0-policy](../../wiki/methods/π0-policy.md)
