# Vast.ai

> 来源归档

- **标题：** Vast.ai — GPU Cloud Marketplace
- **类型：** site（国外 GPU 云算力 P2P 市场）
- **来源：** Vast.ai Inc.
- **链接：** https://vast.ai/ 、https://vast.ai/docs/
- **入库日期：** 2026-07-02
- **一句话说明：** 去中心化 GPU 租赁市场：主机方出租闲置算力，用户按市场竞价租卡；价格常为传统云 40–60% 折扣，但可靠性与硬件质量因主机而异。

## 为什么值得保留

- **本库已有引用**：`sources/repos/ppf-contact-solver.md` 提及 vast.ai 远程求解模板。
- **极致低价实验**：适合可 checkpoint 的 RL/渲染批任务。
- **与 RunPod/Lambda 互补**：成本优先 vs 可靠性优先的分工。

## 平台要点（公开资料 2026-07 归纳）

| 维度 | 要点 |
|------|------|
| **模式** | P2P 市场；按主机报价筛选 |
| **计费** | 通常按秒/按小时；可中断 |
| **模板** | 500+ Docker 模板（市场资料） |
| **可靠性** | 需看 reliability score；数据中心级主机与消费级主机混排 |
| **适用** | 短跑实验、批处理、可恢复训练 |
| **不适用** | 生产推理 SLA、不可中断长训（除非强 checkpoint） |

## 对 wiki 的映射

- 实体页：[vast-ai.md](../../wiki/entities/vast-ai.md)
- 统一选型：[international-gpu-cloud-platforms.md](../../wiki/comparisons/international-gpu-cloud-platforms.md)
