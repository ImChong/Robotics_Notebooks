# Rethinking Video Storage for AI Training

> 来源归档（ingest）

- **论文：** <https://arxiv.org/abs/2610.33079>（v1，2026-10-21；全文及附录）
- **机构：** 北京大学、清华大学
- **项目页：** <https://skywalker-yqz.github.io/codec4AI/>；[站点归档](../sites/codec4ai.md)
- **代码：** <https://github.com/kaname-madoka18/codec4AI>；[仓库归档](../repos/codec4ai.md)
- **入库日期：** 2026-10-30
- **知识节点：** [codec4AI](../../wiki/entities/paper-codec4ai.md)

## 核心摘录与映射

- 稀疏解码解析参考图，对请求帧做依赖闭包，仅重建闭包内画面；其余包仍需维护解码器状态。映射：实体页的核心机制。
- DDRA 让区间内 B 帧仅引用两侧 I 帧，单目标至多重建三帧；共享锚点使批量平均开销更低。映射：实体页的流程与原理。
- 三个机器人数据集、四种采样方式、质量对齐的加载测试中，H.265 DDRA 比所选 LeRobot v3 基线吞吐高 1.45 倍、媒体载荷少 58.7%；该测试不含完整 RGB 跨进程传输、GPU 计算与离线转码。映射：实体页评测及限制。
